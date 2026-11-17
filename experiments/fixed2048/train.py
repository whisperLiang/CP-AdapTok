"""Small fixed-token training loop; exact resume at completed optimizer boundaries."""

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from cp_adaptok.runtime import read_config, resolve, seed_everything, write_json
from experiments.fixed2048.checkpoint import load_reconstructor, resume_training, save_training
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.diagnostics import aggregate, distortion, module_norms, psnr
from experiments.fixed2048.evaluate import FLAGS, save_image


def parameter_snapshot(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def update_norms(model, before):
    groups = {'tokenizer': 'tokenizer.', 'project_in': 'vq.project_in.', 'project_out': 'vq.project_out.',
              'decoder': 'decoder.', 'codebook': 'vq.codebook.'}
    squared = {group: 0.0 for group in groups}
    for name, parameter in model.named_parameters():
        if name in before:
            prefix = next(key for key, value in groups.items() if name.startswith(value))
            squared[prefix] += (parameter.detach() - before[name]).float().square().sum().item()
    return {key: value ** 0.5 for key, value in squared.items()}


def training_frame(model, frame, state, index, settings):
    rgb = frame[None].to(model.device)
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16,
                        enabled=settings['amp_bf16'] and model.device.type == 'cuda'):
        encoded = model.encode(rgb, state, index)
        output = model.reconstruct(encoded, model.training_path)
        mse = F.mse_loss(output.rgb.float(), rgb.float())
        loss = mse + (output.vq_loss if model.training_path == 'R2' else 0)
        state = model.advance(encoded, output, 'full2048')
    row = {'mse': mse.detach().item(), 'psnr': psnr(mse.detach().item()),
           'frame_type': encoded.context.frame_type, 'frame': index,
           'vq_loss': output.vq_loss.detach().item(),
           'output_variance': output.rgb.detach().float().var(unbiased=False).item(),
           'token_variance': encoded.features.detach().float().var(dim=1, unbiased=False).mean().item(),
           **distortion(encoded.latent.detach(), output.latent.detach())}
    return loss, state, row


def backward_clip(model, clip, settings):
    state, pending, rows = model.empty_state(), [], []
    for index, frame in enumerate(clip):
        loss, state, row = training_frame(model, frame, state, index, settings)
        pending.append(loss / len(clip))
        rows.append(row)
        if (index + 1) % settings['tbptt'] == 0 or index + 1 == len(clip):
            torch.stack(pending).sum().backward()
            state, pending = state.detach(), []
    return rows


def frozen_gradient_count(model):
    return sum(p.grad is not None for m in (model.intra, model.ld) for p in m.parameters())


def train_update(model, clip, optimizer, settings):
    if settings['tbptt'] < 1:
        raise ValueError('TBPTT must be positive')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = parameter_snapshot(model)
    start = time.perf_counter()
    rows = backward_clip(model, clip, settings)
    gradients = module_norms(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, settings['clip_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)
    mse = sum(r['mse'] for r in rows) / len(rows)
    frozen_grads = frozen_gradient_count(model)
    if frozen_grads:
        raise RuntimeError('frozen backbone received parameter gradients')
    return {'mse': mse, 'psnr': psnr(mse), 'latent_mse': sum(r['latent_mse'] for r in rows) / len(rows),
            'gradient_norm_before_clip': norm.item(), 'module_gradient_norm': gradients,
            'module_update_norm': update_norms(model, before), 'frames': rows,
            'frozen_backbone_gradients': frozen_grads, 'step_wall_s': time.perf_counter() - start,
            'peak_gpu_mib': torch.cuda.max_memory_allocated(model.device)/2**20 if model.device.type == 'cuda' else 0}


@torch.no_grad()
def validate(model, selected, length, directory=None, label='validation'):
    rows = []
    for row in selected:
        clip = read_clip(row, length)
        _, measured = validate_clip(model, clip, row['content_id'], directory,
                                    f'{label}_{Path(row["path"]).name}')
        rows.extend(measured)
    return aggregate(rows), rows


def training_sources(config):
    sources, hashes = records(config, 'train', exact=False)
    h, w = config['training']['resolution']
    sources = [r for r in sources if r['resolution'][0] >= w and r['resolution'][1] >= h]
    if config['training'].get('train_video'):
        sources = [r for r in sources if r['content_id'] == config['training']['train_video']]
    if not sources:
        raise ValueError('no eligible Train Split source for the declared route')
    return sources, hashes


def sample_clip(sources, length, generator):
    source = sources[int(torch.randint(len(sources), (), generator=generator))]
    start = int(torch.randint(source['original_frames'] - length + 1, (), generator=generator))
    return source, start, read_clip(source, length, start, crop=(1080, 1920))


def reconcile_log(path, completed):
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    kept = [row for row in rows if row['step'] <= completed]
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in kept))
    temporary.replace(path)


@dataclass
class TrainingRun:
    config: dict
    directory: Path
    checkpoints: Path
    model: object
    identity: dict
    optimizer: object
    scheduler: object
    generator: object
    sources: list
    validation: list
    history: dict
    step: int = 0

    def save(self, filename, step):
        save_training(self.checkpoints / filename, self.model, self.optimizer, self.scheduler,
                      self.generator, step, self.config, self.history, self.identity)


def require_training_config(config):
    expected = {'mse':1.0, 'latent':0.0, 'lpips':0.0, 'dists':0.0, 'monotonic':0.0, 'estimated_rate':0.0}
    if config['training']['reference'] != 'full2048' or config['loss'] != expected:
        raise ValueError('this controlled experiment requires full2048 reference and RGB MSE only')
    if min(config['training'][key] for key in ('steps','sequence_length','save_every','validate_every')) < 1:
        raise ValueError('training counts and intervals must be positive')


def initialize_training(config, output, checkpoint_dir, resume):
    directory, checkpoints = resolve(output), resolve(checkpoint_dir)
    if (directory / 'config.json').exists() and not resume:
        raise FileExistsError('training output exists; resume explicitly or use a fresh directory')
    require_training_config(config)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    settings = config['training']
    model, identity = load_reconstructor(config['source_checkpoint'], config['device'],
                                         config['decoder_grid'], config['encoder_kv_grid'])
    model.configure_training(settings['path'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.0)
    generator = torch.Generator().manual_seed(config['seed'])
    sources, hashes = training_sources(config)
    validation, _ = records(config, 'validation', config['validation']['videos'])
    run = TrainingRun(config, directory, checkpoints, model, identity, optimizer, scheduler,
                      generator, sources, validation, {'train':[], 'validation':[]})
    if resume:
        run.step, run.history, run.identity = resume_training(resume, model, optimizer, scheduler, generator, config)
    else:
        initialize_step_zero(run)
    write_json(directory / 'config.json', config)
    write_json(directory / 'inputs.json', {'source_identity':run.identity, 'manifest_hashes':hashes,
               'eligible_train_videos':[r['content_id'] for r in sources], **FLAGS})
    reconcile_log(directory / 'train.jsonl', run.step)
    return run


def initialize_step_zero(run):
    config, directory, model = run.config, run.directory, run.model
    initial, frames = validate(model, run.validation, config['validation']['frames'], directory / 'images', 'step0_val')
    run.history['validation'].append({'step':0, 'summary':initial, 'frames':frames})
    source = run.sources[0]
    clip = read_clip(source, config['training']['sequence_length'], crop=(1080,1920))
    initial, frames = validate_clip(model, clip, source['content_id'], directory / 'images', 'step0_train')
    run.history['step0_train'] = {'summary':initial, 'frames':frames}
    for filename in ('step0.pt', 'best_validation.pt', 'latest.pt'):
        run.save(filename, 0)


def validate_update(run, completed, clip, source):
    result, frames = validate(run.model, run.validation, run.config['validation']['frames'],
                              run.directory / 'images', f'step{completed}_val')
    previous = min(v['summary']['global_mse'] for v in run.history['validation'])
    run.history['validation'].append({'step':completed, 'summary':result, 'frames':frames})
    measured, measured_frames = validate_clip(run.model, clip, source['content_id'],
                                              run.directory / 'images', f'step{completed}_train')
    run.history['last_train_eval'] = {'step':completed, 'summary':measured, 'frames':measured_frames}
    if result['global_mse'] <= previous:
        run.save('best_validation.pt', completed)


def due(completed, interval, limit):
    return completed % interval == 0 or completed == limit


def execute_updates(run, limit, cached):
    settings = run.config['training']
    with (run.directory / 'train.jsonl').open('a') as log:
        for completed in range(run.step + 1, limit + 1):
            source, start, clip = cached or sample_clip(run.sources, settings['sequence_length'], run.generator)
            measured = train_update(run.model, clip, run.optimizer, settings)
            run.scheduler.step()
            measured.update(step=completed, sample={'video':source['content_id'], 'start':start,
                            'frames':len(clip)}, lr=run.optimizer.param_groups[0]['lr'], **FLAGS)
            run.history['train'].append(measured)
            log.write(json.dumps(measured, ensure_ascii=False, allow_nan=False) + '\n')
            log.flush()
            if due(completed, settings['validate_every'], limit):
                validate_update(run, completed, clip, source)
            if due(completed, settings['save_every'], limit):
                run.save('latest.pt', completed)
                write_json(run.directory / 'history.json', run.history)
            if completed == 1 or completed % 10 == 0:
                print(json.dumps({k:measured[k] for k in
                      ('step','mse','psnr','latent_mse','step_wall_s','peak_gpu_mib')}), flush=True)


def record_completion(run):
    history = run.history
    actual = history['train'][-1]['step'] if history['train'] else run.step
    write_json(run.directory / 'completion.json', {
               'status':'complete' if actual == run.config['training']['steps'] else 'pilot_checkpointed',
               'actual_optimizer_steps':actual, 'declared_steps':run.config['training']['steps'],
               'actual_training_frames':sum(len(r['frames']) for r in history['train']),
               'optimizer_wall_s':sum(r['step_wall_s'] for r in history['train']), **FLAGS})


def run_training(config, output, checkpoint_dir, resume=None, stop_after=None):
    run = initialize_training(config, output, checkpoint_dir, resume)
    settings = config['training']
    limit = min(settings['steps'], stop_after) if stop_after is not None else settings['steps']
    cached = None
    if settings.get('train_video'):
        cached = (run.sources[0], 0, read_clip(run.sources[0], settings['sequence_length'], crop=(1080,1920)))
    if run.model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(run.model.device)
    execute_updates(run, limit, cached)
    record_completion(run)
    return run.history


@torch.no_grad()
def validate_clip(model, clip, video, directory=None, label='train'):
    model.eval()
    state, rows = model.empty_state(), []
    for index, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode(rgb, state, index)
        output = model.reconstruct(encoded, model.training_path)
        mse = F.mse_loss(output.rgb.float(), rgb).item()
        rows.append({'video':video, 'frame':index, 'frame_type':encoded.context.frame_type,
                     'mse':mse, 'psnr':psnr(mse), 'pixels':rgb.shape[-2]*rgb.shape[-1],
                     **distortion(encoded.latent, output.latent)})
        state = model.advance(encoded, output, 'full2048')
        if directory and index in (0, len(clip)-1):
            save_image(Path(directory) / f'{label}_{index:03}.png', output.rgb)
    return aggregate(rows), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fixed2048.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--pilot-frames', type=int, choices=[1,4,16])
    parser.add_argument('--path', choices=['R1','R2','raw_192'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--decoder-grid', type=int, nargs=2)
    parser.add_argument('--encoder-kv-grid', type=int, nargs=2)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    config = copy.deepcopy(read_config(args.config))
    if args.pilot_frames:
        config['training'].update(train_video=config['pilot']['train_video'], sequence_length=args.pilot_frames,
                                  steps=config['pilot']['steps'], lr=config['pilot']['lr'])
    for key, value in (('steps',args.steps), ('path',args.path)):
        if value is not None:
            config['training'][key] = value
    for key, value in (('source_checkpoint',args.source_checkpoint), ('decoder_grid',args.decoder_grid),
                       ('encoder_kv_grid',args.encoder_kv_grid)):
        if value is not None:
            config[key] = value
    seed_everything(config['seed'], config['device'], 0.9)
    run_training(config, args.output, args.checkpoint_dir, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
