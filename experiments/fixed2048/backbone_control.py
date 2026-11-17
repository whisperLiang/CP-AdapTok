"""Separate continuous I-latent scale control grounded in official DCVC code.

The task's R0 raw-y definition is retained in codec.py. Official image prior
multiplies by q_scale_y_enc before reconstruction and q_scale_y_dec afterwards.
Bypassing rounding must retain both factors. This control does not call priors,
entropy models, native compression, or any bitstream method.
"""

import argparse

import torch

from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import read_config, resolve, seed_everything, write_json
from experiments.fixed2048.checkpoint import load_reconstructor
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.diagnostics import aggregate, distortion
from experiments.fixed2048.evaluate import FLAGS, save_image


@torch.no_grad()
def run_control(config, output, frames=8):
    directory = resolve(output)
    if directory.exists():
        raise FileExistsError('scale-control output exists')
    model, identity = load_reconstructor(config['source_checkpoint'], config['device'])
    suite = MetricSuite(model.device)
    if suite.errors:
        raise RuntimeError(f'required metrics unavailable: {suite.errors}')
    selected, hashes = records(config, 'validation', config['baseline']['videos'])
    scales = model.intra.quant('q_scale_y_enc', model.config.qp) * model.intra.quant('q_scale_y_dec', model.config.qp)
    rows = []
    for source in selected:
        clip = read_clip(source, frames)
        for reference in ('i_only', 'full2048'):
            state = model.empty_state()
            for index, frame in enumerate(clip):
                rgb = frame[None].to(model.device)
                encoded = model.encode(rgb, state, index, force_intra=reference == 'i_only')
                latent = encoded.latent * scales if encoded.context.frame_type == 'I' else encoded.latent
                rec, feature = model.decode_latent(latent, encoded.context)
                rows.append({'video':source['content_id'], 'frame':index, 'frame_type':encoded.context.frame_type,
                             'reference':reference, 'path':'R0-scale-control', 'pixels':1080*1920,
                             **suite(rgb, rec), **distortion(encoded.latent, latent), **FLAGS})
                if index == 0:
                    save_image(directory / 'images' / f'{source["content_id"].split(":")[-1]}_{reference}.png', rec)
                if reference == 'full2048':
                    state = encoded.context.commit(feature, f'scale-control:{index}')
    write_json(directory / 'frame_metrics.json', rows)
    write_json(directory / 'summary.json', {reference:aggregate([r for r in rows if r['reference'] == reference])
                                           for reference in ('i_only','full2048')})
    write_json(directory / 'run.json', {'checkpoint':identity, 'manifest_hashes':hashes, 'frames':len(rows),
               'scales_min':scales.min().item(), 'scales_max':scales.max().item(),
               'official_source':'third_party/DCVC/src/models/common_model.py:forward_prior_4x',
               'note':'retains both I-latent scale factors, bypasses all rounding; separate from raw-y R0', **FLAGS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fixed2048.yaml')
    parser.add_argument('--output', default='results/fixed2048/baseline_scale_control')
    parser.add_argument('--frames', type=int, default=8)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config['seed'], config['device'], 0.9)
    run_control(config, args.output, args.frames)


if __name__ == '__main__':
    main()
