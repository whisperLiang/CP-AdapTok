"""Actual native-1080p inference FLOPs/latency/memory, excluding metric networks."""

import argparse
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from cp_adaptok.runtime import read_config, seed_everything, write_json
from experiments.fixed2048.checkpoint import load_reconstructor
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.evaluate import FLAGS, stage_times


@torch.no_grad()
def profile_model(config, checkpoint, path, repeats=10):
    model, identity = load_reconstructor(checkpoint, config['device'], profile=True)
    sources, _ = records(config, 'validation', config['validation']['videos'][:1])
    clip = read_clip(sources[0], 2).to(model.device)
    setup = model.encode(clip[0:1], model.empty_state(), 0)
    setup_output = model.reconstruct(setup, path)
    reference = model.advance(setup, setup_output, 'full2048')
    rows = []
    for frame_type, rgb, state, index in (('I',clip[0:1],model.empty_state(),0), ('P',clip[1:2],reference,1)):
        for attempt in range(3 + repeats):
            torch.cuda.reset_peak_memory_stats(model.device)
            torch.cuda.synchronize(model.device)
            start = time.perf_counter()
            encoded = model.encode(rgb, state, index)
            output = model.reconstruct(encoded, path)
            torch.cuda.synchronize(model.device)
            if attempt >= 3:
                rows.append({'frame_type':frame_type, 'repeat':attempt-3,
                             'wall_ms':(time.perf_counter()-start)*1000,
                             'peak_gpu_mib':torch.cuda.max_memory_allocated(model.device)/2**20,
                             **stage_times(encoded,output,path)})
        model.profile = False
        with FlopCounterMode(display=False) as counter:
            encoded = model.encode(rgb, state, index)
            model.reconstruct(encoded,path)
        flops = counter.get_total_flops()
        for row in rows:
            if row['frame_type'] == frame_type:
                row['flops'] = flops
        model.profile = True
    return {'checkpoint':identity, 'path':path, 'resolution':[1080,1920], 'rows':rows,
            'warmup_repeats':3, 'measured_repeats':repeats,
            'flop_convention':'PyTorch FlopCounterMode; multiply+add=2; unsupported non-arithmetic ops omitted',
            'metric_networks_present':False, **FLAGS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/fixed2048.yaml')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--path',choices=['R0','R1','R2','raw_192'],default='R1')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config['seed'],config['device'],0.9)
    write_json(args.output,profile_model(config,args.checkpoint,args.path))


if __name__ == '__main__':
    main()
