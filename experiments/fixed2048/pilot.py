"""Gate progressively harder native-1080p fits using observed MSE reduction."""

import argparse
import copy
import json

from cp_adaptok.runtime import read_config, resolve, seed_everything, write_json
from experiments.fixed2048.train import run_training


def passed(history, threshold):
    initial = history['step0_train']['summary']['global_mse']
    actual = history['last_train_eval']['summary']['global_mse']
    return (initial - actual) / initial >= threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fixed2048.yaml')
    parser.add_argument('--lengths', type=int, nargs='+', default=[1,4,16])
    args = parser.parse_args()
    base = read_config(args.config)
    gates = []
    for length in args.lengths:
        config = copy.deepcopy(base)
        config['training'].update(train_video=base['pilot']['train_video'], sequence_length=length,
                                  steps=base['pilot']['steps'], lr=base['pilot']['lr'])
        directory = resolve(f'results/fixed2048/overfit/{length}frame')
        completion = directory / 'completion.json'
        if completion.exists() and json.loads(completion.read_text())['status'] == 'complete':
            history = json.loads((directory / 'history.json').read_text())
        else:
            checkpoint = resolve(f'checkpoints/fixed2048/continuous/overfit{length}/latest.pt')
            resume = checkpoint if checkpoint.exists() else None
            seed_everything(config['seed'], config['device'], 0.9)
            history = run_training(config, directory, checkpoint.parent, resume)
        success = passed(history, base['pilot']['minimum_relative_mse_drop'])
        gates.append({'frames':length, 'passed':success, 'actual_steps':len(history['train']),
                      'initial_mse':history['step0_train']['summary']['global_mse'],
                      'final_mse':history['last_train_eval']['summary']['global_mse']})
        write_json('results/fixed2048/overfit/gates.json', gates)
        if not success:
            raise RuntimeError('native single-sample fitting gate failed; stop later training and inspect gradients')


if __name__ == '__main__':
    main()
