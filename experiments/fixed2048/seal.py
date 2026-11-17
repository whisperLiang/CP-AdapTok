"""Freeze validation-selected checkpoint identities and the final test protocol."""

import argparse
import json

from cp_adaptok.runtime import metadata, read_config, resolve, write_json
from experiments.fixed2048.checkpoint import require_checkpoint, sha256


def seal_experiment(config, plan, output):
    output = resolve(output)
    if output.exists():
        raise FileExistsError('test seal is immutable; retain the existing sealed protocol')
    protocols, entries = {}, {}
    for name, entry in plan.items():
        checkpoint = require_checkpoint(entry['checkpoint'])
        digest = sha256(checkpoint)
        protocol = {'paths':entry['paths'], 'references':entry['references']}
        if digest in protocols and protocols[digest] != protocol:
            raise ValueError('one checkpoint must have one complete declared final protocol')
        protocols[digest] = protocol
        entries[name] = dict(entry, sha256=digest, checkpoint=str(checkpoint))
    result = {'checkpoint_sha256s':sorted(protocols), 'protocols':protocols, 'models':entries,
              'test_config':config['final_test'],
              'test_manifest_sha256':sha256(resolve(config['manifest_dir'])/'cpat_test.json'),
              'no_training_or_selection_after_seal':True, **metadata(config,'sealed_before_test')}
    write_json(output,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/fixed2048.yaml')
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',default='results/fixed2048/final_test/seal.json')
    args = parser.parse_args()
    seal_experiment(read_config(args.config),json.loads(resolve(args.plan).read_text()),args.output)


if __name__ == '__main__':
    main()
