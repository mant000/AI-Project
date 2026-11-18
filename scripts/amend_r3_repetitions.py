"""Preserve the initial pre-run metadata while increasing planned training repeats to three."""
from stage3_common import CODE, read, write, digest, stamp


def main():
    assert not (CODE/'results/training/grpo-v1').exists(), 'Amendment must precede all RL output'
    assert not (CODE/'results/main/discovery-v1').exists(), 'Amendment must precede main output'
    doc = CODE/'docs/R3_REPETITION_AMENDMENT.json'; amendment = read(doc)
    for phase, name, path in [('grpo', 'grpo_v1.json', 'results/training/grpo-prepare-v1'),
                              ('main', 'main_control_v1.json', 'results/main/prepare-v1')]:
        out = CODE/path; prior = read(out/'protocol_frozen.json')
        assert prior['protocol'] == amendment['old_protocols'][name]['protocol']
        assert digest(out/'bank_observables.jsonl') == read(out/'completed.json')['bank_sha256']
        backup = out/'protocol_before_repetition_amendment.json'; assert not backup.exists()
        (backup).write_bytes((out/'protocol_frozen.json').read_bytes())
        cfgpath = CODE/'configs'/name
        prior.update(protocol=read(cfgpath), protocol_sha256=digest(cfgpath),
            repetition_amendment_sha256=digest(doc), repetition_amended_at_utc=stamp(), bank_recomputed=False)
        write(out/'protocol_frozen.json', prior)
    print('Amended before all RL/main output; original metadata preserved, observable banks reused exactly.')


if __name__ == '__main__': main()
