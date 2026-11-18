"""Recreate standalone SFT diagnostics from actual events; no route-quality claim."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_common import CODE, read, write, digest, stamp


def main():
    source = CODE/'results/training/sft-v1'; assert read(source/'verification.json')['status'] == 'passed'
    out = CODE/'results/training/sft-analysis-v1'; out.mkdir(exist_ok=True)
    events = [json.loads(x) for x in (source/'update_events.jsonl').read_text().splitlines()]
    train = [r for r in events if r['kind'] == 'training']
    vals = [read(source/f'validation_epoch{e}.json')['mean_response_nll'] for e in range(3)]
    heldout = read(source/'heldout_nll.json')['mean_response_nll']
    with (out/'nll.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['split', 'epoch', 'token_weighted_response_nll'])
        for e, nll in enumerate(vals): writer.writerow(['validation', e, nll])
        writer.writerow(['heldout_after_freeze', read(source/'checkpoint_frozen.json')['selected_epoch'], heldout])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9, 'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), layout='constrained')
    ax[0].plot([r['logical_step'] for r in train], [r['mean_example_nll'] for r in train], lw=.8, color='#2a6fbb')
    ax[0].set(xlabel='Logical optimizer update', ylabel='Mean example response NLL', title='Executed grammar training')
    ax[0].axvline(77.5, color='gray', linestyle='--', lw=.7)
    ax[1].plot(range(3), vals, marker='o', color='#2a6fbb', label='Family validation (14 examples)')
    chosen = read(source/'checkpoint_frozen.json')['selected_epoch']
    ax[1].scatter([chosen], [heldout], marker='x', s=60, color='#c25f22', label='Family heldout after freeze (40)')
    ax[1].set(xlabel='Epoch', ylabel='Token-weighted response NLL', title='Checkpoint selection and diagnostic', xticks=range(3))
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax: a.grid(axis='y', alpha=.15)
    for suffix in ('png', 'pdf', 'svg'): fig.savefig(out/f'sft_diagnostics.{suffix}', dpi=180)
    plt.close(fig)
    write(out/'summary.json', dict(at_utc=stamp(), selected_epoch=chosen, validation_nll=vals, family_heldout_nll=heldout,
        source_verification_sha256=digest(source/'verification.json'),
        interpretation='Program imitation only; shared discovery instances, small family validation, not route-quality evidence', final_opened=False))


if __name__ == '__main__': main()
