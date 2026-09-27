"""Read-only checkpoint comparison; write analysis artifacts outside the repository."""
from pathlib import Path
import json
import os
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import torch
from challenge_lib import paths, data as bl_data
from challenge_lib.baseline_model import Encoder as OriginalEncoder, Decoder as OriginalDecoder
from iteration_1_residual import train_residual_betavae as evaluation
from iteration_1_residual.model import Encoder as ResidualEncoder, Decoder as ResidualDecoder
paths.configure_matplotlib()
OUTPUT = paths.ANALYSIS_DIR / "iteration_1_residual"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    evaluation.DEVICE = torch.device('cpu')
    data = bl_data.BLPlanes()
    runs = [
        ('Original, latent 30', paths.experiment_results('baseline') / 'lat30_beta0_ep500_ntr4500', False),
        ('Residual, latent 30', paths.experiment_results('iteration_1_residual') / 'lat30_beta0_ep500_ntr4500', True),
        ('Original, latent 50 / 1000 epochs', paths.experiment_results('baseline') / 'lat50_beta0_ep1000_ntr4500', False),
    ]
    records = []
    reference = None
    for label, folder, residual in runs:
        checkpoint = next(folder.glob('model_*.pt'))
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if reference is None:
            reference = {key: ck[key] for key in ('tr_idx', 'val_idx', 'mu_C', 'std_C', 'fluc_mean')}
        assert all(np.array_equal(value, ck[key]) for key, value in reference.items())
        Enc, Dec = (ResidualEncoder, ResidualDecoder) if residual else (OriginalEncoder, OriginalDecoder)
        enc = Enc(ck['H'], ck['W'], ck['C'], ck['latent_dim'])
        dec = Dec(ck['H'], ck['W'], ck['C'], ck['latent_dim'])
        enc.load_state_dict(ck['enc_state'])
        dec.load_state_dict(ck['dec_state'])
        print(f'Evaluating {label}: {len(ck["val_idx"])} snapshots on CPU...', flush=True)
        start = time.perf_counter()
        metrics = evaluation.evaluate(enc, dec, data, ck['val_idx'], ck['mu_C'], ck['std_C'],
                                      ck['fluc_mean'], ck['beta'], batch=8)
        print(f'{label}: rootNMSE={metrics["rootNMSE"]:.8f}; Ek={metrics["Ek"]:.5f}; {time.perf_counter()-start:.1f}s', flush=True)
        saved_metrics = json.loads(next(folder.glob('metrics_*.json')).read_text())
        params = json.loads((folder / 'parameters.json').read_text())
        curve = []
        pattern = r'Ep\s+(\d+)/\d+ \| tr ([\d.]+) \| va ([\d.]+) \| rootNMSE ([\d.]+)'
        for match in re.finditer(pattern, (folder / 'train.log').read_text()):
            epoch, train, val, rn = match.groups()
            curve.append({'epoch': int(epoch), 'train': float(train), 'val': float(val), 'rootNMSE': float(rn)})
        records.append({'label': label, 'checkpoint': str(checkpoint), 'best_epoch': ck['best_epoch'],
                        'same_split_and_normalisation': True, 'parameters': params, 'common_cpu_metrics': metrics,
                        'saved_metrics': saved_metrics, 'logged_curve': curve})
        (OUTPUT / 'comparison.json').write_text(json.dumps(records, indent=2) + '\n')
        del ck, enc, dec
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#727b8b', '#007e80', '#b37a1c']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for record, color in zip(records, colors):
        curve = record['logged_curve']
        axes[0].plot([x['epoch'] for x in curve], [x['rootNMSE']*100 for x in curve],
                     marker='o', markersize=3, label=record['label'], color=color)
    axes[0].set(xlabel='Epoch', ylabel='Validation root-NMSE (%)', title='Learning curves from training logs')
    axes[0].legend(fontsize=8)
    positions = np.arange(4)
    for i, (record, color) in enumerate(zip(records, colors)):
        m = record['common_cpu_metrics']
        values = [m['rootNMSE']] + [m['rootNMSE_comp'][c] for c in ['u', 'v', 'w']]
        axes[1].bar(positions + (i-1)*.25, np.array(values)*100, width=.24, color=color)
    axes[1].set(xticks=positions, xticklabels=['Overall', 'u', 'v', 'w'], ylabel='Validation root-NMSE (%)',
                title='Best checkpoints: same CPU evaluation')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Challenge 1.1: residual architecture comparison — lower error is better')
    fig.savefig(OUTPUT / 'comparison.png', dpi=160)
    plt.close(fig)
    a, b = [r['common_cpu_metrics'] for r in records[:2]]
    print('Root-NMSE reduction (%):',100*(1-b['rootNMSE']/a['rootNMSE']))
    print('Squared-error reduction (%):',100*(1-b['NMSE']/a['NMSE']))
    print('Ek increase (percentage points):',b['Ek']-a['Ek'])


if __name__ == '__main__':
    main()
