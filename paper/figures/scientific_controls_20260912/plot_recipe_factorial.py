"""Render the standardisation-factorial comparison from ten-seed summaries."""
from pathlib import Path
import csv
import hashlib
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
RESULTS_DIR = Path(os.environ.get('NV_RESULTS_DIR', HERE)).expanduser()
OUTPUT_ROOT = Path(os.environ.get('NV_OUTPUT_DIR', CODE_ROOT / 'outputs')).expanduser()
FIGURE_OUT = OUTPUT_ROOT / 'figures'
PREVIEW_OUT = OUTPUT_ROOT / 'previews'
BLUE, RED, INK = '#285761', '#B55847', '#202326'


def interaction_preview():
    """Plot MLP8-minus-CNN gaps with paired-acquisition bootstrap intervals."""
    import pandas as pd
    from matplotlib import font_manager

    approved_export = sys.argv[1:] == ['--approved-export']
    output_dir = FIGURE_OUT if approved_export else PREVIEW_OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    font_manager.findfont('Arial', fallback_to_default=False)
    paths = [RESULTS_DIR / name for name in ('recipe_full.csv', 'recipe_additional.csv',
                                            'recipe_summary.csv', 'recipe_factorial_effects.csv')]
    fingerprints = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    summary = pd.read_csv(paths[2])
    effects = pd.read_csv(paths[3])
    conditions, epochs = ((0, 0), (0, 1), (1, 0), (1, 1)), (1500, 3000)
    folds, seeds = tuple(f'A{i}' for i in range(1, 9)), set(range(20260804, 20260814))
    expected = [(a, b, e) for a, b in conditions for e in epochs]
    assert list(summary[['input_standardised', 'target_standardised', 'epoch']]
                .itertuples(index=False, name=None)) == expected
    assert summary.acquisitions.eq(8).all() and summary.optimisation_seeds.eq(10).all()
    for field, path in zip(('old_results_sha256', 'additional_results_sha256'), paths[:2]):
        assert summary[field].str.lower().eq(fingerprints[path]).all(), 'Frozen results drift'
        assert effects[field].str.lower().eq(fingerprints[path]).all(), 'Contrast results drift'

    columns = ['fold', 'model', 'seed', 'input_standardised', 'target_standardised',
               'epoch', 'row_index', 'error_um']
    old, new = (pd.read_csv(path, usecols=columns) for path in paths[:2])
    assert (len(old), len(new)) == (39312, 91728)
    data = pd.concat((old, new), ignore_index=True)
    keys = columns[:-2]
    assert not data.duplicated(keys + ['row_index']).any()
    assert set(data.seed) == seeds and set(data.fold) == set(folds)
    assert set(data.model) == {'cnn', 'mlp8'} and set(data.epoch) == set(epochs)
    assert np.isfinite(data.error_um).all()
    endpoint = data.groupby(keys, sort=True).error_um.mean()
    assert len(endpoint) == 1280
    assert endpoint.groupby(level=['fold', 'model', 'input_standardised',
                                   'target_standardised', 'epoch']).size().eq(10).all()
    acquisition = endpoint.groupby(level=['fold', 'model', 'input_standardised',
                                         'target_standardised', 'epoch']).mean()
    assert len(acquisition) == 128
    paired = acquisition.unstack('model')
    assert len(paired) == 64 and not paired.isna().any().any()

    draws = np.random.default_rng(20260930).integers(0, 8, size=(100000, 8))
    draw_hash = hashlib.sha256(draws.astype(np.int64, copy=False).tobytes()).hexdigest()
    assert effects.resample_indices_sha256.eq(draw_hash).all()
    assert summary.bootstrap_seed.eq(20260930).all() and summary.bootstrap_draws.eq(100000).all()
    groups = []
    for row in summary.itertuples(index=False):
        a, b, e = row.input_standardised, row.target_standardised, row.epoch
        values = np.array([[paired.loc[(fold, a, b, e), model]
                            for model in ('cnn', 'mlp8')] for fold in folds])
        gap = values[:, 1] - values[:, 0]
        boot = gap[draws].mean(axis=1)
        for i, name in enumerate(('cnn', 'mlp8')):
            assert np.isclose(values[:, i].mean(), getattr(row, name + '_mean'),
                              rtol=0, atol=1e-10)
            ci = np.quantile(values[:, i][draws].mean(axis=1), [.025, .975])
            assert np.allclose(ci, [getattr(row, name + '_ci_low'),
                                   getattr(row, name + '_ci_high')], rtol=0, atol=1e-10)
        assert np.isclose(gap.mean(), row.d_um_mean, rtol=0, atol=1e-10)
        assert np.allclose(np.quantile(boot, [.025, .975]),
                           [row.d_um_ci_low, row.d_um_ci_high], rtol=0, atol=1e-10)
        groups.append(row)

    ink = '#202020'
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'font.size': 10.2, 'axes.labelsize': 10.8, 'axes.linewidth': .85,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': ink, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.4, 'ytick.major.size': 3.4,
        'xtick.major.width': .85, 'ytick.major.width': .85,
        'text.color': ink, 'axes.labelcolor': ink,
        'xtick.color': ink, 'ytick.color': ink,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig = plt.figure(figsize=(150 / 25.4, 88 / 25.4))
    lookup = {(row.input_standardised, row.target_standardised, row.epoch): row
              for row in groups}
    assert len(lookup) == 8
    recipe_styles = (
        ((0, 0), 'Neither', '#168CBB'),
        ((0, 1), 'Target only', '#ED1746'),
        ((1, 0), 'Input only', '#F0A600'),
        ((1, 1), 'Both', '#9063B5'),
    )
    markers_drawn, intervals_drawn = 0, 0
    for column, (left, epoch) in enumerate(((.13, 1500), (.615, 3000))):
        ax = fig.add_axes([left, .315, .36, .614])
        ax.set_xlim(-.55, 3.55)
        ax.set_ylim(-20, 125)
        ax.set_xticks(range(4), [style[1] for style in recipe_styles],
                      rotation=35, ha='right', rotation_mode='anchor')
        ax.set_yticks([0, 40, 80, 120])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(pad=3.3, labelsize=10.2)
        ax.grid(False)
        ax.axhline(0, color='#D1D1D1', lw=.6, zorder=0)
        if column == 0:
            ax.set_ylabel('MLP8 − CNN error (μm)', labelpad=6.0)
        fig.text(left + .18, .039, f'({"ab"[column]}) {epoch:,} epochs',
                 ha='center', va='center', fontsize=10.4)
        for x, ((input_state, target_state), _, color) in enumerate(recipe_styles):
            row = lookup[(input_state, target_state, epoch)]
            mean, low, high = row.d_um_mean, row.d_um_ci_low, row.d_um_ci_high
            assert np.isfinite([mean, low, high]).all()
            assert ax.get_ylim()[0] < low <= mean <= high < ax.get_ylim()[1]
            interval = ax.errorbar(x, mean, yerr=[[mean-low], [high-mean]],
                                  fmt='o', linestyle='none', color=color,
                                  ecolor=color, elinewidth=1.15, ms=4.8,
                                  mfc='white', mec=color, mew=1.1,
                                  capsize=2.0, capthick=1.15, zorder=3)
            assert np.isclose(interval.lines[0].get_ydata()[0], row.d_um_mean)
            assert np.allclose(interval.lines[2][0].get_segments()[0][:, 1],
                               [row.d_um_ci_low, row.d_um_ci_high], rtol=0, atol=1e-12)
            markers_drawn += 1
            intervals_drawn += 1
        assert len(ax.patches) == 0 and len(ax.containers) == 4
    assert markers_drawn == intervals_drawn == 8 and len(fig.axes) == 2
    assert not fig.legends
    fig.text(.55, .118, 'Standardisation', ha='center', va='center', fontsize=10.8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Verify that all visible text lies within the figure canvas.
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
    output = output_dir / 'study_l_recipe_factorial.png'
    fig.savefig(output, dpi=450, facecolor='white')
    if approved_export:
        mpl.rcParams['pdf.fonttype'] = 42
        fig.savefig(output.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == fingerprints[path]
               for path in paths), 'An immutable scientific input changed'
    print('Preview:', output)
    print('8 frozen mean gaps and 8 pointwise paired-acquisition 95% CI displayed.')
    print('Existing bootstrap draws:', draw_hash)
    print('Frozen means, CI and source hashes verified; ' +
          ('approved PNG/PDF exported.' if approved_export else 'manuscript assets unchanged.'))

def main():
    with (RESULTS_DIR/'recipe_summary.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert [(int(r['input_standardised']), int(r['target_standardised']), int(r['epoch']))
            for r in rows] == [(a,b,e) for a,b in ((0,0),(0,1),(1,0),(1,1)) for e in (1500,3000)]
    mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                         'font.size':9.3,'axes.labelsize':10,'axes.linewidth':.75,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'xtick.direction':'out','ytick.direction':'out',
                         'xtick.major.size':3.5,'ytick.major.size':3.5,
                         'xtick.major.width':.75,'ytick.major.width':.75,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, (a,b) = plt.subplots(2,1,figsize=(183/25.4,5.2),sharex=True,
                              gridspec_kw={'height_ratios':[1.15,1]})
    fig.subplots_adjust(left=.105,right=.985,bottom=.105,top=.85,hspace=.22)
    x = np.array([0,.68,1.83,2.51,3.66,4.34,5.49,6.17])
    for ax in (a,b):
        ax.set_xlim(-.48,6.65)
        for sep in (1.255,3.085,4.915): ax.axvline(sep,color='#E5E9EB',lw=.7,zorder=0)
    for i,(si,st) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        center=(x[2*i]+x[2*i+1])/2
        a.text(center,1.045,f'Input {"on" if si else "off"}\nTarget {"on" if st else "off"}',
               transform=a.get_xaxis_transform(),ha='center',va='bottom',linespacing=1.15,fontsize=9.1)
    for model,color,marker,offset,label in [('cnn',BLUE,'o',-.10,'CNN'),('mlp8',RED,'s',.10,'MLP8')]:
        mean=np.array([float(r[model+'_mean']) for r in rows])
        lo=np.array([float(r[model+'_ci_low']) for r in rows])
        hi=np.array([float(r[model+'_ci_high']) for r in rows])
        a.errorbar(x+offset,mean,yerr=[mean-lo,hi-mean],fmt=marker,ms=5.0,
                   color=color,mfc=color,mec=INK,mew=.55,lw=.9,capsize=2.1,label=label,zorder=3)
    a.set_yscale('log');a.set_ylim(30,185)
    a.set_yticks([30,40,60,100,160],['30','40','60','100','160'])
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel('LOAO error (μm; log scale)',labelpad=7)
    a.tick_params(axis='x',bottom=False)
    a.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.37),frameon=False,
             columnspacing=2.0,handletextpad=.4)
    b.axhline(0,color='#8A9297',lw=.95,ls=(0,(3,3)),zorder=1)
    for xi,row in zip(x,rows):
        mean=float(row['d_um_mean']);lo=float(row['d_um_ci_low']);hi=float(row['d_um_ci_high'])
        color='#4B5054'
        b.errorbar(xi,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',ms=5.1,
                   color=color,mfc=color,mec=INK,mew=.55,lw=.95,capsize=2.2,zorder=3)
    b.set_ylim(-20,126);b.set_yticks([-20,0,20,40,60,80,100,120])
    b.set_ylabel('MLP8 − CNN error (μm)',labelpad=7)
    b.set_xticks(x,['1,500','3,000']*4);b.set_xlabel('Training epochs',labelpad=8)
    for ax,lab in ((a,'a'),(b,'b')):
        ax.text(-.10,1.03,lab,transform=ax.transAxes,fontweight='bold',fontsize=10,
                ha='left',va='bottom')
    FIGURE_OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ('.pdf','.png'):
        fig.savefig(FIGURE_OUT/('study_l_recipe_factorial'+suffix),dpi=360,facecolor='white')
    plt.close(fig)
    print('Created recipe factorial PDF and PNG')

if __name__ == '__main__':
    if sys.argv[1:] in (['--preview'], ['--approved-export']):
        interaction_preview()
    elif sys.argv[1:]:
        raise SystemExit('Use --preview for outputs/previews, or --approved-export for outputs/figures.')
    else:
        sys.argv.append('--approved-export')
        interaction_preview()
