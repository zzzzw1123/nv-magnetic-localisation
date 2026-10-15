"""Render seven supplementary figures from retained source-data tables.

The script uses stored estimates without fitting, resampling, or smoothing.
"""
from pathlib import Path
import hashlib
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
SOURCE = Path(os.environ.get('NV_FIGURE_SOURCE_DIR', CODE_ROOT / 'figure_data')).expanduser()
OUTPUT_ROOT = Path(os.environ.get('NV_OUTPUT_DIR', CODE_ROOT / 'outputs')).expanduser()
APPROVED_EXPORT = sys.argv[1:] == ['--approved-export']
OUT = OUTPUT_ROOT / ('figures' if APPROVED_EXPORT else 'previews')
BLUE, RED, GRAY, INK = '#168CBB', '#ED1746', '#777D83', '#202020'
HASHES = {}


def read(relative):
    path = SOURCE / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    HASHES[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def setup():
    font_manager.findfont('Arial', fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'font.size': 10.2, 'axes.labelsize': 10.8, 'axes.linewidth': .85,
        'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.4, 'ytick.major.size': 3.4,
        'xtick.major.width': .85, 'ytick.major.width': .85,
        'legend.frameon': False, 'legend.fontsize': 9.3,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'path.simplify': False, 'mathtext.fontset': 'dejavusans',
    })


def frame(ax):
    ax.grid(False)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(pad=3.3)


def panels(height=88, bottom=.28, axes_height=.65):
    fig = plt.figure(figsize=(150 / 25.4, height / 25.4))
    axes = [fig.add_axes([left, bottom, .36, axes_height]) for left in (.13, .615)]
    for ax in axes:
        frame(ax)
    return fig, axes


def bottom_panel_names(fig, axes, names, y):
    assert len(axes) == len(names)
    for i, (ax, name) in enumerate(zip(axes, names)):
        fig.text(ax.get_position().x0 + ax.get_position().width / 2, y,
                 f'({"abc"[i]}) {name}', ha='center', va='center', fontsize=10.4)


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (stem, artist.get_text())
            assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (stem, artist.get_text())
    target = OUT / (stem+'.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=450)
    if APPROVED_EXPORT:
        mpl.rcParams['pdf.fonttype'] = 42
        fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)
    print('PNG:', stem)


def candidate_frontier():
    data = read('supplementary/candidate_frontier_source.csv')
    raw = read('supplementary/candidate_frontier_raw_points.csv')
    assert len(data) == 12 and len(raw) == 60
    assert data.development_folds.eq(5).all()
    assert raw.groupby('candidate').size().eq(5).all()
    fig = plt.figure(figsize=(150/25.4, 88/25.4))
    ax = fig.add_axes([.135, .17, .825, .67])
    frame(ax)
    families = {'MLP': (RED, 'o'), 'Polynomial Ridge': (GRAY, 's'), 'RFF Ridge': (GRAY, '^')}
    for row in data.itertuples(index=False):
        group = raw[raw.candidate.eq(row.candidate)]
        mean, sd = group.error_um.mean(), group.error_um.std(ddof=1)
        assert np.allclose([mean, sd], [row.mean_development_error_um, row.sd_development_error_um], atol=1e-12, rtol=0)
        assert group.display_x.eq(row.display_x).all()
        color, marker = families[row.family]
        ax.plot(group.display_x, group.error_um, 'o', ms=2.2, color='#B8BDC2',
                markeredgewidth=0, alpha=.8, zorder=1)
        ax.errorbar(row.display_x, mean, yerr=sd, fmt=marker, linestyle='none',
                    color=color, mfc='white', mec=color, mew=1.1,
                    ms=5.6 if row.selected else 4.8, elinewidth=1.0, capsize=1.8, zorder=3)
    for candidate, label, offset, align in (
            ('mlp8_fixed', 'MLP8 (selected)', (-6, 11), 'right'),
            ('mlp16_fixed', 'MLP16', (0, -12), 'center'),
            ('mlp32_fixed', 'MLP32', (-3, -12), 'right')):
        row = data[data.candidate.eq(candidate)].iloc[0]
        ax.annotate(label, (row.display_x, row.mean_development_error_um),
                    xytext=offset, textcoords='offset points', ha=align, va='center',
                    fontsize=9.4, color=RED if candidate=='mlp8_fixed' else INK,
                    fontweight='bold' if candidate=='mlp8_fixed' else 'normal')
    ax.set(xscale='log', xlim=(112, 960), ylim=(10, 270),
           xlabel='Fitted parameters (log scale)', ylabel='Development CV error (μm)',
           yticks=[50, 100, 150, 200, 250])
    ax.xaxis.set_major_locator(FixedLocator([200, 400, 600, 800]))
    ax.xaxis.set_major_formatter(FixedFormatter(['200','400','600','800']))
    ax.xaxis.set_minor_locator(NullLocator())
    handles = [Line2D([], [], marker=m, mfc='white', mec=c, color=c, ls='none',
                      ms=4.8, mew=1.1, label=name) for name, (c,m) in families.items()]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.545, .985),
               handletextpad=.45, columnspacing=1.25)
    save(fig, 'supplementary/candidate_frontier')


def cdf(ax, values, color, style, limits, label):
    values = np.sort(np.asarray(values, dtype=float))
    assert np.isfinite(values).all() and limits[0] <= values.min() <= values.max() <= limits[1]
    x = np.r_[limits[0], values, limits[1]]
    y = np.r_[0, np.arange(1, len(values)+1)/len(values), 1]
    line, = ax.step(x, y, where='post', color=color, ls=style, lw=1.25, label=label)
    assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
    return line


def residual_diagnostics():
    data = read('supplementary/residual_diagnostics_source.csv')
    assert len(data) == 2457 and data.row_index.nunique()==819 and data.seed.nunique()==3
    fig, (a,b) = panels(86)
    for model, color, style in [('CNN',BLUE,'-'), ('MLP8',RED,(0,(4,2.5)))]:
        cdf(a, data[model.lower()+'_error_um'], color, style, (.2,600), model)
    a.set(xscale='log', xlim=(.2,600), ylim=(0,1.02),
          xlabel='Euclidean error (μm)', ylabel='Cumulative probability', yticks=[0,.25,.5,.75,1])
    a.xaxis.set_major_locator(FixedLocator([.2,1,10,100,600]))
    a.xaxis.set_major_formatter(FixedFormatter(['0.2','1','10','100','600']))
    a.xaxis.set_minor_locator(NullLocator())
    a.legend(loc='lower right', handlelength=1.8)
    handles=[]
    for coord, color, marker in [('x','#6F7B85','o'),('y','#A99BB0','^')]:
        xx, yy = data['cnn_pred_'+coord+'_um'], data['mlp8_pred_'+coord+'_um']
        assert np.isfinite([xx,yy]).all() and min(xx.min(), yy.min())>1080 and max(xx.max(), yy.max())<2370
        b.scatter(xx, yy, s=4.0, color=color, marker=marker, alpha=.4, edgecolors='none', zorder=2)
        handles.append(Line2D([],[],ls='none',marker=marker,color=color,ms=4,label=coord+' coordinate'))
    b.plot([1080,2370], [1080,2370], color='#B9BDC1', lw=.7, ls=(0,(4,3)), zorder=0)
    handles.append(Line2D([],[],ls=(0,(4,3)),color='#A0A5AA',lw=.7,label='Identity'))
    b.set(xlim=(1080,2370), ylim=(1080,2370), aspect='equal',
          xlabel='CNN prediction (μm)', ylabel='MLP8 prediction (μm)',
          xticks=[1200,1800,2300], yticks=[1200,1800,2300])
    b.legend(handles=handles,loc='upper left',fontsize=8.4,handlelength=1.5)
    bottom_panel_names(fig, (a,b), ('Error distributions', 'Paired predictions'), .052)
    save(fig,'supplementary/residual_diagnostics')


def source_split_diagnostics():
    data = read('supplementary/source_split_diagnostics_source.csv')
    pairs = read('supplementary/source_split_diagnostics_pairs_source.csv')
    assert len(pairs)==25 and pairs['rank'].tolist()==list(range(1,26))
    fig, (a,b) = panels(88)
    models = [('architecture_matched_cnn','CNN',BLUE,'o','-'),
              ('mlp8','MLP8',RED,'s',(0,(4,2.5))),
              ('sparse16_ridge','Sparse16',GRAY,'^',(0,(1,2)))]
    ids=None
    for model, label, color, marker, style in models:
        rows = data[data.model.eq(model)]
        assert len(rows)==25 and rows.coordinate_id.nunique()==25
        if ids is None: ids=set(rows.coordinate_id)
        assert set(rows.coordinate_id)==ids
        cdf(a, rows.euclidean_error_um, color, style, (0,165), label)
    a.set(xlim=(0,165),ylim=(0,1.02),xlabel='Localisation error (μm)',
          ylabel='Cumulative probability',xticks=[0,80,160],yticks=[0,.25,.5,.75,1])
    ce, me, ranks = pairs.cnn_error_um, pairs.mlp8_error_um, pairs['rank']
    for model, saved in [('architecture_matched_cnn',ce), ('mlp8',me)]:
        table = data[data.model.eq(model)].set_index('coordinate_id')
        assert np.allclose(table.loc[pairs.coordinate_id].euclidean_error_um,saved,atol=1e-12,rtol=0)
    b.hlines(ranks, np.minimum(ce,me),np.maximum(ce,me),color='#D3D6D9',lw=.6,zorder=0)
    for vals, marker, color in [(ce,'o',BLUE),(me,'s',RED)]:
        b.plot(vals,ranks,ls='none',marker=marker,ms=3.7,mfc='white',mec=color,mew=.9,zorder=3)
    b.set(xlim=(0,96),ylim=(0,26),xlabel='Localisation error (μm)',ylabel='Observation rank',
          xticks=[0,40,80],yticks=[1,5,10,15,20,25])
    handles=[Line2D([],[],color=c,marker=m,mfc='white',mec=c,mew=.9,ms=4.4,ls=s,lw=1.1,label=label)
             for _,label,c,m,s in models]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.997),ncol=3,
               columnspacing=1.3,handletextpad=.5)
    bottom_panel_names(fig, (a,b), ('Error distributions', 'Paired errors'), .052)
    save(fig,'supplementary/source_split_diagnostics')


def spatial_data(relative):
    data = read(relative)
    assert len(data)==714 and not data.duplicated(['true_x_um','true_y_um']).any()
    xs, ys = np.sort(data.true_x_um.unique()), np.sort(data.true_y_um.unique())
    assert len(xs)==14 and len(ys)==51 and np.all(np.diff(xs)==20) and np.all(np.diff(ys)==20)
    assert data.n_observations.sum()==819 and data.n_seed_repeats_per_observation.eq(3).all()
    assert np.allclose(data.mlp8_mean_error_um-data.cnn_mean_error_um,data.mlp8_minus_cnn_error_um,atol=1e-12,rtol=0)
    return data, xs, ys


def cells(ax, data, xs, ys, field, cmap, norm):
    table=data.set_index(['true_x_um','true_y_um'])
    grid=np.full((len(xs),len(ys)),np.nan)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            if (x,y) in table.index:
                grid[i,j]=table.loc[(x,y),field]
    assert np.isfinite(grid).sum()==len(data)
    image=ax.pcolormesh(np.r_[ys-10,ys[-1]+10],np.r_[xs-10,xs[-1]+10],
                       np.ma.masked_invalid(grid),cmap=cmap,norm=norm,shading='flat',edgecolors='none')
    ax.set(xlim=(1190,2210),ylim=(1295,1575),aspect='equal',
           xticks=[1200,1400,1600,1800,2000,2200],yticks=[1305,1435,1565])
    frame(ax)
    return image


def cmaps():
    absolute=LinearSegmentedColormap.from_list('absolute',['#F3F5F3','#D3E1DF','#88B1B7','#3A7188','#254B62'])
    difference=LinearSegmentedColormap.from_list('difference',['#AF5B49','#DFA08F','#F6F5F1','#B9D4DF','#356783'])
    return absolute,difference


def spatial_maps():
    data,xs,ys=spatial_data('supplementary/spatial_maps_complete_source.csv')
    absolute,difference=cmaps()
    figure_height = 138
    fig=plt.figure(figsize=(150/25.4,figure_height/25.4))
    maps=[];axes=[]
    map_height = .72*150*280/1020
    for i,(bottom,field,title) in enumerate(zip((102,62,22),
            ('cnn_mean_error_um','mlp8_mean_error_um','mlp8_minus_cnn_error_um'),('CNN','MLP8','MLP8 − CNN'))):
        ax=fig.add_axes([.145,bottom/figure_height,.72,map_height/figure_height])
        norm=Normalize(0,200) if i<2 else TwoSlopeNorm(vcenter=0,vmin=-100,vmax=100)
        assert data[field].between(norm.vmin,norm.vmax).all()
        image=cells(ax,data,xs,ys,field,absolute if i<2 else difference,norm)
        ax.set_ylabel('Stage X (μm)',fontsize=9.7)
        ax.tick_params(labelsize=9.0)
        if i<2: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel('Stage Y (μm)')
        axes.append(ax);maps.append(image)
        label_y = (bottom-7)/figure_height if i<2 else 2.5/figure_height
        fig.text(.145+.36, label_y, f'({"abc"[i]}) {title}',
                 ha='center', va='center', fontsize=10.4)
    cax=fig.add_axes([.885,62/figure_height,.012,(102+map_height-62)/figure_height])
    cb=fig.colorbar(maps[0],cax=cax,ticks=[0,50,100,150,200]);cb.set_label('Mean error (μm)',fontsize=9.5,labelpad=4)
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8.8,length=2.5,width=.7)
    cax=fig.add_axes([.885,22/figure_height,.012,map_height/figure_height])
    cb=fig.colorbar(maps[2],cax=cax,ticks=[-100,0,100]);cb.set_label('Difference (μm)',fontsize=9.5,labelpad=4)
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8.8,length=2.5,width=.7)
    fig.canvas.draw()
    for ax in axes:
        bounds=ax.get_window_extent()
        assert np.isclose((bounds.width/1020)/(bounds.height/280),1)
    save(fig,'supplementary/spatial_maps_complete')


def spatial_difference():
    data,xs,ys=spatial_data('study_l_fig6_spatial_error_source.csv')
    _,difference=cmaps()
    fig=plt.figure(figsize=(150/25.4,62/25.4))
    ax=fig.add_axes([.12,.235,.815,(.815*150*280/1020)/62])
    image=cells(ax,data,xs,ys,'mlp8_minus_cnn_error_um',difference,TwoSlopeNorm(vcenter=0,vmin=-60,vmax=60))
    ax.set(xlabel='Stage Y (μm)',ylabel='Stage X (μm)')
    assert ((data.true_x_um==1325)&(data.true_y_um==2200)).sum()==1
    ax.add_patch(Rectangle((2190,1315),20,20,fill=False,edgecolor=INK,lw=.9,zorder=4,clip_on=False))
    ax.annotate('P',(2200,1325),xytext=(8,0),textcoords='offset points',fontsize=10.2,va='center')
    cax=fig.add_axes([.31,.885,.43,.027])
    cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=[-60,-30,0,30,60])
    cb.ax.set_xticklabels(['≤−60','−30','0','30','≥60'])
    cb.ax.tick_params(labelsize=8.9,length=2.5,width=.7,pad=2)
    cb.outline.set_visible(False)
    fig.text(.525,.977,'MLP8 − CNN error (μm)',ha='center',va='top',fontsize=10.8)
    fig.canvas.draw();bounds=ax.get_window_extent()
    assert np.isclose((bounds.width/1020)/(bounds.height/280),1)
    save(fig,'study_l_fig6_spatial_error')


def training():
    data=read('study_l_fig5_training_dynamics_source.csv')
    minima=read('study_l_fig5_training_dynamics_minima.csv')
    assert len(data)==904 and data.seed.eq(20260804).all() and len(minima)==4
    fig,(a,b)=panels(88,bottom=.30)
    for ax,model,color,budget,ylabel in [(a,'architecture_matched_cnn',BLUE,3000,'Coordinate MSE (μm²)'),
                                       (b,'mlp8',RED,1500,'Standardised-target MSE')]:
        for schedule,style in [('fixed_lr_0.005','-'),('cosine_annealing',(0,(4,2.5)))]:
            rows=data[(data.model==model)&(data.schedule==schedule)].sort_values('epoch')
            assert np.array_equal(rows.epoch,np.r_[np.arange(1,budget,10),budget])
            assert (rows[['train_mse','validation_mse']].to_numpy()>0).all()
            ax.plot(rows.epoch,rows.train_mse,color='#B8BDC2',lw=.8,ls=style)
            line,=ax.plot(rows.epoch,rows.validation_mse,color=color,lw=1.25,ls=style)
            assert np.array_equal(line.get_ydata(),rows.validation_mse)
            best=rows.loc[rows.validation_mse.idxmin()]
            original=minima[(minima.model==model)&(minima.schedule==schedule)].iloc[0]
            assert best.epoch==original.epoch and best.validation_mse==original.validation_mse
            ax.plot(best.epoch,best.validation_mse,'o',ms=4.8,mfc='white',mec=color,mew=1.1,zorder=5)
        ax.set(yscale='log',xlim=(0,budget*1.015),xlabel='Epoch',ylabel=ylabel,
               xticks=np.linspace(0,budget,4),ylim=(100,1e6) if model!='mlp8' else (.01,2))
        ax.set_yticks([100,1000,10000,100000,1000000] if model!='mlp8' else [.01,.1,1])
        ax.yaxis.set_minor_locator(NullLocator())
    handles=[Line2D([],[],color='#B8BDC2',lw=.9,label='Training'),
             Line2D([],[],color=INK,lw=1.25,label='Validation'),
             Line2D([],[],color=INK,lw=1.1,ls='-',label='Fixed LR'),
             Line2D([],[],color=INK,lw=1.1,ls=(0,(4,2.5)),label='Cosine LR')]
    fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.55,.005),
               columnspacing=1.0,handletextpad=.5,fontsize=8.5)
    bottom_panel_names(fig, (a,b), ('CNN', 'MLP8'), .115)
    save(fig,'study_l_fig5_training_dynamics')


def feature_robustness():
    raw=read('study_l_fig7_feature_robustness_acquisitions_source.csv')
    summary=read('study_l_fig7_feature_robustness_source.csv')
    assert len(raw)==32 and raw.seed_count.eq(3).all()
    variants=('full24','without_peak_split')
    values=raw.pivot(index='session',columns='variant',values='mean_error_um')
    assert len(values)==8 and not values[list(variants)].isna().any().any()
    for variant in variants:
        row=summary[(summary.panel=='a')&(summary.condition==variant)].iloc[0]
        assert np.allclose([values[variant].mean(),values[variant].std(ddof=1)],
                           [row.centre,row.upper-row.centre],rtol=0,atol=1e-10)
    figure_height = 94
    fig,(a,b)=panels(figure_height,bottom=.34,axes_height=(.65*88)/figure_height)
    for _,row in values.iterrows():
        a.plot([0,1],row[list(variants)],color='#AEB4BA',lw=.7,zorder=0)
    for i,(variant,color) in enumerate(zip(variants,(BLUE,RED))):
        a.plot(np.full(8,i),values[variant],ls='none',marker='o',ms=4.8,mfc='white',mec=color,mew=1.1,zorder=3)
    a.set(xlim=(-.25,1.25),ylim=(0,150),xticks=[0,1],xticklabels=['All features','Without Δμ'],
          yticks=[0,30,60,90,120,150],ylabel='MLP8 LOAO error (μm)')
    a.tick_params(axis='x',labelsize=8.9)
    for model,color,marker in [('CNN',BLUE,'o'),('MLP8',RED,'s')]:
        rows=summary[(summary.panel=='b')&(summary.model==model)].sort_values('x')
        assert len(rows)==5 and np.array_equal(rows.x,[0,1,2,5,10])
        assert rows.interval.eq('acquisition IQR after seed averaging').all()
        assert (rows.lower<=rows.centre).all() and (rows.upper>=rows.centre).all()
        interval=b.errorbar(rows.x,rows.centre,yerr=[rows.centre-rows.lower,rows.upper-rows.centre],
                            fmt=marker,ls='-',color=color,mfc='white',mec=color,mew=1.05,
                            ms=4.8,lw=1.1,elinewidth=.85,capsize=1.8,label=model)
        for segment,low,high in zip(interval.lines[2][0].get_segments(),rows.lower,rows.upper):
            assert np.allclose(segment[:,1],[low,high],atol=1e-12,rtol=0)
    b.set(xlim=(-.4,10.5),ylim=(30,55),xticks=[0,1,2,5,10],yticks=[30,35,40,45,50,55],
          xlabel='Noise SD / training-feature SD\n(%)',ylabel='LOAO error (μm)')
    b.xaxis.label.set_fontsize(9.4)
    b.legend(loc='upper left',handlelength=1.6)
    bottom_panel_names(fig, (a,b), ('Feature ablation', 'Perturbation sensitivity'), .047)
    save(fig,'study_l_fig7_feature_robustness')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    for draw in (candidate_frontier,residual_diagnostics,source_split_diagnostics,
                 spatial_maps,training,spatial_difference,feature_robustness):
        draw()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==value for path,value in HASHES.items())
    print('Seven PNG previews; retained source tables unchanged.')


if __name__=='__main__':
    main()
