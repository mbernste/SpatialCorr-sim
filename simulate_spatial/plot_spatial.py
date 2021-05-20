import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import math
import importlib
import seaborn as sns
from collections import defaultdict

PALETTE = [
    "#0652ff", #  electric blue
    "#e50000", #  red
    "#9a0eea", #  violet
    "#01b44c", #  shamrock
    "#fedf08", #  dandelion
    "#00ffff", #  cyan
    "#89fe05", #  lime green
    "#a2cffe", #  baby blue
    "#dbb40c", #  gold
    "#029386", #  teal
    "#ff9408", #  tangerine
    "#d8dcd6", #  light grey
    "#80f9ad", #  seafoam
    "#3d1c02", #  chocolate
    "#fffd74", #  butter yellow
    "#536267", #  gunmetal
    "#f6cefc", #  very light purple
    "#650021", #  maroon
    "#020035", #  midnight blue
    "#b0dd16", #  yellowish green
    "#9d7651", #  mocha
    "#c20078", #  magenta
    "#380282", #  indigo
    "#ff796c", #  salmon
    "#874c62"  #  dark muave
]


def plot_and_run_cov(
        gene_1,
        gene_2,
        kernel_matrix,
        expr_1,
        expr_2,
        df,
        show=True,
        ct_to_indices=None,
        dot_size=30
    ):
    cov_mats  = llrt.kernel_estimation(
        kernel_matrix,
        np.array([expr_1, expr_2])
    )

    cov = [
        C[0][1]
        for C in cov_mats
    ]

    corr = [
        C[0][1] / np.sqrt(C[0][0]*C[1][1])
        for C in cov_mats
    ]

    df_cov = pd.DataFrame(
        data={'covariance': cov},
        index=df.index
    )
    df_corr = pd.DataFrame(
        data={'correlation': corr},
        index=df.index
    )
    df_cov = df.join(df_cov)
    df_corr = df.join(df_corr)

    figure, axarr = plt.subplots(
        2,
        2,
        figsize = (10,10)
    )

    y = -1 * np.array(df['row'])
    x = df['col']
    color = expr_1
    axarr[0][0].scatter(x,y,c=color, cmap='viridis', s=dot_size)
    axarr[0][0].set_title('{} expression'.format(gene_1))

    y = -1 * np.array(df['row'])
    x = df['col']
    color = expr_2
    axarr[0][1].scatter(x,y,c=color, cmap='viridis', s=dot_size)
    axarr[0][1].set_title('{} expression'.format(gene_2))

    y = -1 * np.array(df_corr['row'])
    x = df_corr['col']
    color = df_corr['correlation']
    #color = df_cov['covariance']
    im = axarr[1][0].scatter(x,y,c=color, cmap='RdBu_r', s=dot_size, vmin=-1, vmax=1)
    #im = axarr[1][0].scatter(x,y,c=color, cmap='viridis', s=30)
    figure.colorbar(im, ax=axarr[1][0], boundaries=np.linspace(-1,1,100))
    axarr[1][0].set_title('Dynamic Correlation')

    return np.array(df_corr['correlation'])


def plot_neighborhood(
        df,
        sources,
        bc_to_neighbs,
        plot_vals,
        plot=False,
        ax=None,
        keep_inds=None,
        dot_size=30,
        vmin=0,
        vmax=1,
        cmap='RdBu_r',
        ticks=True,
        title=None,
        condition=False,
        cell_type_key=None,
        title_size=12,
        neighb_color='black'
    ):

    # Get all neighborhood spots
    all_neighbs = set()
    for source in sources:
        neighbs = set(bc_to_neighbs[source])
        if condition:
            ct_spots = set(df.loc[df[cell_type_key] == df.loc[source][cell_type_key]].index)
            neighbs = neighbs & ct_spots 
        all_neighbs.update(neighbs)

    if keep_inds is not None:
        all_neighbs &= set(keep_inds)

    y = -1 * np.array(df['row'])
    x = df['col']
    colors=plot_vals
    ax.scatter(x,y,c=colors, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
    
    colors = []
    plot_inds = []
    for bc_i, bc in enumerate(df.index):
        if bc in sources:
            plot_inds.append(bc_i)
            colors.append(neighb_color)
        elif bc in all_neighbs:
            plot_inds.append(bc_i)
            colors.append(neighb_color)
    if ax is None:
        figure, ax = plt.subplots(
            1,
            1,
            figsize=(5,5)
        )
    y = -1 * np.array(df.iloc[plot_inds]['row'])
    x = df.iloc[plot_inds]['col']
    ax.scatter(x,y,c=colors, s=dot_size)

    # Re-plot the colored dots over the highlighted neighborhood. Make 
    # the dots smaller so that the highlights stand out.
    colors=np.array(plot_vals)[plot_inds]
    ax.scatter(x,y,c=colors, cmap=cmap, s=dot_size*0.25, vmin=vmin, vmax=vmax)
    
    if not title:
        ax.set_title('Neighborhood around ({}, {})'.format(df.loc[source]['row'], df.loc[source]['col']), fontsize=title_size)
    else:
        ax.set_title(title, fontsize=title_size)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot:
        plt.show()
    return ax


def plot_slide(
        df, 
        values, 
        cmap='viridis', 
        colorbar=False, 
        vmin=None, 
        vmax=None, 
        title=None, 
        ax=None, 
        figure=None, 
        ticks=True, 
        dsize=37, 
        colorticks=None, 
        row_key='row', 
        col_key='col'
    ):
    y = -1 * np.array(df[row_key])
    x = df[col_key]

    if ax is None:
        if colorbar:
            width = 7
        else:
            width = 5
        figure, ax = plt.subplots(
            1,
            1,
            figsize=(width,5)
        )

    if cmap == 'categorical':
        val_to_index = {
            val: ind
            for ind, val in enumerate(sorted(set(values)))
        }
        colors = [
            PALETTE[val_to_index[val]]
            for val in values
        ]
        patches = [
            mpatches.Patch(color=PALETTE[val_to_index[val]], label=val)
            for val in sorted(set(values))
        ]
        ax.scatter(x,y,c=colors, s=dsize)
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',)
    else:
        im = ax.scatter(x,y,c=values, cmap=cmap, s=dsize, vmin=vmin, vmax=vmax)
        if colorbar:
            if vmin is None or vmax is None:
                figure.colorbar(im, ax=ax, ticks=colorticks)
            else:
                figure.colorbar(im, ax=ax, boundaries=np.linspace(vmin,vmax,100), ticks=colorticks)
    if title is not None:
        ax.set_title(title)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_cluster_cell_type(df, ct_key, dotsize=10):
    y = -1 * np.array(df['row'])
    x = df['col']
    try:
        cell_types = [str(y) for y in sorted([int(x) for x in set(df[ct_key])])]
    except ValueError: 
        cell_types = sorted(set(df[ct_key])) 
    cell_type_to_ind = {ct: i for i, ct in enumerate(cell_types)}
    color = [PALETTE[cell_type_to_ind[str(ct)]] for ct in df[ct_key]]
    patches = [
        mpatches.Patch(color=c, label=ct)
        for c, ct in zip(PALETTE, cell_types)
    ]
    plt.scatter(x,y,c=color, s=dotsize)
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',)
    plt.show()


