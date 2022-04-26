#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


from mintpy import view
from mintpy.utils import readfile, utils as ut, plot as pp

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import h5py




#############################################################

def save_velo_h5(vel, atr, outfile):
    # Normalize a pathname by collapsing redundant separators and up-levels
    outfile = os.path.normpath(outfile)

    # create outdir if not exists
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save velocity
    with h5py.File(outfile,'w') as out_data:
        ds = out_data.create_dataset('velocity', shape=vel.shape, dtype=vel.dtype)
        ds[:] = vel
        for key in atr.keys():
            out_data.attrs[key] = str(atr[key])


def save_fig(fig, outfile, **kwargs):
    # Normalize a pathname by collapsing redundant separators and up-levels
    outfile = os.path.normpath(outfile)

    # create outdir if not exists
    outdir = os.path.dirname(outfile)
    ext = os.path.splitext(outfile)[-1].split('.')[-1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f'Save to file: {outfile}')
    fig.savefig(outfile, transparent=True, dpi=kwargs['dpi'], bbox_inches='tight', format=ext)


def depth(lst):
    if not isinstance(lst, list):
        return 0
    else:
        return 1 + max(depth(sublist) for sublist in lst)


def read_mask(mask_file):
    mask_data = readfile.read(mask_file)[0]
    mask      = 1*(mask_data==1)
    return mask


def read_img(fname, mask=None):
    if mask in ['no', 'No', 'NO']:
        mask = None
    # The dataset unit is meter
    v     = readfile.read(fname, datasetName='velocity')[0]*1000     # Unit: mm/y
    #meta  = readfile.read(fname, datasetName='velocity')[1]          # metadata
    #v_std = readfile.read(fname, datasetName='velocityStd')[0] *1000  # Unit: mm/y

    # read mask and mask the dataset
    if mask is not None:
        mask_data = readfile.read(mask)[0] # 'waterMask.h5' or 'maskTempCoh.h5'
        v[mask_data==0] = np.nan
    #v_std[mask_data==0] = np.nan
    #water_mask = readfile.read('../../waterMask.h5')[0]
    return v



def know_loc_from_subplots(fig, yoff=0.8):
    p = fig.subplotpars
    w, h = fig.get_size_inches()
    #figw = float(w)/(p.right - p.left)
    #figh = float(h)/(p.top   - p.bottom)
    yy = 1 + (yoff/h)
    return yy

def make_cbar_ax_side(fig, yloc='center', ch=0.4, cw_in=0.4):
    """
    yc          y center of the colorbar, in portion of the figure height
    ch          colorbar height, in portion of the figure height
    cw_in       colorbar width, in inches
    """
    y_bottom = 0.05416694
    y_top    = 0.94756917

    par  = fig.subplotpars
    w, h = fig.get_size_inches()
    print(par.bottom, par.top)
    print(y_bottom, y_top)

    cw   = cw_in / w
    cx   = 1 + cw
    if yloc == 'center':
        cy = 0.5 - ch/2
    elif yloc == 'top':
        cy = y_top - ch
    elif yloc == 'bottom':
        cy = y_bottom
    cbar_ax = fig.add_axes([cx, cy, cw, ch])
    return cbar_ax


def colorbar_inset(fig, ax, data, im, atr=None, cbar_inset=True, **kwargs):
    # Initialize the kwargs
    if 'unit'            not in kwargs.keys():   kwargs['unit']           = ''
    if 'ub'              not in kwargs.keys():   kwargs['ub']             = 99.8
    if 'lb'              not in kwargs.keys():   kwargs['lb']             =   .2
    if 'box_alpha'       not in kwargs.keys():   kwargs['box_alpha']      = .9
    if 'orient'          not in kwargs.keys():   kwargs['orient']         = 'vertical'
    if 'unit_fontsize'   not in kwargs.keys():   kwargs['unit_fontsize']  = plt.rcParams['font.size'] * .75
    if 'bound_fontsize'  not in kwargs.keys():   kwargs['bound_fontsize'] = plt.rcParams['font.size'] * .75
    if 'ticks_fontsize'  not in kwargs.keys():   kwargs['ticks_fontsize'] = plt.rcParams['font.size'] * .75
    if 'max_fontsize'    not in kwargs.keys():   kwargs['max_fontsize']   = 22
    if 'box_loc'         not in kwargs.keys():
        if kwargs['orient'] == 'horizontal':
                                                 kwargs['box_loc']        = (0., 0., .6, .08)
        elif kwargs['orient'] == 'vertical':
            if atr == None:
                                                 kwargs['box_loc']        = (1-.16, 0., .16, .3)
            else:
                if   atr['ORBIT_DIRECTION'].startswith('A'):
                                                 kwargs['box_loc']        = (1-.22, 1-.3, .22, .3)
                elif atr['ORBIT_DIRECTION'].startswith('D'):
                                                 kwargs['box_loc']        = (1-.22, 0., .22, .3)
    if   kwargs['orient'] == 'horizontal':
        text_rotate, ubx, uby, lbx, lby, uha, lha  = 0 , 0.98, 0.5,  0.02, 0.5, 'right', 'left'
        cbar_w, cbar_h = '80%', '40%'
        cbar_anchor    = (0, 0.15, 1, 1)
    elif kwargs['orient'] == 'vertical':
        text_rotate, ubx, uby, lbx, lby, uha, lha  = 270, 0.5 , 1.08, 0.5, -0.08, 'center', 'center'
        cbar_w, cbar_h = '19%', '74%'
        cbar_anchor    = (-.1, 0, 1, 1)

    upper_val = np.nanpercentile(data, kwargs['ub'])
    lower_val = np.nanpercentile(data, kwargs['lb'])

    if cbar_inset:
        cbox = ax.inset_axes(kwargs['box_loc'], zorder=1)
        cbox.set_facecolor('w')
        cbox.patch.set_alpha(kwargs['box_alpha'])
        cbox.get_xaxis().set_visible(False)
        cbox.get_yaxis().set_visible(False)
        cax = inset_axes(cbox, width=cbar_w, height=cbar_h, loc='center', bbox_transform=cbox.transAxes, bbox_to_anchor=cbar_anchor)
        kwargs['unit_fontsize']  = np.clip(a=kwargs['unit_fontsize'],  a_min=None, a_max=kwargs['max_fontsize']*0.9)
        kwargs['bound_fontsize'] = np.clip(a=kwargs['bound_fontsize'], a_min=None, a_max=kwargs['max_fontsize']*0.9)
        kwargs['ticks_fontsize'] = np.clip(a=kwargs['ticks_fontsize'], a_min=None, a_max=kwargs['max_fontsize'])
    else:
        cax = ax
        kwargs['unit_fontsize']  = plt.rcParams['font.size']
        kwargs['bound_fontsize'] = 0
        kwargs['ticks_fontsize'] = plt.rcParams['font.size']

    fig.colorbar(im, cax=cax, orientation=kwargs['orient'])
    cax.text(0.5, 0.5, kwargs['unit'], ha='center', va='center', rotation=text_rotate, fontsize=kwargs['unit_fontsize'], transform=cax.transAxes)
    cax.text(lbx, lby, '({:.1f})'.format(lower_val), ha=lha, va='center', fontsize=kwargs['bound_fontsize'], transform=cax.transAxes)
    cax.text(ubx, uby, '({:.1f})'.format(upper_val), ha=uha, va='center', fontsize=kwargs['bound_fontsize'], transform=cax.transAxes)
    cax.tick_params(labelsize=kwargs['ticks_fontsize'])


def scalebar_inset(fig, ax, inps, **kwargs):
    # Initialize the kwargs
    if 'loc'            not in kwargs.keys():   kwargs['loc']           = 'lower left'
    if 'box_loc'        not in kwargs.keys():
        if   kwargs['loc'] == 'upper left':     kwargs['box_loc']       = (0, 1-.05, .4, .05)
        elif kwargs['loc'] == 'lower left':     kwargs['box_loc']       = (0, 0, .4, .05)
    if 'box_alpha'      not in kwargs.keys():   kwargs['box_alpha']     = .9
    if 'font_size'      not in kwargs.keys():   kwargs['font_size']     = plt.rcParams['font.size'] * .75
    if 'scalebar_loc'   not in kwargs.keys():   kwargs['scalebar_loc']  = [0.3, kwargs['box_loc'][0]+0.2, kwargs['box_loc'][1]+.015]
    if 'scalebar_pad'   not in kwargs.keys():   kwargs['scalebar_pad']  = 0.015
    if 'coord_unit'     not in kwargs.keys():   kwargs['coord_unit']    = inps.coord_unit
    if 'geo_box'        not in kwargs.keys():   kwargs['geo_box']       = inps.geo_box

    sbox = ax.inset_axes(kwargs['box_loc'], zorder=1)
    sbox.set_facecolor('w')
    sbox.patch.set_alpha(kwargs['box_alpha'])
    sbox.get_xaxis().set_visible(False)
    sbox.get_yaxis().set_visible(False)
    ax = pp.draw_scalebar(ax, geo_box=inps.geo_box, unit=inps.coord_unit, loc=kwargs['scalebar_loc'], labelpad=kwargs['scalebar_pad'], font_size=kwargs['font_size'])


def satellite_inset(fig, ax, atr, **kwargs):
    # Initialize the kwargs
    if 'font_size'  not in kwargs.keys():   kwargs['font_size']     = plt.rcParams['font.size'] * .8
    if 'quiver_w'   not in kwargs.keys():   kwargs['quiver_w']      = 0.04
    if 'box_show'   not in kwargs.keys():   kwargs['box_show']      = 'on'
    if 'box_color'  not in kwargs.keys():   kwargs['box_color']     = 'w'
    if 'box_alpha'  not in kwargs.keys():   kwargs['box_alpha']     = .9
    if 'sat_az_qsc' not in kwargs.keys():   kwargs['sat_az_qsc']    = 1.
    if 'sat_rg_qsc' not in kwargs.keys():   kwargs['sat_rg_qsc']    = kwargs['sat_az_qsc'] * 2
    if 'sat_label'  not in kwargs.keys():   kwargs['sat_label']     = 'rg' # rg, az, both, los
    if 'sat_corner' not in kwargs.keys():   kwargs['sat_corner']    = True

    if 'box_loc'        not in kwargs.keys():
        if atr == None:
                                            kwargs['box_loc']       = (1-.15, 1-.1, .15, .1)
        else:
            if   atr['ORBIT_DIRECTION'].startswith('A'):
                                            kwargs['box_loc']       = (0, .05, .15, .1) #(1-.15, 0, .15, .1) for lower right
            elif atr['ORBIT_DIRECTION'].startswith('D'):
                if kwargs['sat_corner'] is True:
                                            kwargs['box_loc']       = (1-.15, 0, .15, .1)
                elif kwargs['sat_corner'] == 'upperright':
                                            kwargs['box_loc']       = (1-.15, 1-.1, .15, .1)
                else:
                                            kwargs['box_loc']       = (1-.15, .3, .15, .1)

    sbox = ax.inset_axes(kwargs['box_loc'], zorder=1)
    sbox.set_facecolor(kwargs['box_color'])
    sbox.patch.set_alpha(kwargs['box_alpha'])
    sbox.get_xaxis().set_visible(False)
    sbox.get_yaxis().set_visible(False)
    sbox.axis(kwargs['box_show'])

    head_rad = np.deg2rad(float(atr['HEADING']))
    look_rad = np.deg2rad(float(atr['HEADING'])+90.)
    fly      = np.array([np.sin(head_rad), np.cos(head_rad)])
    look     = np.array([np.sin(look_rad), np.cos(look_rad)])

    if atr['ORBIT_DIRECTION'].startswith('A'):
        tail        = np.array([0.4, 0.12])
        rot_rg      = 90 - np.rad2deg(look_rad)
        rot_az      = rot_rg - 90
        text_rg     = [.1, 0]
        text_az     = [-.06, +.4]
        va, ha      = 'top', 'left'
    elif atr['ORBIT_DIRECTION'].startswith('D'):
        tail        = np.array([0.7, 0.85])
        rot_rg      = 270 - np.rad2deg(look_rad)
        rot_az      = rot_rg + 90
        text_rg     = [0, .08]
        text_az     = [-.06, -.3]
        va, ha      = 'bottom', 'right'

    peg = tail + 0.3*fly*(1/kwargs['sat_az_qsc'])
    # draw azimuth arrow
    #sbox.scatter(*peg,  color='k', transform=sbox.transAxes)   # no need to draw the peg point
    sbox.quiver(tail[0], tail[1], fly[0], fly[1], transform=sbox.transAxes, scale=kwargs['sat_az_qsc'], scale_units='width', width=kwargs['quiver_w'])

    # draw range arrow
    if kwargs['sat_label'] == 'los':
        u, v, pivot, label = -look[0], -look[1], 'tip', kwargs['sat_label'].upper()
        sbox.quiver(*peg, u, v, transform=sbox.transAxes, scale=kwargs['sat_rg_qsc'], scale_units='width', width=kwargs['quiver_w'], pivot=pivot)
        sbox.text(*peg+text_rg, label, fontsize=kwargs['font_size'], transform=sbox.transAxes, rotation=rot_rg, va=va, ha=ha)
    elif kwargs['sat_label'] == 'rg':
        u, v, pivot = look[0], look[1], 'tail'
        sbox.quiver(*peg, u, v, transform=sbox.transAxes, scale=kwargs['sat_rg_qsc'], scale_units='width', width=kwargs['quiver_w'], pivot=pivot)
        sbox.text(*peg+text_rg, 'rg', fontsize=kwargs['font_size'], transform=sbox.transAxes, rotation=rot_rg, va=va, ha=ha)
        sbox.text(*peg+text_az, 'az', fontsize=kwargs['font_size'], transform=sbox.transAxes, rotation=rot_az, va=va, ha=ha)


def axe_bound_from_atr(atr):
    """
    atr:     mintpy .h5 attributes
    trim:    trim a fraction of x-axis and y-axis (None or 0.01) for weird white space
    """
    N = float(atr['Y_FIRST'])
    S = float(atr['Y_FIRST']) + float(atr['Y_STEP']) * (int(atr['LENGTH'])-1)
    W = float(atr['X_FIRST'])
    E = float(atr['X_FIRST']) + float(atr['X_STEP']) * (int(atr['WIDTH'])-1)
    return S, N, W, E


def plot_imgs(v, atr, dem, super_title=None, outfile=False, **kwargs):
    # Initialize the kwargs
    if 'coord'      not in kwargs.keys():   kwargs['coord']         = 'geo'
    if 'wspace'     not in kwargs.keys():   kwargs['wspace']        = 0.1
    if 'subplot_h'  not in kwargs.keys():   kwargs['subplot_h']     = 12.
    if 'sharey'     not in kwargs.keys():   kwargs['sharey']        = True
    if 'vlims'      not in kwargs.keys():   kwargs['vlims']         = [None,None]
    if 'font_size'  not in kwargs.keys():   kwargs['font_size']     = plt.rcParams['font.size']
    if 'suptity'    not in kwargs.keys():   kwargs['suptity']       = .8
    if 'units'      not in kwargs.keys():   kwargs['units']         = 'mm/year'
    if 'alpha'      not in kwargs.keys():   kwargs['alpha']         = 0.78
    if 'cmap'       not in kwargs.keys():   kwargs['cmap']          = 'RdBu_r'
    if 'side_cbar'  not in kwargs.keys():   kwargs['side_cbar']     = False
    if 'shadeExg'   not in kwargs.keys():   kwargs['shadeExg']      = 0.06
    if 'shadeMin'   not in kwargs.keys():   kwargs['shadeMin']      = -5e3
    if 'shadeMax'   not in kwargs.keys():   kwargs['shadeMax']      =  5e3
    if 'laloStep'   not in kwargs.keys():   kwargs['laloStep']      = 1
    if 'coastline'  not in kwargs.keys():   kwargs['coastline']     = '10m'
    if 'pts-marker' not in kwargs.keys():   kwargs['pts-marker']    = 'k^'
    if 'pts_ms'     not in kwargs.keys():   kwargs['pts_ms']        = 6.0
    if 'pts_yx'     not in kwargs.keys():   kwargs['pts_yx']        = None
    if 'pts_lalo'   not in kwargs.keys():   kwargs['pts_lalo']      = None
    if 'pts_file'   not in kwargs.keys():   kwargs['pts_file']      = None
    if 'title_pad'  not in kwargs.keys():   kwargs['title_pad']     = 20
    if 'dpi'        not in kwargs.keys():   kwargs['dpi']           = 150


    # Set the figure
    nrows, ncols    = 1, len(v)
    im_dim          = list(v.values())[0].shape
    fat             = im_dim[1]/im_dim[0]
    subplot_kw      = dict(projection=ccrs.PlateCarree())
    gridspec_kw     = { 'wspace'                :   kwargs['wspace']}
    fig_kw          = { 'figsize'               :   [ncols * kwargs['subplot_h'] * fat, kwargs['subplot_h']],
                        'constrained_layout'    :   True}
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=kwargs['sharey'], gridspec_kw=gridspec_kw, subplot_kw=subplot_kw, **fig_kw)

    # specs for different subplots
    if len(v) == 1:
        axs = [axs]
    if depth(kwargs['vlims']) == 1:
        kwargs['vlims'] = [kwargs['vlims']]
    if len(kwargs['vlims']) != len(axs):
        kwargs['vlims'] = [kwargs['vlims'][-1]]*len(axs)
    if isinstance(kwargs['units'], str):
        kwargs['units'] = [kwargs['units']]
    if len(kwargs['units']) != len(axs):
        kwargs['units'] = [kwargs['units'][-1]]*len(axs)

    if kwargs['vlims'].count(kwargs['vlims'][0]) == len(kwargs['vlims']):
        print('only sohw one common colorbar in the first subplot')
        one_cbar = True
    else:
        one_cbar = False


    # get axis bound
    S, N, W, E = axe_bound_from_atr(atr)

    # view.py options
    cmd0 = f" view.py {atr['FILE_PATH']} velocity --fontsize {int(kwargs['font_size'])} "
    opt  = f" --mask no --dem {dem} --dem-nocontour -c {kwargs['cmap']} --nocbar "
    opt += f" --alpha {kwargs['alpha']} --shade-exag {kwargs['shadeExg']} --shade-min {kwargs['shadeMin']} --shade-max {kwargs['shadeMax']} "
    opt += f" --lalo-step {kwargs['laloStep']} --lalo-loc 1 0 0 1 --noscalebar --notitle --noverbose "

    # plot coastline
    if  kwargs['coastline']:
        opt += f" --coastline {kwargs['coastline']} "

    # plot points of interests
    if kwargs['pts_yx'] or kwargs['pts_lalo'] or kwargs['pts_file']:
        opt += f" --pts-marker {kwargs['pts_marker']} --pts-ms {kwargs['pts_ms']} \
                  --pts-yx {kwargs['pts_yx']} --pts-lalo {kwargs['pts_lalo']} --pts-file {kwargs['pts_file']} "

    # plot/modify the reference point if available
    if all(ii in atr for ii in ['REF_LAT', 'REF_LON']):
        opt += f" --ref-lalo {atr['REF_LAT']} {atr['REF_LON']} "

    keys = v.keys()
    # plot each field
    for i, (ax, k, vlim, unit) in enumerate(zip(axs, keys, kwargs['vlims'], kwargs['units'])):
        print('Plot data no. {}: <{}>'.format(i, k))
        data = v[k]

        # view.py cmd
        ui = unit.split('/')[0]
        cmd  = f" {cmd0} -u {ui} -v {vlim[0]} {vlim[1]} " + opt
        inps = view.prep_slice(cmd)[2]

        # plot the slice
        ax, inps, im, _ = view.plot_slice(ax, data, atr, inps)

        # add ocean and lakes from cartopy features
        ax.add_feature(cfeature.OCEAN, zorder=.5, alpha=.7)
        ax.add_feature(cfeature.LAKES, zorder=.5, alpha=.7)

        # plot colorbar in subplots
        if not kwargs['side_cbar']:
            if not one_cbar:
                colorbar_inset(fig, ax, data, im, atr, unit=inps.disp_unit)
            else:
                if i == len(axs)-1:
                    colorbar_inset(fig, ax, data, im, atr, unit=inps.disp_unit, bound_fontsize=0)

        # reference point if available
        #if any(['REF_LON', 'REF_LAT']) in atr:
        #    ax.scatter(atr['REF_LON'], atr['REF_LON'], marker='s', s=50, c='k')

        # labeling the figure
        if inps.coord_unit == 'radar':
            ax.set_xlabel('Range pixel',   fontsize=plt.rcParams['font.size'])
            ax.set_ylabel('Azimuth pixel', fontsize=plt.rcParams['font.size'])
        ax.set_title(k, fontsize=plt.rcParams['font.size'], pad=kwargs['title_pad'])
        ax.add_feature(cfeature.OCEAN, alpha=0.7)
        ax.set_xlim([W, E])
        ax.set_ylim([S, N])

        # scale bar, LoS legend, mute common y-axis
        if i == 0:
            scalebar_inset(fig,  ax, inps)
            satellite_inset(fig, ax, atr, sat_corner=one_cbar)
        elif i>0 and kwargs['sharey']:
            ax.axes.yaxis.set_visible(False)

    # Add super title
    if super_title:
        sup_y = know_loc_from_subplots(fig, yoff=kwargs['suptity'])
        fig.suptitle(super_title, fontsize=plt.rcParams['font.size']*1.2, y=sup_y)

    # Plot the common colorbar by the side
    if kwargs['side_cbar']:
        cax = make_cbar_ax_side(fig, yloc='center', ch=0.5, cw_in=0.3)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(ylabel=inps.disp_unit, rotation=270, labelpad=28)

    # Save figure
    if outfile:
        save_fig(fig, outfile, **kwargs)

    plt.show()
