#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mintpy import view
from mintpy.utils import readfile, utils as ut, plot as pp
from mintpy.utils import ptime



#############################################################

def read_mask(mask_file):
    mask_data = readfile.read(mask_file)[0]
    mask      = 1*(mask_data==1)
    return mask


def read_img(fname, mask=None):
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


def colorbar_inset(ax, im, **kwargs):
    # Initialize the kwargs
    if 'unit'            not in kwargs.keys():   kwargs['unit']  = ''
    if 'ub'              not in kwargs.keys():   kwargs['ub'] = 99.8
    if 'lb'              not in kwargs.keys():   kwargs['lb'] = 0.2
    if 'box_loc'         not in kwargs.keys():   kwargs['box_loc'] = (0., 0., .6, .08)
    if 'box_alpha'       not in kwargs.keys():   kwargs['box_alpha'] = 0.9
    if 'orient'          not in kwargs.keys():   kwargs['orient'] = 'horizontal'
    if 'unit_fontsize'   not in kwargs.keys():   kwargs['unit_fontsize'] = 16
    if 'bound_fontsize'  not in kwargs.keys():   kwargs['bound_fontsize'] = 12
    if 'ticks_fontsize'  not in kwargs.keys():   kwargs['ticks_fontsize'] = 16

    upper_val = np.nanpercentile(data, kwargs['ub'])
    lower_val = np.nanpercentile(data, kwargs['lb'])

    #cbck = inset_axes(ax, width="60%", height="7.5%", loc='lower left',
    #        bbox_transform=ax.transAxes, bbox_to_anchor=(-0.02,-0.015,1,1))
    # colorbar background
    cbck = ax.inset_axes(kwargs['box_loc'])
    cbck.set_facecolor('w')
    cbck.patch.set_alpha(kwargs['box_alpha'])
    cbck.get_xaxis().set_visible(False)
    cbck.get_yaxis().set_visible(False)
    cbar = inset_axes(cbck, width="80%", height="40%",loc='center',
            bbox_transform=cbck.transAxes, bbox_to_anchor=(0, 0.15, 1, 1))
    fig.colorbar(im, cax=cbar, orientation=kwargs['orient'])
    cbar.text(0.5, 0.5, kwargs['unit'], ha='center', va='center', fontsize=kwargs['unit_fontsize'], transform=cbar.transAxes)
    cbar.text(0.02, 0.5, '({:.1f})'.format(lower_val),  ha='left', va='center', fontsize=kwargs['bound_fontsize'], transform=cbar.transAxes)
    cbar.text(0.98, 0.5, '({:.1f})'.format(upper_val), ha='right', va='center', fontsize=kwargs['bound_fontsize'], transform=cbar.transAxes)
    cbar.tick_params(labelsize=kwargs['ticks_fontsize'])


def plot_imgs(v, meta, cmd, outfile=False, **kwargs):

    # Initialize the kwargs
    if 'coord'   not in kwargs.keys():   kwargs['coord']  = 'geo'
    if 'wspace'  not in kwargs.keys():   kwargs['wspace'] = 0.02
    if 'wratio'  not in kwargs.keys():   kwargs['wratio'] = 1.5


    ## get attributes
    if kwargs['coord'] == 'geo':
        xlabel = 'Longitude'
        ylabel = 'Latitude'
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['X_FIRST'])
        x_step    = float(meta['X_STEP'])
        y_min     = float(meta['Y_FIRST'])
        y_step    = float(meta['Y_STEP'])
        lats      = np.arange(y_min,length*y_step+y_min, y_step)
        lons      = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = float(meta['REF_LAT'])
        ref_lon   = float(meta['REF_LON'])
        geo_extent= [lons[0],lons[-1],lats[-1],lats[0]]
    elif kwargs['coord'] == 'rdr':
        xlabel = 'Range bin'
        ylabel = 'Azimuth bin'
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['XMIN'])
        x_step    = 1
        y_min     = float(meta['YMIN'])
        y_step    = 1
        ycoords   = np.arange(y_min,length*y_step+y_min, y_step)
        xcoords   = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = None
        ref_lon   = None
        geo_extent= [xcoords[0],xcoords[-1],ycoords[-1],ycoords[0]]


    # How many subplots to plot
    nrows, ncols = 1, len(v)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*8, kwargs['wratio']*8], sharey=True, gridspec_kw={'wspace':kwargs['wspace']})
    if len(v) == 1:
        axs = [axs]
    if len(vlims) != len(axs):
        vlims = [vlims[-1]]*len(axs)
    if isinstance(unit, str):
        unit = [unit]
    if len(unit) != len(axs):
        unit = [unit[-1]]*len(axs)


    data, atr ,inps = view.prep_slice(cmd)
    ax, inps, im, cbar = view.plot_slice(ax, data, atr, inps)
    plt.show()


    keys = v.keys()
    # plot each field
    for i, (ax, k, vlim) in enumerate(zip(axs, keys, vlims)):
        print('plotting', k)

        data, atr ,inps = view.prep_slice(cmd)
        ax, inps, im, cbar = view.plot_slice(ax, data, atr, inps)

        im = ax.imshow(v[k],    extent=geo_extent, cmap=cmap, vmin=vlim[0], vmax=vlim[1], alpha=alpha, interpolation=inter)

        # colorbar
        high_val = np.nanpercentile(v[k], 99.8)
        low_val  = np.nanpercentile(v[k],  0.2)
        cbck = inset_axes(ax, width="60%", height="7.5%", loc='lower left',
                bbox_transform=ax.transAxes, bbox_to_anchor=(-0.02,-0.015,1,1))    # colorbar background
        cbck.set_facecolor('w')
        cbck.patch.set_alpha(0.7)
        cbck.get_xaxis().set_visible(False)
        cbck.get_yaxis().set_visible(False)
        cbar = inset_axes(cbck, width="90%", height="45%",loc='upper center',
                bbox_transform=cbck.transAxes, bbox_to_anchor=(0, 0.1, 1, 1))
        fig.colorbar(im, cax=cbar, orientation='horizontal')
        cbar.text(0.5, 0.5, unit[i], ha='center', va='center', fontsize=16, transform=cbar.transAxes)
        cbar.text(0.02, 0.5, '({:.1f})'.format(low_val),  ha='left', va='center', fontsize=12, transform=cbar.transAxes)
        cbar.text(0.98, 0.5, '({:.1f})'.format(high_val), ha='right', va='center', fontsize=12, transform=cbar.transAxes)

        if pts:
            pts = np.array(pts).reshape(-1,2)
            ax.scatter(pts[:,1], pts[:,0], marker='s', c='k', s=30)

        # scale bar & xlabel
        if coord == 'geo':
            if i == 0:
                if not all(x is None for x in bbox):
                    cent_lat = np.mean(bbox[2:])
                    span_lon = bbox[1]-bbox[0]
                else:
                    cent_lat = np.mean(lats)
                    span_lon = np.max(lons)-np.min(lons)
                r_earth    = 6378.1370
                km_per_lon = np.pi/180 * r_earth * np.cos(np.pi*cent_lat/180)
                span_km    = span_lon * km_per_lon
                scal_km    = round(span_km/3/10)*10
                scal_lon   = scal_km / km_per_lon
                scax  = inset_axes(ax, width=scal_lon, height="1%", loc='upper left',
                        bbox_transform=ax.transAxes, bbox_to_anchor=(0.04, -0.03, 1, 1))
                scax.set_facecolor('k')
                scax.axes.xaxis.set_ticks([])
                scax.axes.yaxis.set_ticks([])
                scax.set_xlabel('{:d} km'.format(scal_km), fontsize=16, labelpad=2)
                ax.set_ylabel(ylabel, fontsize=font_size+4)

        # reference point if available
        if ref_lon and ref_lat:
            ax.scatter(ref_lon, ref_lat, marker='s', s=50, c='k')

        # others
        ax.set_title(k)
        ax.set_xlabel(xlabel, fontsize=16+4)
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])

    # output
    if outf:
        if not os.path.exists(picdir):
            os.makedirs(picdir)
        out_file = f'{picdir}/{outf}.png'
        plt.savefig(out_file, bbox_inches='tight', dpi=300)
        print('save to file: '+out_file)
    plt.show()






def plot_imgs_old(v, meta, dem=None, vlims=[[None,None]], bbox=[None]*4, unit='mm/yr', coord='geo',
              inter='nearest', cmap='RdYlBu_r', alpha=0.7, wspc=0.02, bratio=1.5, pts=False,
              picdir='./pic', outf=False):

    font_size=16
    plt.rcParams.update({'font.size': font_size})

    ## get attributes
    if coord == 'geo':
        xlabel = 'Longitude'
        ylabel = 'Latitude'
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['X_FIRST'])
        x_step    = float(meta['X_STEP'])
        y_min     = float(meta['Y_FIRST'])
        y_step    = float(meta['Y_STEP'])
        lats      = np.arange(y_min,length*y_step+y_min, y_step)
        lons      = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = float(meta['REF_LAT'])
        ref_lon   = float(meta['REF_LON'])
        geo_extent= [lons[0],lons[-1],lats[-1],lats[0]]
    elif coord == 'rdr':
        xlabel = 'Range bin'
        ylabel = 'Azimuth bin'
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['XMIN'])
        x_step    = 1
        y_min     = float(meta['YMIN'])
        y_step    = 1
        ycoords   = np.arange(y_min,length*y_step+y_min, y_step)
        xcoords   = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = None
        ref_lon   = None
        geo_extent= [xcoords[0],xcoords[-1],ycoords[-1],ycoords[0]]

    # how many subplots to plot
    nrows, ncols = 1, len(v)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*8,bratio*8], sharey=True, gridspec_kw={'wspace':wspc})
    if len(v) == 1:
        axs = [axs]
    if len(vlims) != len(axs):
        vlims = [vlims[-1]]*len(axs)
    if isinstance(unit, str):
        unit = [unit]
    if len(unit) != len(axs):
        unit = [unit[-1]]*len(axs)

    keys = v.keys()
    # plot each field
    for i, (ax, k, vlim) in enumerate(zip(axs, keys, vlims)):
        print('plotting', k)
        # plot DEM and the image
        if dem is not None:
            ie = ax.imshow(dem, extent=geo_extent, vmin=-500, vmax=2000)
        im = ax.imshow(v[k],    extent=geo_extent, cmap=cmap, vmin=vlim[0], vmax=vlim[1], alpha=alpha, interpolation=inter)

        # colorbar
        high_val = np.nanpercentile(v[k], 99.8)
        low_val  = np.nanpercentile(v[k],  0.2)
        cbck = inset_axes(ax, width="60%", height="7.5%", loc='lower left',
                bbox_transform=ax.transAxes, bbox_to_anchor=(-0.02,-0.015,1,1))    # colorbar background
        cbck.set_facecolor('w')
        cbck.patch.set_alpha(0.7)
        cbck.get_xaxis().set_visible(False)
        cbck.get_yaxis().set_visible(False)
        cbar = inset_axes(cbck, width="90%", height="45%",loc='upper center',
                bbox_transform=cbck.transAxes, bbox_to_anchor=(0, 0.1, 1, 1))
        fig.colorbar(im, cax=cbar, orientation='horizontal')
        cbar.text(0.5, 0.5, unit[i], ha='center', va='center', fontsize=16, transform=cbar.transAxes)
        cbar.text(0.02, 0.5, '({:.1f})'.format(low_val),  ha='left', va='center', fontsize=12, transform=cbar.transAxes)
        cbar.text(0.98, 0.5, '({:.1f})'.format(high_val), ha='right', va='center', fontsize=12, transform=cbar.transAxes)

        if pts:
            pts = np.array(pts).reshape(-1,2)
            ax.scatter(pts[:,1], pts[:,0], marker='s', c='k', s=30)

        # scale bar & xlabel
        if coord == 'geo':
            if i == 0:
                if not all(x is None for x in bbox):
                    cent_lat = np.mean(bbox[2:])
                    span_lon = bbox[1]-bbox[0]
                else:
                    cent_lat = np.mean(lats)
                    span_lon = np.max(lons)-np.min(lons)
                r_earth    = 6378.1370
                km_per_lon = np.pi/180 * r_earth * np.cos(np.pi*cent_lat/180)
                span_km    = span_lon * km_per_lon
                scal_km    = round(span_km/3/10)*10
                scal_lon   = scal_km / km_per_lon
                scax  = inset_axes(ax, width=scal_lon, height="1%", loc='upper left',
                        bbox_transform=ax.transAxes, bbox_to_anchor=(0.04, -0.03, 1, 1))
                scax.set_facecolor('k')
                scax.axes.xaxis.set_ticks([])
                scax.axes.yaxis.set_ticks([])
                scax.set_xlabel('{:d} km'.format(scal_km), fontsize=16, labelpad=2)
                ax.set_ylabel(ylabel, fontsize=font_size+4)

        # reference point if available
        if ref_lon and ref_lat:
            ax.scatter(ref_lon, ref_lat, marker='s', s=50, c='k')

        # others
        ax.set_title(k)
        ax.set_xlabel(xlabel, fontsize=font_size+4)
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])

    # output
    if outf:
        if not os.path.exists(picdir):
            os.makedirs(picdir)
        out_file = f'{picdir}/{outf}.png'
        plt.savefig(out_file, bbox_inches='tight', dpi=300)
        print('save to file: '+out_file)
    plt.show()

