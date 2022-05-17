#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################

# Usage:
#   from codes import plot_bulkMotion

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import scipy
import sklearn
import cvxpy as cp

## Load from MintPy
from mintpy.utils import readfile, utils as ut
from mintpy.objects.coord import coordinate


from mintpy.utils import readfile, writefile, utils as ut
from mintpy.mask import mask_matrix
from mintpy.diff import diff_file
from mintpy import reference_point
from mintpy.save_gmt import get_geo_lat_lon
from mintpy.solid_earth_tides import prepare_los_geometry

## Load my codes
import misc

plt.rcParams.update({'font.size': 16})


##################################################################


def cvxpy_reg(x, y, p=1, report=False):
    # Define and solve the CVXPY problem.
    A = np.vstack([x, np.ones(len(x))]).T
    m = cp.Variable(2)          # 2 parameter space (slope & intercept)
    cost = cp.norm(A@m-y, p=p)  # L1 norm cost function
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    y_pred = A @ m.value
    if report:
        print("The optimal value is", prob.value)
        print("The optimal model is")
        print(m.value)
        print("The L{} norm of the residual = {} ".format(p, cp.norm(A@m-y, p=p).value))
    return m.value, y_pred


def sklearn_L2_reg(x, y, report=False):
    # x need to be reshape(-1,1)
    # Create an instance of a linear regression model
    # and fit it to the data with the fit() function
    fit = sklearn.linear_model.LinearRegression().fit(x, y)
    # Obtain the coefficient of determination by calling the model
    # with the score() function, then print the coefficient:
    r_sq = fit.score(x, y)
    y_pred = fit.predict(x)
    if report:
        print('coefficient of determination:', r_sq)
        print('slope:', fit.coef_[0])
        print('intercept:', fit.intercept_)
    return fit, y_pred


def sklearn_L1_reg(x, y, report=False):
    fit = sklearn.linear_model.Lasso(alpha=1.0)
    fit.fit(x, y)
    r_sq = fit.score(x, y)
    y_pred = fit.predict(x)
    if report:
        print('coefficient of determination:', r_sq)
        print('slope:', fit.coef_[0])
        print('intercept:', fit.intercept_)
    return fit, y_pred


def scipy_L2_reg(x, y, report=False):
    fit = scipy.stats.linregress(x, y)
    y_pred = fit.intercept + fit.slope * x
    if report:
        print('slope:', fit.slope)
        print('intercept:', fit.intercept)
    return fit, y_pred


def flatten_isnotnan(x):
    x = x.flatten()[~np.isnan(x.flatten())]
    return x


def scatter_fields(data1, data2, labels=['data1','data2'], vlim=[-20,20], title='', outfile=False):
    ## linear fit to the trend
    x = data1.flatten()[~np.isnan(data2.flatten())]
    y = data2.flatten()[~np.isnan(data2.flatten())]

    fit, y_pred = scipy_L2_reg(x, y)

    # plot
    fig, ax = plt.subplots(figsize=[6,6])
    plt.scatter(x, y)
    plt.scatter(x, y_pred, s=0.3, label='y=ax+b \n a={:.3f}±{:.3f}, b={:.3f}'.format(fit.slope, fit.stderr, fit.intercept))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(vlim[0], vlim[1])
    plt.legend(loc='upper left')
    plt.title(title)

    # Save figure
    if outfile:
        misc.save_fig(fig, outfile, dpi=150)
        plt.close()
    else:
        plt.show()
    return fit


def plot_range_ramp(data1, data2=None, range_dist=None, type=None, latitude=None, titstr='', super_title=None, vlim=[None, None], norm='L2', plot=True, outfile=False, cmap='viridis'):

    # compare different inputs
    if data2 is not None:
        if data1.shape == data2.shape:
            ncols = 3
            if isinstance(titstr, str):
                titstr = ['Data LOS velocity',
                          'Plate bulk motion LOS projection',
                          'Corrected LOS velocity']
    else:
        ncols = 1

    # get range and azimuth bins
    length, width = data1.shape
    rbins = np.tile(np.arange(width), (length, 1))
    abins = np.tile(np.arange(length), (width, 1)).T

    # range_bin, groundRangeDistance, or slantRangeDistance for x-axis
    if range_dist is None:
        xarry      = np.array(rbins)
        yarry      = np.array(abins)
        xlabel     = 'Range bin'
        ylabel     = 'Azimuth bin'
        slope_unit = 'mm/yr/track'
        factor     = float(width)
    else:
        xarry      = np.array(range_dist)
        xlabel     = '{} range [km]'.format(type.title())
        slope_unit = 'mm/yr/km'
        factor     = 1.0
        if latitude is None:
            yarry  = np.array(abins)
            ylabel = 'Azimuth bin'
        else:
            yarry  = np.array(latitude)
            ylabel = r'Latitude [$^\circ$]'

    # Plot single scatter plot
    if ncols == 1:
        if plot:
            # Set the figure
            fig, ax = plt.subplots(figsize=[8,8], ncols=ncols, sharey=True)
        x = xarry.flatten()[~np.isnan(data1.flatten())]
        y = data1.flatten()[~np.isnan(data1.flatten())]
        c = yarry.flatten()[~np.isnan(data1.flatten())]
        y = y - np.nanmedian(y)
        print('Range ramp scatter plot shifted by median {}'.format(np.nanmedian(y)))
        print('Ground range min/max:', np.nanmin(xarry), np.nanmax(xarry))
        print('Valid (non-nan pixels) ground range min/max:', np.nanmin(x), np.nanmax(x))

        # linear fit to the trend
        if norm == 'L2':
            fit, y_pred  = scipy_L2_reg(x, y)
            params_legend = 'y=ax+b, a={:.3f} ± {:.2e}\n'.format(fit[0], fit.stderr)
        elif norm == 'L1':
            fit, y_pred  = cvxpy_reg(x, y, p=1)
            params_legend = 'y=ax+b, a={:.3f}\n'.format(fit[0])

        if range_dist is None:
            params_legend += 'slope = {:.3f} {:s}'.format(fit[0]*factor, slope_unit)
        else:
            range_span = np.max(x) - np.min(x)
            print('{} range distance spans {:.1f} km'.format(type.title(), range_span))
            params_legend += 'slope = {:.3f} {:s}'.format(fit[0]*factor, slope_unit)
            show_legend    = '{:.3f} mm/yr/track'.format(fit[0]*range_span)
            params_legend += '\n'+show_legend

        if plot:
            sc = ax.scatter(x, y, s=1.5, c=c, cmap=cmap)
            sc.set_rasterized(True)
            ax.plot(x, y_pred, lw=2, label=show_legend, c='r')
            ax.legend(loc='lower left', frameon=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('LOS velocity [mm/yr]')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.ax.set_ylabel(ylabel='Azimuth', rotation=270, labelpad=20)
            ax.set_title(titstr, fontsize=plt.rcParams['font.size'], pad=20.0)
            ax.set_ylim(vlim[0], vlim[1])
            ax.set_xlim(0, 250)
            ax.set_xticks(np.arange(0,300,50))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(direction='in', length=10, width=2, colors='k', top=True, right=True)
            [sp.set_linewidth(2) for sp in ax.spines.values()]
            [sp.set_zorder(10)   for sp in ax.spines.values()]
            plt.show()

    # Plot scatter plots comparison
    if ncols == 3:
        if plot:
            # Set the figure
            fig, axs = plt.subplots(figsize=[7*ncols,8], ncols=ncols, sharey=True, gridspec_kw={'wspace':.03}, constrained_layout=True)

        for i, (ax, data) in enumerate(zip(axs, [data1, data2, data1-data2])):
            x = xarry.flatten()[~np.isnan(data.flatten())]
            y =  data.flatten()[~np.isnan(data.flatten())]
            c = yarry.flatten()[~np.isnan(data.flatten())]
            y = y - np.nanmedian(y)
            print('Range ramp scatter plot shifted by median {}'.format(np.nanmedian(y)))
            print('Ground range min/max:', np.nanmin(xarry), np.nanmax(xarry))
            print('Valid (non-nan pixels) ground range min/max:', np.nanmin(x), np.nanmax(x))

            # linear fit to the trend
            if norm == 'L2':
                fit, y_pred  = scipy_L2_reg(x, y)
                params_legend = 'y=ax+b, a={:.3f} ± {:.2e}\n'.format(fit[0], fit.stderr)
            elif norm == 'L1':
                fit, y_pred  = cvxpy_reg(x, y, p=1)
                params_legend = 'y=ax+b, a={:.3f}\n'.format(fit[0])

            if range_dist is None:
                params_legend += 'slope = {:.3f} {:s}'.format(fit[0]*factor, slope_unit)
            else:
                range_span = np.max(x) - np.min(x)
                print('{} range distance spans {:.1f} km'.format(type.title(), range_span))
                params_legend += 'slope = {:.3f} {:s}'.format(fit[0]*factor, slope_unit)
                show_legend    = '{:.3f} mm/yr/track'.format(fit[0]*range_span)
                params_legend += '\n'+show_legend

            if plot:
                sc = ax.scatter(x, y, s=1.5, c=c, cmap=cmap)
                sc.set_rasterized(True)
                ax.plot(x, y_pred, lw=3, label=show_legend, c='crimson')
                leg=ax.legend(loc='lower left', markerscale=0.6, frameon=False)
                leg.get_lines()[0].set_linewidth(5)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('LOS velocity [mm/yr]')
                ax.set_title(titstr[i], fontsize=plt.rcParams['font.size'], pad=20.0)
                ax.set_ylim(vlim[0], vlim[1])
                ax.set_xlim(0, 250)
                ax.set_xticks(np.arange(50,250,50)) # skip ticks at 0 and 250 for cleaness
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.tick_params(direction='in', length=10, width=2, colors='k', top=True, right=True)
                [sp.set_linewidth(2) for sp in ax.spines.values()]
                [sp.set_zorder(10)   for sp in ax.spines.values()]
                if i>0:
                    ax.tick_params(labelleft=False)
                    ax.set(ylabel=None)

        if plot:
            if super_title:
                sup_y = misc.know_loc_from_subplots(fig, yoff=0.8)
                fig.suptitle(super_title, fontsize=plt.rcParams['font.size']*1.2, y=sup_y)

            #cax = misc.make_cbar_ax_side(fig, yloc='center', ch=1.0, cw_in=0.3)
            #cbar = plt.colorbar(sc, cax=cax)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.ax.set_ylabel(ylabel=ylabel, rotation=270, labelpad=28)
            [sp.set_linewidth(2) for sp in cbar.ax.spines.values()]

    # Save figure
    if outfile:
        misc.save_fig(fig, outfile, dpi=150)

    if plot:
        plt.show()

    return fit[0]*factor


def plot_dem_var(data, dem, titstr='', vlim=[None, None], plot=True):
    # flatten the field for scatter plot
    length, width = data.shape
    abins = np.tile(np.arange(length), (width, 1)).T
    vflat = data.flatten()

    # linear fit to the trend
    x = dem.flatten()
    x = x[~np.isnan(data.flatten())]
    y = flatten_isnotnan(data)
    fit, y_pred = scipy_L2_reg(x, y)
    params_legend = 'y=ax+b \n a={:.3f}±{:.3f}, b={:.3f}'.format(fit.slope, fit.stderr, fit.intercept)

    if plot:
        plt.figure(figsize=[12,8])
        sc = plt.scatter(dem, vflat, s=0.1, c=abins)
        plt.plot(x, y_pred, lw=2, label=params_legend, c='r')
        plt.legend(loc='upper left')
        plt.xlabel('Elevation from DEM [meter]')
        plt.ylabel('LOS velocity [mm/yr]')
        plt.colorbar(sc, label='Azimuth bin')
        plt.title(titstr)
        plt.ylim(vlim[0], vlim[1])
        plt.show()
    return fit.slope * width


def get_geo_lat_lon(atr):
    X_FIRST = float(atr['X_FIRST'])
    Y_FIRST = float(atr['Y_FIRST'])
    X_STEP = float(atr['X_STEP'])
    Y_STEP = float(atr['Y_STEP'])
    W = int(atr['WIDTH'])
    L = int(atr['LENGTH'])
    Y_END = Y_FIRST + L*Y_STEP
    X_END = X_FIRST + W*X_STEP

    X = np.linspace(X_FIRST, X_END, W)
    Y = np.linspace(Y_FIRST, Y_END, L)
    XI,YI = np.meshgrid(X,Y)
    return YI, XI



def compute_ground_range(atr, rs, z):
    """
    atr     MintPy attributes from e.g., gemetryGeo.h5
    rs      slant range distance [m]
    z       DEM elevation [m]
    """

    Re    = float(atr['EARTH_RADIUS'])    # approx. Earth's radius
    h     = float(atr['HEIGHT'])          # satellite height

    # angle at the center of the Earth
    gamma = np.arccos(( (Re+h)**2 + (Re+z)**2 - rs**2 ) / ( 2*(Re+h)*(Re+z) ))

    # set the zero ground range at the nearest range pixel
    rg  = Re * gamma   # ground range
    rg -= np.nanmin(rg)

    return rg


def prepare_range_geometry(geom_file):
    """Prepare LOS geometry data/info in geo-coordinates
    Parameters: geom_file  - str, path of geometry file
    Returns:    range_s    - 2D np.ndarray, slant range distace in km
                range_g    - 2D np.ndarray, ground range distance in km
                atr        - dict, metadata in geo-coordinate
    """
    print('prepare range distance in geo-coordinates from file: {}'.format(geom_file))
    atr = readfile.read_attribute(geom_file)
    if 'slantRangeDistance' in readfile.get_dataset_list(geom_file):
        range_s  = readfile.read(geom_file, datasetName='slantRangeDistance')[0]
        height   = readfile.read(geom_file, datasetName='height')[0]
        range_g = compute_ground_range(atr, range_s, height)
        range_s  /= 1e3
        range_g /= 1e3
    return range_s, range_g, atr



def plot_enulos(v, inc_deg, head_deg, ref=False, display=True, display_more=False, cmap='RdYlBu_r'):
    # v            [E, N, U] floats; Three-component model motion (ENU); unit: mm/yr
    # inc_deg      an array of floats (length * width); unit: degrees
    # head_deg     an array of floats (length * width); unit: degrees
    # ref          reference pixel for v_los

    if len(v.shape) == 1:
        v_los = ut.enu2los(v[0], v[1], v[2], inc_angle=inc_deg, head_angle=head_deg)
        disp         = dict()
        disp['inc']  = inc_deg
        disp['head'] = head_deg
        disp['ve']   = v[0] * np.ones_like(inc_deg)
        disp['vn']   = v[1] * np.ones_like(inc_deg)
        disp['vu']   = v[2] * np.ones_like(inc_deg)
        disp['v_los']= v_los    # sign convention: positive for motion towards satellite

    elif len(v.shape) == 3:
        v_los = ut.enu2los(v[:,:,0], v[:,:,1], v[:,:,2], inc_angle=inc_deg, head_angle=head_deg)
        disp         = dict()
        disp['inc']  = inc_deg
        disp['head'] = head_deg
        disp['ve']   = v[:,:,0]
        disp['vn']   = v[:,:,1]
        disp['vu']   = v[:,:,2]
        disp['v_los']= v_los    # sign convention: positive for motion towards satellite

    # Take the reference pixel from the middle of the map (for v_los only)
    if ref:
        if ref is True:
            idx = np.array(v_los.shape) // 2
        else:
            idx = ref
        V_ref = disp['v_los'][idx[0], idx[1]]
        if np.isnan(V_ref):
            print('Reference point is NaN, choose another point!')
            sys.exit(1)
        disp['v_los'] -= V_ref

    if display:
        ## Make a quick plot
        fontsize_orig = float(plt.rcParams['font.size'])
        plt.rcParams['font.size'] = np.clip(fontsize_orig, a_min=None, a_max=20)
        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=[24,8], sharey=True, gridspec_kw={'wspace':0.1}, constrained_layout=True)
        for i, key in enumerate(disp):
            im = axs[i].imshow(disp[key], cmap=cmap)
            fig.colorbar(im, ax=axs[i], fraction=0.05)
            axs[i].set_title(key)
        plt.show()

        # report
        print('Min. LOS motion = {:.3f}'.format(np.nanmin(disp['v_los'])))
        print('Max. LOS motion = {:.3f}'.format(np.nanmax(disp['v_los'])))
        print('Dynamic range of LOS motion = {:.3f}'.format(np.nanmax(disp['v_los'])-np.nanmin(disp['v_los'])))
        if display_more:
            ## Make a plot about sin, cos
            unit_vector = los_unit_vector(np.deg2rad(inc_deg), np.deg2rad(head_deg))
        plt.rcParams['font.size'] = float(fontsize_orig)


def reference_dataDict(v, atr, **kwargs):
    """
    Referencing for visualization the data
    + If None, then use the original reference from the file
    + If False, then remove the median from the corresponding data (no single reference pixel is plotted)
    """
    coord = coordinate(atr)
    if kwargs['refpoint']:
        refpoint = np.array(kwargs['refpoint'])
        if any([isinstance(i, float) for i in refpoint]):
            ref_y = coord.lalo2yx(refpoint[0], coord_type='lat')
            ref_x = coord.lalo2yx(refpoint[1], coord_type='lon')
            atr['REF_LAT'], atr['REF_LON'] = refpoint
            atr['REF_Y'], atr['REF_X'] = ref_y, ref_x
        else:
            ref_y, ref_x = refpoint
            atr['REF_LAT'] = coord.yx2lalo(refpoint[0], coord_type='y')
            atr['REF_LON'] = coord.yx2lalo(refpoint[1], coord_type='x')
            atr['REF_Y'], atr['REF_X'] = ref_y, ref_x
        print('Use new reference point at lat/lon = {} / {}'.format(atr['REF_LAT'], atr['REF_LON']))
    elif kwargs['refpoint'] is None:
        ref_y, ref_x = int(atr['REF_Y']), int(atr['REF_X'])
        atr['REF_LAT'] = coord.yx2lalo(ref_y, coord_type='y')
        atr['REF_LON'] = coord.yx2lalo(ref_x, coord_type='x')
        print('Use original reference point at lat/lon = {} / {}'.format(atr['REF_LAT'], atr['REF_LON']))
    elif kwargs['refpoint'] is False:
        print('No reference point, remove a median from each dataset!!')

    for ikey in v:
        data_key  = f'<{ikey}>'

        # mask the weird pixels by identifying modes
        N_thres      = 50
        vals, counts = np.unique(v[ikey], return_counts=True)
        index        = np.argsort(counts, axis=0)[::-1]
        modes        = np.where(counts[index]>N_thres)[0]
        if len(modes)>0:
            #print('Set weird modes to NaN: {}'.format(ikey))
            for m in modes:
                #print('  Count: {} \t value: {}'.format(counts[index[m]], vals[index[m]]))
                v[ikey][v[ikey]==vals[index[m]]] = np.nan

        # shift by the reference pixel
        if kwargs['refpoint'] is not False:
            if np.isnan(v[ikey][ref_y, ref_x]):
                print('Reference point ({},{}) is NaN in {}, choose another point!'.format(ref_y, ref_x, ikey))
                sys.exit(1)
            else:
                tmp_ref   = v[ikey][ref_y, ref_x]
                v[ikey]  -= tmp_ref
                print('Reference data: {:30s} \t shifted by {}'.format(data_key, tmp_ref))

        # shift by the median
        elif kwargs['refpoint'] is False:
            tmp_ref   = np.nanmedian(v[ikey])
            v[ikey]  -= tmp_ref
            print('Reference data: {:30s} \t shifted by {}'.format(data_key, tmp_ref))





############################


def plot_inputs(self, ref=False, display=True, cmap='RdBu_r'):
    ## Plot {e, n, u, inc, head, vlos}
    if ref:
        self.ref = ref
    plot_enulos(self.V_pmm_enu, self.inc_deg, self.head_deg, self.ref, display=display, cmap=cmap)



def plot_ramp(self, track=False, indata=None, compare=False, vlim=[None]*2, distance='ground', sn=None, outfile=False):
    ## Plot forward model range ramp
    # get title name
    if track:
        self.name = str(track)
    else:
        self.name = self.orbit

    # use slant range
    if distance == 'ground':
        range_dist = np.array(self.groundrange)
    elif distance == 'slant':
        range_dist  = np.array(self.slantrange)
    else:
        range_dist = None

    # copy the input data
    if indata is not None:
        data = np.array(indata)

    # fit whole data or only to a subset
    lat   = np.array(self.lat)
    V_los = np.array(self.V_pmm)
    if sn is not None:
        if any([isinstance(k, float) for k in sn]):
            # dealing with latitude
            idx = (lat>sn[0])*(lat<sn[1])
            lat[~idx]   = np.nan
            V_los[~idx] = np.nan
            if indata is not None:
                data[~idx] = np.nan
            if distance:
                range_dist[~idx] = np.nan
        else:
            # dealing with azimuth bin
            lat   = lat[sn[0]:sn[1]]
            V_los = V_los[sn[0]:sn[1]]
            if indata is not None:
                data = data[sn[0]:sn[1]]
            if distance:
                range_dist = range_dist[sn[0]:sn[1]]

    if len(self.V_pmm_enu.shape) == 1:
        vstr = '({}E, {}N mm/y)'.format(self.V_pmm_enu[0], self.V_pmm_enu[1])
    elif len(self.V_pmm_enu.shape) == 3:
        vstr = '(PMM)'


    ## How many subplots you need
    if indata is None:
        # show model
        suptit = '{}'.format(self.name)
        titstr = 'Plate bulk motion {}'.format(vstr)
        self.rangeslope = plot_range_ramp(data1=V_los, range_dist=range_dist, type=distance, latitude=lat, titstr=titstr, super_title=suptit, vlim=vlim, outfile=outfile)
    elif (indata is not None) and (compare is True):
        # show data and compare with model
        suptit = '{}'.format(self.name)
        titstr = ['Data LOS velocity',
                    'Plate bulk motion {}'.format(vstr),
                    'Corrected LOS velocity']
        self.rangeslope = plot_range_ramp(data1=data, data2=V_los, range_dist=range_dist, type=distance, latitude=lat, titstr=titstr, super_title=suptit, vlim=vlim, outfile=outfile)
        self.V_data = np.array(indata)
    else:
        # show data
        suptit = '{}'.format(self.name)
        titstr = '{} Data LOS velocity'.format(self.name)
        self.rangeslope = plot_range_ramp(data1=data, range_dist=range_dist, type=distance, latitude=lat, titstr=titstr, super_title=suptit, vlim=vlim, outfile=outfile)
        self.V_data = np.array(indata)
    return self.rangeslope




############################



def plot_motion(vDict, atr, dName, **kwargs):
    """
    dDict           dict();             a dictioanry of where the file paths are for potentially different datasets
    dName           str;                dictionary key of the dataset of interest
    kwargs          dict()              mostly for plotting details
    """

    if 'pmm'         not in kwargs.keys():   kwargs['pmm']             = 'MyPMM'
    if 'font_pmm'    not in kwargs.keys():   kwargs['font_pmm']        = 14
    if 'plot_pmm'    not in kwargs.keys():   kwargs['plot_pmm']        = False
    if 'plot_ins'    not in kwargs.keys():   kwargs['plot_ins']        = False
    if 'plot_rmp'    not in kwargs.keys():   kwargs['plot_rmp']        = False
    if 'plot_ion'    not in kwargs.keys():   kwargs['plot_ion']        = False
    if 'plot_maj'    not in kwargs.keys():   kwargs['plot_maj']        = True
    if 'fig_ext'     not in kwargs.keys():   kwargs['fig_ext']         = 'pdf'
    if 'vlim_i'      not in kwargs.keys():   kwargs['vlim_i']          = [None,None]
    if 'vlim_b'      not in kwargs.keys():   kwargs['vlim_b']          = [None,None]
    if 'vlim_r'      not in kwargs.keys():   kwargs['vlim_r']          = [None,None]
    if 'refpoint'    not in kwargs.keys():   kwargs['refpoint']        = None
    if 'saveh5_dir'  not in kwargs.keys():   kwargs['saveh5_dir']      = '../out_h5'
    if 'pic_dir'     not in kwargs.keys():   kwargs['pic_dir']         = '../pic'
    if 'suffix'      not in kwargs.keys():   kwargs['suffix']          = ''
    if 'cmap'        not in kwargs.keys():   kwargs['cmap']            = 'RdBu_r'
    if 'dpi'         not in kwargs.keys():   kwargs['dpi']             = 300

    if (len(kwargs['suffix'])>0) and not kwargs['suffix'].startswith('_'):
        kwargs['suffix'] = '_' + kwargs['suffix']

    print('Analyze the data from: ', dName)
    name, track_id = dName.split()


    ### Reference point for visualization the velocity / remove median
    reference_dataDict(vDict, atr, **kwargs)


    # The ramp in the real data
    if kwargs['plot_rmp']:
        fn = f"{kwargs['pic_dir']}/{name}_{track_id}{kwargs['suffix']}_rampfit.{kwargs['fig_ext']}"
        plot_ramp(track=f'{name}_{track_id}', indata=vDict['LOS velocity (iono corr)'], compare=True, vlim=kwargs['vlim_r'], sn=None, outfile=fn)

    # ~~Major result~~: plot to compare the effect of bulk motion compensation
    if kwargs['plot_maj']:
        v_show = {ikey: vDict[ikey] for ikey in ['LOS velocity (iono corr)', 'LOS Plate bulk motion', 'LOS velocity (plate motion corr)']}
        kwargs['vlims'] = list(kwargs['vlim_b'])
        fn = f"{kwargs['pic_dir']}/{name}_{track_id}{kwargs['suffix']}_bmCorr.{kwargs['fig_ext']}"
        misc.plot_imgs(v_show, atr, tobj.dem, super_title=f'{name}_{track_id}', outfile=fn, **kwargs)

