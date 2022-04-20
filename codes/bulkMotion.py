#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################

# Usage:
#   from codes import bulkMotion

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import scipy
import cvxpy as cp


## Load from MintPy
from mintpy.utils import readfile, utils as ut

## Load plate motion package, unit, etc.
import pandas as pd
from scipy import interpolate
import platemotion
from astropy import units as u

## Load my codes
import sarut.tools.plot as sarplt

plt.rcParams.update({'font.size': 16})


##################################################################

def build_PMM(platename, omega_cart=None, omega_sph=None, plateBound='MRVL', plot=False):

    pkg_dir = os.path.dirname(platemotion.__file__)

    if plateBound == 'MRVL':
        csvfile = f'{pkg_dir}/data/NNR_MRV56.csv'
        bndsdir = 'NnrMRVL_PltBndsLatLon'
    elif plateBound == 'GSRM':
        csvfile = f'{pkg_dir}/data/NNR_GSRMv2.1.csv'
        bndsdir = 'NnrGSRMv2.1_PltBndsLatLon'
    else:
        print('No such plate boundary table built-in: {}'.format(plateBound))
        sys.exit(1)

    df = pd.read_csv(csvfile, header=0, squeeze=True)
    plate_ab = df.set_index('Plate').T.to_dict('records')[0]


    plate = platemotion.Plate.from_file(f'{pkg_dir}/data/{bndsdir}/{plate_ab[platename]}',skiprows=1)
    plate.set_name(platename)

    # Check input variables
    if (omega_cart is None) and (omega_sph is None):
        print('Need to give either omega_cartesian (wxyz) or omega_spherical (euler pole)!!')
        sys.exit(1)

    elif omega_cart is not None:
        print('Input: omega_cartesian (wxyz)')
        omega = np.array(omega_cart) * u.mas/u.yr
        plate.set_omega(omega,'cartesian')

    else:
        print('Input: omega_spherical (euler pole)')
        omega = [omega_sph[0]*u.deg, omega_sph[1]*u.deg, omega_sph[2]*u.deg/u.Ma]
        plate.set_omega(omega,'spherical')


    print('\nPlate: {}'.format(platename))
    print('\nCartesian rotation vector:')
    print(' wx:            {:.4f}'.format(plate.omega_cartesian[0]))
    print(' wy:            {:.4f}'.format(plate.omega_cartesian[1]))
    print(' wz:            {:.4f}'.format(plate.omega_cartesian[2]))
    print('\nEuler pole representation:')
    print(' Latitude:      {:.4f} deg'.format(plate.omega_spherical[0].degree))
    print(' Longitude:     {:.4f} deg'.format(plate.omega_spherical[1].degree))
    print(' Rotation rate: {:.4f}  \n'.format(plate.omega_spherical[2].to(u.deg/u.Ma)))

    if plot:
        plate.plot()
    return plate


def mesh_2dll(lat12, lon12, dlat=0.1, dlon=0.1):
    lats = np.arange(lat12[0], lat12[1], dlat)
    lons = np.arange(lon12[0], lon12[1], dlon)
    Lons, Lats = np.meshgrid(lons, lats)
    return Lats, Lons


def interp_2d3l_grid(data, X, Y, nx, ny, kind):
    ## Interpolate 3-layer 2D arrays individually
    y_new    = np.linspace(np.min(Y), np.max(Y), ny)
    x_new    = np.linspace(np.min(X), np.max(X), nx)
    data_new = np.empty([len(y_new), len(x_new), 3])
    X_new, Y_new = np.meshgrid(x_new, y_new)

    mask = np.ma.masked_invalid(data[:,:,0])

    print('Mask {}. unmask {}'.format(np.sum(mask.mask), np.sum(~mask.mask)))

    x1     = X[~mask.mask]
    y1     = Y[~mask.mask]
    data_0 = data[:,:,0][~mask.mask]
    data_1 = data[:,:,1][~mask.mask]
    data_2 = data[:,:,2][~mask.mask]

    data_new[:,:,0] = interpolate.griddata((x1, y1), data_0.ravel(), (X_new, Y_new), method=kind)
    data_new[:,:,1] = interpolate.griddata((x1, y1), data_1.ravel(), (X_new, Y_new), method=kind)
    data_new[:,:,2] = interpolate.griddata((x1, y1), data_2.ravel(), (X_new, Y_new), method=kind)

    return data_new


def interp_2d3l(data, X, Y, nx, ny, kind):
    ## Interpolate 3-layer 2D arrays individually

    # Check and flip Y array if needed
    if Y[0,0] > Y[-1,0]:
        Y = np.flipud(Y)

    f0 = interpolate.interp2d(X, Y, data[:,:,0], kind=kind)
    f1 = interpolate.interp2d(X, Y, data[:,:,1], kind=kind)
    f2 = interpolate.interp2d(X, Y, data[:,:,2], kind=kind)

    y_new = np.linspace(np.min(Y), np.max(Y), ny)
    x_new = np.linspace(np.min(X), np.max(X), nx)

    data_new = np.empty([len(y_new), len(x_new), 3])
    data_new[:,:,0] = f0(x_new, y_new)
    data_new[:,:,1] = f1(x_new, y_new)
    data_new[:,:,2] = f2(x_new, y_new)

    return data_new


def pmm_lalo_enu(pmm, Lats, Lons):
    """
    Input:
        pmm     plate motion model instance
        Lats    2D array of latitudes;                 dim = (length, width)
        Lons    2D array of longitudes;                dim = (length, width)

    Output:
        enu     3D array of {east, north, up} motions; dim = (length, width, 3)
    """
    try:
        n    = len(Lats)
        Lats = np.array(Lats)
        Lons = np.array(Lons)
    except:
        pass

    if isinstance(Lats, float) or isinstance(Lats, int):
        print('Single location')
        loc = np.array([Lats, Lons, 0])
        v   = pmm.velocity_at(loc,'geodetic')
        en  = np.array(v.en)
        enu = np.concatenate((en, [0]))

    elif len(Lats.shape) == 1:
        print('1D array locations')
        ele  = np.zeros_like(Lats)
        locs = np.dstack((Lats, Lons, ele))[0].T
        v    = pmm.velocity_at(locs,'geodetic')
        en   = np.array(v.en).reshape([-1,2])
        enu  = np.concatenate((en, np.zeros([en.shape[0],1])),1)

    elif len(Lats.shape) > 1:
        print('2D array locations')
        length, width = Lats.shape
        ele  = np.zeros_like(Lats)
        locs = np.dstack((Lats, Lons, ele))
        locs = locs.reshape([-1, 3]).T
        v    = pmm.velocity_at(locs,'geodetic')
        en   = np.array(v.en).reshape([-1,2])
        enu  = np.concatenate((en, np.zeros([en.shape[0],1])),1)
        enu  = enu.reshape([length, width, -1])

    else:
        print('Weird input')

    return enu


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
    return prob, y_pred


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


def dem_shading(dem, shade_azdeg=315, shade_altdeg=45, shade_exag=0.5, shade_min=-2e3, shade_max=3e3):
    # prepare shade relief
    import warnings
    from matplotlib.colors import LightSource
    from mintpy.objects.colors import ColormapExt

    ls = LightSource(azdeg=shade_azdeg, altdeg=shade_altdeg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dem_shade = ls.shade(dem, vert_exag=shade_exag, cmap=ColormapExt('gray').colormap, vmin=shade_min, vmax=shade_max)
    dem_shade[np.isnan(dem_shade[:, :, 0])] = np.nan
    return dem_shade


def scatter_fields(data1, data2, labels=['data1','data2'], vlim=[-20,20], title='', savedir=False):
    ## linear fit to the trend
    x = data1.flatten()[~np.isnan(data2.flatten())]
    y = data2.flatten()[~np.isnan(data2.flatten())]

    fit, y_pred = scipy_L2_reg(x, y)

    # plot
    plt.figure(figsize=[6,6])
    plt.scatter(x, y)
    plt.scatter(x, y_pred, s=0.3, label='y=ax+b \n a={:.3f}±{:.3f}, b={:.3f}'.format(fit.slope, fit.stderr, fit.intercept))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(vlim[0], vlim[1])
    plt.legend(loc='upper left')
    plt.title(title)
    if savedir is not False:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # output
        out_file = f'{savedir}/{title}.png'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=200)
        print('save to file: '+out_file)
        plt.close()
    else:
        plt.show()
    return fit



def plot_range_ramp(data1, data2=None, range_dist=None, type=None, latitude=None, titstr='', vlim=[None, None], plot=True, picdir='./pic', outf=True):
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
        xlabel     = '{} range distance [km]'.format(type.title())
        slope_unit = 'mm/yr/km'
        factor     = 1.0
        if latitude is None:
            yarry  = np.array(abins)
            ylabel = 'Azimuth bin'
        else:
            yarry  = np.array(latitude)
            ylabel = 'Latitude [deg]'

    # Plot single scatter plot
    if ncols == 1:
        if plot:
            fig, ax = plt.subplots(figsize=[8*ncols,8], ncols=ncols, sharey=True)
        x = xarry.flatten()[~np.isnan(data1.flatten())]
        y = data1.flatten()[~np.isnan(data1.flatten())]
        c = yarry.flatten()[~np.isnan(data1.flatten())]
        y = y - np.nanmedian(y)
        print('Range ramp scatter plot shifted by median {}'.format(np.nanmedian(y)))

        # linear fit to the trend
        fit, y_pred  = scipy_L2_reg(x, y)
        params_legend = 'y=ax+b, a={:.3f} ± {:.2e}\n'.format(fit.slope, fit.stderr)

        if range_dist is None:
            params_legend += 'slope = {:.3f} {:s}'.format(fit.slope*factor, slope_unit)
        else:
            range_span = np.max(x) - np.min(x)
            print('{} range distance spans {:.1f} km'.format(type.title(), range_span))
            params_legend += 'slope = {:.3f} {:s}'.format(fit.slope*factor, slope_unit)
            params_legend += '\n ({:.3f} mm/yr/track)'.format(fit.slope * range_span)

        if plot:
            sc = ax.scatter(x, y, s=0.1, c=c)
            ax.plot(x, y_pred, lw=2, label=params_legend, c='r')
            ax.legend(loc='upper left')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('LOS velocity [mm/yr]')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.ax.set_ylabel(ylabel='Azimuth', rotation=270, labelpad=20)
            ax.set_title(titstr)
            ax.set_ylim(vlim[0], vlim[1])
            plt.show()

    # Plot scatter plots comparison
    if ncols == 3:
        if plot:
            fig, ax = plt.subplots(figsize=[8*ncols,8], ncols=ncols, sharey=True)
        for i, data in enumerate([data1, data2, data1-data2]):
            x = xarry.flatten()[~np.isnan(data.flatten())]
            y =  data.flatten()[~np.isnan(data.flatten())]
            c = yarry.flatten()[~np.isnan(data.flatten())]
            y = y - np.nanmedian(y)
            print('Range ramp scatter plot shifted by median {}'.format(np.nanmedian(y)))

            # linear fit to the trend
            fit, y_pred  = scipy_L2_reg(x, y)
            params_legend = 'y=ax+b, a={:.3f} ± {:.2e}\n'.format(fit.slope, fit.stderr)

            if range_dist is None:
                params_legend += 'slope = {:.3f} {:s}'.format(fit.slope*factor, slope_unit)
            else:
                range_span = np.max(x) - np.min(x)
                print('{} range distance spans {:.1f} km'.format(type.title(), range_span))
                params_legend += 'slope = {:.3f} {:s}'.format(fit.slope*factor, slope_unit)
                params_legend += '\n ({:.3f} mm/yr/track)'.format(fit.slope * range_span)

            if plot:
                sc = ax[i].scatter(x, y, s=0.1, c=c)
                ax[i].plot(x, y_pred, lw=2, label=params_legend, c='r')
                ax[i].legend(loc='upper left')
                ax[i].set_xlabel(xlabel)
                ax[i].set_ylabel('LOS velocity [mm/yr]')
                cbar = plt.colorbar(sc, ax=ax[i])
                cbar.ax.set_ylabel(ylabel=ylabel, rotation=270, labelpad=20)
                ax[i].set_title(titstr[i])
                ax[i].set_ylim(vlim[0], vlim[1])
        # output
        if outf:
            if not os.path.exists(picdir):
                os.makedirs(picdir)
            out_file = f'{picdir}/{outf}.png'
            plt.savefig(out_file, bbox_inches='tight', dpi=200)
            print('save to file: '+out_file)

        if plot:
            plt.show()

    return fit.slope*factor


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
    rg    = Re * gamma   # ground range
    rg -= np.nanmin(rg)

    return rg



def prepare_los_geometry(geom_file):
    """Prepare LOS geometry data/info in geo-coordinates
    Parameters: geom_file  - str, path of geometry file
    Returns:    inc_rad    - 2D np.ndarray, incidence angle in radians
                head_rad   - 2D np.ndarray, heading   angle in radians
                atr        - dict, metadata in geo-coordinate
    """

    print('prepare LOS geometry in geo-coordinates from file: {}'.format(geom_file))
    atr = readfile.read_attribute(geom_file)

    inc_deg = readfile.read(geom_file, datasetName='incidenceAngle')[0]

    if 'latitude' in readfile.get_dataset_list(geom_file):
        lat_deg = readfile.read(geom_file, datasetName='latitude')[0]
    else:
        lat_deg = get_geo_lat_lon(atr)[0]

    if 'longitude' in readfile.get_dataset_list(geom_file):
        lon_deg = readfile.read(geom_file, datasetName='longitude')[0]
    else:
        lon_deg = get_geo_lat_lon(atr)[1]

    if 'azimuthAngle' in readfile.get_dataset_list(geom_file):
        print('convert azimuth angle to heading angle')
        azi_deg  = readfile.read(geom_file, datasetName='azimuthAngle')[0]
        head_deg = ut.azimuth2heading_angle(azi_deg)
    else:
        print('use the HEADING attribute as the mean heading angle')
        head_deg = np.ones(inc_deg.shape, dtype=np.float32) * float(atr['HEADING'])

    if 'slantRangeDistance' in readfile.get_dataset_list(geom_file):
        print('convert slantRangeDistance from meter to kilometer')
        slantrange  = readfile.read(geom_file, datasetName='slantRangeDistance')[0]
        elevation   = readfile.read(geom_file, datasetName='height')[0]
        groundrange = compute_ground_range(atr, slantrange, elevation)

        slantrange  /= 1e3
        groundrange /= 1e3

    # turn default null value to nan
    inc_deg[inc_deg==0]    = np.nan
    head_deg[head_deg==90] = np.nan
    # unit: degree to radian
    inc_rad  = np.deg2rad(inc_deg)
    head_rad = np.deg2rad(head_deg)
    return lat_deg, lon_deg, inc_rad, head_rad, slantrange, groundrange, atr


def los_unit_vector(inc_rad, head_rad, in_unit=None):
    """
    Make a plot trigonometry of inc angle and head angle (in radians)

    inc_rad      inc_angle in radians
    head_rad     head_angle in radians
    in_unit      given input unit vector, for direct plotting

    """
    if in_unit is not None:
        unitv = np.array(in_unit)
    else:
        unitv = np.array([-1*np.sin(inc_rad)*np.cos(head_rad),
                             np.sin(inc_rad)*np.sin(head_rad),
                             np.cos(inc_rad)                  ])

    tstrs = ['for E\n-sin(inc)*cos(head)', 'for N\nsin(inc)*sin(head)', 'for U\ncos(inc)']

    if in_unit is not None:
        fig, axs = plt.subplots(nrows=1, ncols=len(unitv), figsize=[12,10], sharey=True, gridspec_kw={'wspace':0.14})
        for i, (u, tstr) in enumerate(zip(unitv, tstrs)):
            im = axs[i].imshow(u)
            fig.colorbar(im, ax=axs[i], fraction=0.05)
            axs[i].set_title(tstr)
        plt.show()
    return unitv


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
        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=[24,10], sharey=True, gridspec_kw={'wspace':0.14})
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


def simple_orbit_geom(orbit, length, width, given_inc='table', given_head='table'):
    # Default table from Aqaba
    # Caveat: heading angle varies a lot! e.g. Australia heading angle = -166.47261 to -164.69743 deg
    table = dict()
    table['ASCENDING'] = {}
    table['ASCENDING']['inc'] = (30.68, 46.24)     # (degrees vary from near_range to far_range)
    table['ASCENDING']['azi'] = (-258.93, -260.52)
    table['DESCENDING'] = {}
    table['DESCENDING']['inc'] = (30.78, 46.28)    # (degrees vary from near_range to far_range)
    table['DESCENDING']['azi'] = (-101.06, -99.43)

    if given_inc == 'table':
        inc_vary = table[orbit]['inc']
    else:
        inc_vary = given_inc

    if given_head == 'table':
        azi_vary = table[orbit]['azi']
    else:
        azi_vary  = (ut.heading2azimuth_angle(given_head[0]), ut.heading2azimuth_angle(given_head[1]))

    head_vary = (ut.azimuth2heading_angle(azi_vary[0]), ut.azimuth2heading_angle(azi_vary[1]))

    print('Incidence angle range: [{:.2f}, {:.2f}] deg'.format(*inc_vary))
    print('Heading   angle range: [{:.2f}, {:.2f}] deg'.format(*head_vary))

    inc_deg  = np.tile(np.linspace(*inc_vary, width), (length,1))
    azi_deg  = np.tile(np.linspace(*azi_vary, width), (length,1))
    head_deg = ut.azimuth2heading_angle(azi_deg)   #  literally: 90.0 - azi_deg
    inc_rad  = np.deg2rad(inc_deg)
    azi_rad  = np.deg2rad(azi_deg)
    head_rad = np.deg2rad(head_deg)
    return inc_rad, head_rad, azi_rad


def create_v(v_hor=1, theta=90., v_ver=0.):
    """
    Input:
        v_hor       horizontal absolute motion
        theta       angle (deg) of motion vector, clockwise from North is positive (default=90.; purely eastward)
                    this is same definition as satellite heading in Mintpy: (https://mintpy.readthedocs.io/en/latest/api/attributes/#optional_attributes)
        v_ver       vertical absolute motion (default=0.)
    Output:
        v           [ve, vn, vu]
    """
    v = np.array([  v_hor*np.sin(np.deg2rad(theta)),
                    v_hor*np.cos(np.deg2rad(theta)),
                    v_ver])
    return v



############################

class LOSgeom():
    """
    Classs to compute the bulk motion on a given geometry
    """

    def __init__(self, geom_file=None):
        ## geom_file: path to geometryGeo.h5 or geometryRadar.h5
        if geom_file is not None:
            # Strongly recommended !!
            # prepare LOS geometry: need to be in radar coord
            self.lat, self.lon, self.inc_rad, self.head_rad, self.slantrange, self.groundrange, self.atr_geo = prepare_los_geometry(geom_file)
            self.width    = int(self.atr_geo['WIDTH'])
            self.length   = int(self.atr_geo['LENGTH'])
            self.orbit    = self.atr_geo['ORBIT_DIRECTION']
            self.head_deg = np.rad2deg(self.head_rad)
            self.azi_deg  = ut.heading2azimuth_angle(self.head_deg)
            self.azi_rad  = np.deg2rad(self.azi_deg)
            self.inc_deg  = np.rad2deg(self.inc_rad)
        else:
            print('No geometry file (.h5) specified!')
            print('Please run guess_geom(orbit, given_inc, given_head, length, width) with default/custom angles and orbit')
            print('Note that this method is only a ballpard estimate, not recommended')
        return


    def guess_geom(self, orbit='ASCENDING', given_inc='table', given_head='table', length=500, width=300):
        ## A simple guess. Not recommend
        # orbit           "ASCENDING" or "DESCENDING"
        # given_inc       'table' or a custom tuple (inc0, inc1)  ; assume inc angle from simple guess
        # given_head      'table' or a custom tuple (head0, head1); assume head angle from simple guess
        self.orbit      = orbit
        self.given_inc  = given_inc
        self.given_head = given_head
        self.length     = length
        self.width      = width
        print('Initialize a simple "{}" geometry with dimension ({}, {})'.format(self.orbit, self.length, self.width))
        print('Given incidence: {}'.format(self.given_inc))
        print('Given heading: {}'.format(self.given_head))

        angles = simple_orbit_geom(self.orbit, self.length, self.width, self.given_inc, self.given_head)
        self.inc_rad  = angles[0]
        self.head_rad = angles[1]
        self.azi_rad  = angles[2]
        self.inc_deg  = np.rad2deg(self.inc_rad)
        self.head_deg = np.rad2deg(self.head_rad)
        self.azi_deg  = np.rad2deg(self.azi_rad)


    def enu2los(self, V, ref=False):
        ## V               [ve, vn, vu]; dim=(N,3) or (3,); floats; three component motions
        ## ref             reference the vlos to the middle pixel, or a custom pixel
        #print('Dynamic range of head_angle:', np.nanmax(self.head_deg), np.nanmin(self.head_deg))

        V = np.array(V)

        if len(V.shape) == 1:
            #print('Using a single-vector ENU velocity input')
            self.V_pmm_enu = np.array(V)
            self.ve        = float(self.V_pmm_enu[0])
            self.vn        = float(self.V_pmm_enu[1])
            self.vu        = float(self.V_pmm_enu[2])
            self.V_pmm     = ut.enu2los(self.ve, self.vn, self.vu, inc_angle=self.inc_deg, head_angle=self.head_deg)

            self.ref = ref
            if ref:
                if ref is True:
                    idx = self.length//2, self.width//2
                else:
                    idx = ref[0], ref[1]
                V_ref = self.V_pmm[idx[0], idx[1]]
                if np.isnan(V_ref):
                    print('Reference point is NaN, choose another point!')
                    sys.exit(1)
                self.V_pmm -= V_ref
                self.ref = idx


        elif len(V.shape) == 3:
            print('Using pixel-wise ENU velocity input')
            self.V_pmm_enu = V.reshape([self.length, self.width, -1])
            self.ve        = np.array((self.V_pmm_enu[:,:,0]))
            self.vn        = np.array((self.V_pmm_enu[:,:,1]))
            self.vu        = np.array((self.V_pmm_enu[:,:,2]))
            self.V_pmm     = ut.enu2los(self.ve, self.vn, self.vu, inc_angle=self.inc_deg, head_angle=self.head_deg)

            self.ref = ref
            if ref:
                if ref is True:
                    idx = self.length//2, self.width//2
                else:
                    idx = ref[0], ref[1]
                V_ref = self.V_pmm[idx[0], idx[1]]
                if np.isnan(V_ref):
                    print('Reference point is NaN, choose another point!')
                    sys.exit(1)
                self.V_pmm -= V_ref
                self.ref = idx

        else:
            print('Input V should be a list or an array. Either (3,) or (N,3) or (N,M,3)')
            sys.exit(1)



    def plot_inputs(self, ref=False, display=True, cmap='RdYlBu_r'):
        ## Plot {e, n, u, inc, head, vlos}
        if ref:
            self.ref = ref
        plot_enulos(self.V_pmm_enu, self.inc_deg, self.head_deg, self.ref, display=display, cmap=cmap)



    def plot_ramp(self, track=False, indata=None, compare=False, vlim=[None]*2, distance='ground', sn=None, picdir='./pic', outf=False):
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
            vstr = '(pixel-wise ENU)'

        ## How many subplots you need
        if indata is None:
            # show model
            titstr = '{} Plate bulk motion LOS projection\n{}'.format(self.name, vstr)
            self.rangeslope = plot_range_ramp(data1=V_los, range_dist=range_dist, type=distance, latitude=lat, titstr=titstr, vlim=vlim, picdir=picdir, outf=outf)
        elif (indata is not None) and (compare is True):
            # show data and compare with model
            titstr = ['{} Data LOS velocity'.format(self.name),
                      '{} Plate bulk motion LOS projection\n{}'.format(self.name, vstr),
                      '{} Corrected LOS velocity'.format(self.name)]
            self.rangeslope = plot_range_ramp(data1=data, data2=V_los, range_dist=range_dist, type=distance, latitude=lat, titstr=titstr, vlim=vlim, picdir=picdir, outf=outf)
            self.V_data = np.array(indata)
        else:
            # show data
            titstr = '{} Data LOS velocity'.format(self.name)
            self.rangeslope = plot_range_ramp(data1=data, range_dist=range_dist, type=distance, latitude=lat, titstr=titstr, vlim=vlim, picdir=picdir, outf=outf)
            self.V_data = np.array(indata)
        return self.rangeslope



    def plot_demshade(self, dem_shade, coord='geo', picdir='./pic', outf='los_test'):
        ## Plot with DEM overlaid (need to have dem_shade input)
        if dem_shade and self.atr_geo:
            v_show = {ikey: vars(self)[ikey] for ikey in ['inc_deg', 'head_deg', 've', 'vn', 'vu' 'V_pmm']}
            sarplt.plot_imgs(v=v_show, meta=self.atr_geo, dem=dem_shade, coord=coord, picdir=picdir, outf=outf)



############################
