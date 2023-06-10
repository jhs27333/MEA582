## import python modules
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
import pandas as pd
import datetime as dt
import os
import pyproj
import xesmf as xe
import matplotlib.cm as cm
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap


####### colormap for precipitation ########
nws_precip_colors = [
#     "#04e9e7",  # 0.01 - 0.10 inches
#     "#019ff4",  # 0.10 - 0.25 inches
#     "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
#     "#f800fd",  # 6.00 - 8.00 inches
#     "#9854c6",  # 8.00 - 10.00 inches
#     "#fdfdfd"   # 10.00+
]
precip_colormap = ListedColormap(nws_precip_colors)
levels_prc=np.arange(0.0,5.0,0.5)
prc_norm = BoundaryNorm(levels_prc, precip_colormap.N)
#######################################

class FeatureEnv:
    def __init__(self, year, tracks_file, env_info, date_str_info, var_transform, dim_vars,
                 regridding, offset, save_path):
        self.env_info =  env_info
        self.date_str_info=date_str_info
        self.var_transform=var_transform
        self.tracks_file = tracks_file
        self.year = year
        self.offset = offset
        self.save_path=save_path
        self.dim_vars=dim_vars
        self.regridding=regridding
        self.feature_tracks = {}
        self.ds_comp = {}

        ## set lat lon ##
        self.lat=None
        self.lon=None

    @staticmethod
    def __extract_dates(date_array):
        """
        converts date array to date string
        :param date_array:
        :return:
        """
        dates = []
        for n in range(date_array.shape[0]):
            yr, mo, dy, hr = np.int_(date_array[n])
            dates.append(dt.datetime(yr, mo, dy, hr))

        return dates

    def extract_te_tracks(self):
        """
        Reads features from TE tracks file
        that assumes info stored in tab-separated columns
        :return: track information in a dictionary
        """

        ### read tracks file
        f = open(self.tracks_file, 'r')
        lines = f.readlines()  # Read file and close
        f.close()

        i = 0
        ctr = 0
        while i < len(lines) - 1:
            line = lines[i]
            i = i + 1
            line_split = line.strip().split('\t')
            if line_split[0] == 'start':
                track_length = int(line_split[1])
                track_array = np.genfromtxt(lines[i:(i + track_length)])
                dates = self.__extract_dates(np.int_(track_array[:, -4:]))
                lon = track_array[:, 2]
                lat = track_array[:, 3]
                psl = track_array[:, 4] * 1e-2  ### in hPa
                max_wind = track_array[:, 5]

                ### this is TE-specific code !!

                if dates[0].year == self.year & dates[-1].year == self.year:
                    self.feature_tracks[ctr] = {}
                    self.feature_tracks[ctr]['lat'] = lat
                    self.feature_tracks[ctr]['lon'] = lon
                    self.feature_tracks[ctr]['dates'] = dates
                    self.feature_tracks[ctr]['psl'] = psl
                    self.feature_tracks[ctr]['max_wind'] = max_wind
                    ctr += 1

        print('There are {:d} features in {:d}'.format(ctr + 1, self.year))


    def plot_tracks(self,key,ax):

        lat_array = self.feature_tracks[key]['lat']
        lon_array = self.feature_tracks[key]['lon']
        sizes = self.feature_tracks[key]['psl'].copy()
        sizes /= sizes.max()
        sizes *= 50
        colors = cm.Reds(np.linspace(0, 1, sizes.size))

        extent = [lon_array.min() - self.offset, lon_array.max() + self.offset,
                  lat_array.min() - self.offset, lat_array.max() + self.offset]

        ax.scatter(lon_array, lat_array, s=sizes, c=colors,
                   alpha=0.5, transform=ccrs.PlateCarree())  # Plot

        ax.scatter(lon_array[0], lat_array[0], s=sizes.max() * 2.0,
                   c='black', marker='^', alpha=1.0,
                   transform=ccrs.PlateCarree())  # Plot

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_aspect('auto')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = False
        gl.top_labels = False

        gl.xlines = True
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

    def get_lat_lon_info(self):

        env_keys = list(self.env_info.keys())
        fil_path= self.env_info[env_keys[0]]
        fil_env = glob.glob(fil_path+'{}*'.format(self.year))[0]
        ds_env = xr.open_dataset(fil_env)
        self.lat=ds_env[self.dim_vars['lat']]
        self.lon=ds_env[self.dim_vars['lon']]
        ds_env.close()


    def __create_empty_var(self,feature_dates):

        data_vars = dict()
        coords=dict()
        nan_array = np.full((len(feature_dates), self.lat.size, self.lon.size), np.nan)
        nan_array_1d = np.full(len(feature_dates), np.nan)

        ### only using 2D vars here, can be expanded to include 3D vars
        dims_2d = [self.dim_vars['time'], self.dim_vars['lat'], self.dim_vars['lon']]
        dims_1d = [self.dim_vars['time']]

        var_keys = list(self.env_info.keys())
        for vk in var_keys:
            for k1 in vk:
                data_vars[k1] = (dims_2d, nan_array.copy())


        data_vars['lat0']=(dims_1d, nan_array_1d.copy())
        data_vars['lon0']=(dims_1d, nan_array_1d.copy())

        ### create coords dict ###
        coords[self.dim_vars['time']]=feature_dates
        coords[self.dim_vars['lat']]=self.lat
        coords[self.dim_vars['lon']]=self.lon

        ### create empty array ###
        return xr.Dataset(data_vars=data_vars, coords=coords)


    @staticmethod
    def __regrid_data(loc_dict_in,loc_dict_out):
        ds_in=xr.Dataset(loc_dict_in)
        ds_out=xr.Dataset(loc_dict_out)
        regridder=xe.Regridder(ds_in, ds_out, "bilinear")
        return regridder

    def extract_env_save(self,key):

        """
        This is the function that reads the variables
        fields from the env. paths and saves to disk.
        """

        ## Use feature number to extract environment ###
        dates = self.feature_tracks[key]['dates']
        ## create empty data array to store env. info.
        self.ds_comp[key]=self.__create_empty_var(dates)

        self.ds_comp[key]['lat0'].loc[dict(time=dates)] = self.feature_tracks[key]['lat']
        self.ds_comp[key]['lon0'].loc[dict(time=dates)] = self.feature_tracks[key]['lon']

        self.ds_comp[key]['lat0'].attrs = ({'description': 'Latitude center of tracked feature'})
        self.ds_comp[key]['lon0'].attrs = ({'description': 'Longitude center of tracked feature'})

        for i,j,date in zip(self.feature_tracks[key]['lat'],
                              self.feature_tracks[key]['lon'],
                              self.feature_tracks[key]['dates']):

            la_out=self.lat
            lo_out=self.lon

            loc_dict_out = dict(time=date,
                                 lat=la_out[(la_out >= i - self.offset) & (la_out <= i + self.offset)],
                                 lon=lo_out[(lo_out >= j - self.offset) & (lo_out <= j + self.offset)])

            ### open env. file paths ###
            ds_env={}
            for key_name in list(self.env_info.keys()):
                date_str = dt.datetime.strftime(date, self.date_str_info[key_name])
                fils_env=glob.glob(self.env_info[key_name] + date_str + '*')[0]

                ds_env[key_name]=xr.open_mfdataset(fils_env)
                ## perform var. transformations like coordinate renaming (if specified)
                ds_env[key_name]=self.var_transform[key_name](ds_env[key_name])

                la_in = ds_env[key_name].lat
                lo_in = ds_env[key_name].lon

                loc_dict_in = dict(lat=la_in[(la_in >= i - self.offset) & (la_in <= i + self.offset)],
                                   lon=lo_in[(lo_in >= j - self.offset) & (lo_in <= j + self.offset)])

                for var in key_name:
                    var_slice=ds_env[key_name][var].sel(time=date, method='nearest').loc[loc_dict_in]
                    if self.regridding[key_name]:
                        regridder=self.__regrid_data(loc_dict_in,loc_dict_out)
                        var_slice=regridder(var_slice)

                    self.ds_comp[key][var].loc[loc_dict_out]=var_slice

                ds_env[key_name].close()

        ## drop NaNs
        self.ds_comp[key]=self.ds_comp[key].dropna(dim='lat', how='all').dropna(dim='lon', how='all')
        if self.save_path:
            fil_name=self.save_path+'{:02d}_{}.nc'.format(key,self.year)
            if os.path.isfile(fil_name):
                os.remove(fil_name)
            self.ds_comp[key].to_netcdf(fil_name)
            print('File saved as {}'.format(fil_name))

        self.ds_comp[key].close()


class ProcessFeatureEnv:
    def __init__(self,path,key,year):
        self.ds=xr.open_dataset(path+"{}_{}.nc".format(key,year))
        self.ds.close()
        self.ds_composite=None
        self.precip_binned_1d=None
        self.precip_binned_2d=None
        self.precip_stderr_1d=None
        self.precip_hist_1d=None
        self.precip_hist_2d=None


    def compute_secondary_vars(self,compute_dict):
        for k in compute_dict.keys():
            self.ds = self.ds.assign({k : compute_dict[k]})

    @staticmethod
    def __convert_latlon_to_stereo(lat0,lon0,xmesh,ymesh):

        source_crs = 'epsg:4326'
        target_crs = '+proj=stere +lat_0={} +lon_0={}'.format(lat0, lon0)
        stereo_to_latlon=pyproj.Transformer.from_crs(target_crs,source_crs)
        latmesh,lonmesh=stereo_to_latlon.transform(xmesh, ymesh)

        lat_targ = xr.DataArray(latmesh, dims=('y','x'))
        lon_targ = xr.DataArray(lonmesh, dims=('y','x'))

        return lat_targ,lon_targ

    @staticmethod
    def __dropna(ds, var_list):
        vars_out = {}
        for var in var_list:
            vars_out[var] = ds[var].dropna('lat', how='all').dropna('lon', how='all')
        return vars_out

    def compute_composites(self,var_list,xdim,ydim):

        xmesh, ymesh = np.meshgrid(xdim, ydim)

        ds_list = []
        for n, ti in enumerate(self.ds.time):
            ds0 = self.ds.isel(time=n)
            lat0 = ds0.lat0.values
            lon0 = ds0.lon0.values
            vars_out = self.__dropna(ds0, var_list)
            ds0.close()

            ds_new = xr.Dataset(vars_out)
            lat_targ, lon_targ = self.__convert_latlon_to_stereo(lat0, lon0,xmesh,ymesh)
            ds_new = ds_new.interp({'lon': lon_targ, 'lat': lat_targ}, method='nearest')
            ds_new = ds_new.assign_coords({'x': (['x'], xdim),
                                           'y': (['y'], ydim)})

            ds_new.close()
            ds_list.append(ds_new)

        self.ds_composite = xr.concat(ds_list[:], dim='time')

    @staticmethod
    def __bin_1d(x,y,xbins):
        y = y[np.isfinite(x)]
        x = x[np.isfinite(x)]

        xindx = np.int_((x - xbins[0]) / np.diff(xbins)[0])
        ybinned = np.zeros_like(xbins)
        xhist = np.zeros_like(xbins)
        ystderr = np.zeros_like(xbins)

        for i in np.arange(xbins.size):
            indx = np.where(xindx == i)
            ybinned[i] = y[indx].mean()
            xhist[i] = indx[0].size
            ystderr[i] = y[indx].std() / np.sqrt(xhist[i])

        xhist = xhist / (xhist.sum() * np.diff(xbins)[0])

        return ybinned, ystderr, xhist

    @staticmethod
    def __bin_2d(x, y, z, xbins, ybins):

        z = z[np.isfinite(x)]
        y = y[np.isfinite(x)]
        x = x[np.isfinite(x)]

        dx = np.diff(xbins)[0]
        dy = np.diff(ybins)[0]
        xindx = np.int_((x - xbins[0]) / dx)
        yindx = np.int_((y - ybins[0]) / dy)

        zbinned = np.zeros((xbins.size, ybins.size))
        zhist = np.zeros((xbins.size, ybins.size))

        for i in np.arange(xbins.size):
            for j in np.arange(ybins.size):
                indx = np.where(np.logical_and(xindx == i, yindx == j))
                zbinned[i, j] = z[indx].mean()
                zhist[i, j] = indx[0].size

        zhist = zhist / (zhist.sum() * dx * dy)

        return zbinned, zhist

    def compute_prc_buoy_stats(self,buoy_bins,instab_bins,subsat_bins):

        x1 = self.ds.buoy.stack(z=['time', 'lat', 'lon']).dropna('z')
        y1 = self.ds.precip_trmm.stack(z=['time', 'lat', 'lon']).dropna('z')
        self.precip_binned_1d, self.precip_stderr_1d, self.precip_hist_1d  = self.__bin_1d(x1, y1, buoy_bins)


        x2 = self.ds.instab.stack(z=['time', 'lat', 'lon']).dropna('z')
        y2 = self.ds.subsat.stack(z=['time', 'lat', 'lon']).dropna('z')
        z2 = self.ds.precip_trmm.stack(z=['time', 'lat', 'lon']).dropna('z')

        self.precip_binned_2d, self.precip_hist_2d = self.__bin_2d(x2, y2, z2, instab_bins, subsat_bins)


def time_mean_plot(ax, var, range, cmap, cbar_kwargs, title, lat0, lon0, norm=None):

        extent=[lon0.min()-5,lon0.max()+10,
                lat0.min()-10,lat0.max()]

        ax.coastlines()
        cb=var.plot.pcolormesh(ax=ax, vmax=range[0], vmin=range[1], cmap=cmap,  cbar_kwargs=cbar_kwargs, norm=norm)
        ax.scatter(lon0,lat0, s = 25, c = 'black', alpha = 0.5)
        ax.scatter(lon0[0],lat0[0],s=25, marker='D', c='red', alpha=0.5)
        ax.set_aspect('auto')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = False
        gl.top_labels = False

        gl.xlines = True
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180,180,10))
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}
        ax.set_title(title,fontsize=12)
        ax.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=180))

        return cb


def composite_plot(ax, var, range, cmap, cbar_kwargs, title, norm=None):
    # extent = [lon0.min() - 5, lon0.max() + 10,
    #           lat0.min() - 10, lat0.max()]

    cb = var.plot.pcolormesh(ax=ax, vmax=range[0], vmin=range[1], cmap=cmap, cbar_kwargs=cbar_kwargs, norm=norm)
    ax.scatter(0, 0, s=25, c='black', alpha=0.5)
    ax.set_aspect('auto')
    ax.xaxis.set_major_formatter(lambda x, pos: x * 1e-3)
    ax.yaxis.set_major_formatter(lambda x, pos: x * 1e-3)

    ax.set_aspect('auto')

    ax.set_title(title, fontsize=12)
    # ax.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=180))

    return cb


def snapshot_plot(ax, var, range, cmap, cbar_kwargs, title, cbar_on=False, norm=None):

    if cbar_on:
        var.plot.pcolormesh(ax=ax, vmin=range[0], vmax=range[1], cbar_kwargs=cbar_kwargs, cmap=cmap, norm=norm)
    else:
        var.plot.pcolormesh(ax=ax, vmin=range[0], vmax=range[1], add_colorbar=cbar_on, cmap=cmap, norm=norm)


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False

    gl.xlines = True
    gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
    gl.xlocator = mticker.FixedLocator(np.arange(-180,180,10))
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    ax.set_title('{}'.format(title),fontsize=12)
    ax.set_aspect('auto')













