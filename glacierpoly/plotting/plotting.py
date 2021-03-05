import matplotlib.pyplot as plt
import cartopy
import rasterio
from rasterio.plot import show
import geopandas as gpd
import glob
import os


def plot_tif(tif_file, gdf, vlim = 10):
    source = rasterio.open(tif_file)

    fig, ax = plt.subplots(figsize=(10, 10))

    crs = cartopy.crs.epsg(source.crs.to_epsg())

    ax = plt.axes(projection=crs)

    show(source,
         cmap='RdBu',
         vmin=-vlim,
         vmax=vlim,
         ax=ax,
         interpolation='none')

    gl = ax.gridlines(draw_labels=True,x_inline=False, y_inline=False)

    gl.xlabel_style = {'size': 12, 'color': 'black', 'rotation': 0, 'rotation_mode': 'anchor'}
    gl.ylabel_style = {'size': 12, 'color': 'black', 'rotation': 0, 'rotation_mode': 'anchor'}
    gl.right_labels = False
    gl.top_labels = False

    sm = plt.cm.ScalarMappable(cmap='RdBu',norm=plt.Normalize(vmin=-vlim, vmax=vlim))

    fig.colorbar(sm, ax=ax,fraction=0.035, pad=0.001, extend='both', aspect=50)
    
    gdf.plot(ax=ax, facecolor='none', edgecolor='b')