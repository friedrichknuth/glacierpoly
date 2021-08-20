import matplotlib.pyplot as plt
import cartopy
import rasterio
from rasterio.plot import show
import geopandas as gpd
import glob
import os


def plot_tif(
    tif_file,
    glacier_outline_gdf=None,
    cmap=None,
    vmin=None,
    vmax=None,
    cbar_fraction=0.035,
):

    source = rasterio.open(tif_file)

    fig, ax = plt.subplots(figsize=(10, 10))

    sm = ax.imshow(source.read(1), cmap=cmap, vmin=vmin, vmax=vmax)

    crs = cartopy.crs.epsg(source.crs.to_epsg())

    ax = plt.axes(projection=crs)

    show(source, cmap=cmap, ax=ax, interpolation="none", vmin=vmin, vmax=vmax)

    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    gl.xlabel_style = {
        "size": 12,
        "color": "black",
        "rotation": 0,
        "rotation_mode": "anchor",
    }
    gl.ylabel_style = {
        "size": 12,
        "color": "black",
        "rotation": 0,
        "rotation_mode": "anchor",
    }
    gl.right_labels = False
    gl.top_labels = False

    fig.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=0.001, extend="both", aspect=50)

    glacier_outline_gdf.plot(ax=ax, facecolor="none", edgecolor="b")
