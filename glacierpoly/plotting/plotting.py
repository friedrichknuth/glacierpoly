import cartopy
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import rasterio
from rasterio.plot import show


def create_detection_qc_gallery(
    arrays, detected_polygon_gdf, output_directory, difference_map_file, merged=None
):

    file_name = str(Path(difference_map_file).stem)
    # create qc plots
    fig = plt.figure(figsize=(10, 7))
    qc_plot_file_name = "qc_plot_" + file_name.split("_")[-1]
    fig.suptitle(qc_plot_file_name)
    rows = 2
    columns = 3
    for i in range(rows * columns):
        try:
            arr = arrays[i]
            ax = plt.subplot(rows, columns, i + 1)
            ax.imshow(arr)
            ax.set_xticks(())
            ax.set_yticks(())
        except:
            try:
                ax = plt.subplot(rows, columns, i + 1)
                try:
                    merged.plot(ax=ax)
                except:
                    pass
                detected_polygon_gdf.plot(ax=ax, color="r")
                ax.set_xticks(())
                ax.set_yticks(())
            except:
                pass
    fig.savefig(os.path.join(output_directory, qc_plot_file_name + ".jpg"))
    plt.close()


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
