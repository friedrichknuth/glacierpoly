import cartopy
import copy
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
from pathlib import Path
import rasterio
from rasterio.plot import show
import numpy as np
import glacierpoly as gpoly


def create_detection_qc_gallery(
    arrays, detected_polygon_gdf, output_directory, difference_map_file, merged=None
):

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    file_name = str(Path(difference_map_file).stem)

    labels = [
        "unit8 array",
        "canny edges",
        "dilated edges",
        "area closing",
        "largest area",
        "final polygon",
    ]

    # create qc plots
    fig = plt.figure(figsize=(10, 8))
    qc_plot_file_name = file_name.split("_")[-1] + "_00_gallery"
    fig.suptitle(file_name.split("_")[-1])
    rows = 2
    columns = 3
    for i in range(rows * columns):
        try:
            arr = arrays[i]
            ax = plt.subplot(rows, columns, i + 1)
            if labels[i] == "canny edges" or labels[i] == "dilated edges":
                ax.imshow(arr, vmin=0, vmax=1)
            else:
                ax.imshow(arr)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(labels[i])
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
                ax.set_title(labels[i])
                labels = ["fill from rgi", "detected"]
                patches = [
                    Patch(facecolor="C0", label=labels[0]),
                    Patch(facecolor="r", label=labels[1]),
                ]
                ax.legend(handles=patches)
            except:
                pass
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_directory, qc_plot_file_name + ".jpg"),
        bbox_inches="tight",
        pad_inches=0,
    )
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


def plot_tif_with_polygons(
    difference_map_file,
    reference_glacier_polygon_file,
    merged_polygon_file,
    output_directory,
    suffix=None,
    cmap_name=None,
    vmin=None,
    vmax=None,
    cbar_fraction=0.035,
):

    cmap = copy.copy(plt.cm.get_cmap(cmap_name))
    cmap.set_bad(color="black", alpha=1)

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    file_name = str(Path(difference_map_file).stem)
    qc_plot_file_name = file_name.split("_")[-1] + suffix

    source = rasterio.open(difference_map_file)

    if vmin == None and vmax == None:
        array = source.read(1)
        array = array.astype(float)
        array = gpoly.core.replace_and_fill_nodata_value(array, source.nodata, np.nan)
        vmin, vmax = np.nanpercentile(array, [5, 95])

    fig, ax = plt.subplots(figsize=(10, 10))

    sm = ax.imshow(source.read(1), cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)

    crs = cartopy.crs.epsg(source.crs.to_epsg())

    ax = plt.axes(projection=crs)
    #     ax.set_facecolor('black')

    show(
        source, cmap=cmap, ax=ax, interpolation="none", vmin=vmin, vmax=vmax, alpha=0.5
    )

    cb = fig.colorbar(
        sm, ax=ax, fraction=cbar_fraction, pad=0.001, extend="both", aspect=50
    )
    cb.set_label(label="Elevation difference [m]", size=15)
    cb.ax.tick_params(labelsize=12)

    reference_glacier_polygon = gpd.read_file(reference_glacier_polygon_file)
    merged_polygon = gpd.read_file(merged_polygon_file)

    reference_glacier_polygon.plot(ax=ax, facecolor="none", edgecolor="b")
    merged_polygon.plot(ax=ax, facecolor="none", edgecolor="g")

    if reference_glacier_polygon.geometry.area[0] > merged_polygon.geometry.area[0]:
        tmp_poly = reference_glacier_polygon
    else:
        tmp_poly = merged_polygon
    bounds = tmp_poly.geometry.total_bounds
    xlim = [bounds[0] - 10, bounds[2] + 10]
    ylim = [bounds[1] - 10, bounds[3] + 10]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    rgi_area = str(np.around((reference_glacier_polygon.area[0] / 1e6), 2))
    detected_area = str(np.around((merged_polygon.area[0] / 1e6), 2))

    labels = [
        "RGI area " + rgi_area + " km2",
        "Detected area " + detected_area + " km2",
    ]
    patches = [
        Line2D([0], [0], color="b", label=labels[0]),
        Line2D([0], [0], color="g", label=labels[1]),
    ]
    ax.legend(handles=patches)
    ax.set_title(file_name.split("_")[-1])
    #         ax.fill_between([0,1],[1,1],hatch="X")

    fig.savefig(
        os.path.join(output_directory, qc_plot_file_name + ".jpg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
