import os
from pathlib import Path
import glob
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imshow
from skimage.morphology import (
    erosion,
    dilation,
    closing,
    opening,
    area_closing,
    area_opening,
)
from skimage.measure import label, regionprops, regionprops_table
import subprocess
from rasterstats import zonal_stats
from rasterio.plot import show
from rasterio import mask

import glacierpoly as gpoly


def clip_area_beyond_previous_glacier_boundary(
    reference_glacier_polygon_file,
    detected_polygon_gdf,
    keep_positive_change=False,
    difference_map_file=None,
    keep_positive_change_greater_than=0,
):
    """
    Removes change areas beyond previous glacier boundary.
    These are most likely sediment signals.
    Optionally keep positive change areas in case of advance beyond
    previous polygon extent.

    TODO
    - implement keeping positive change
    """
    reference_polygon = gpd.read_file(reference_glacier_polygon_file)

    diff = gpd.overlay(detected_polygon_gdf, reference_polygon, how="difference")

    if keep_positive_change:
        print("not implemented")
    #         if not difference_map_file:
    #             print('must specify difference map file to get positive change areas')

    #         diff = diff.explode().reset_index().iloc[: , 2:]
    #         diff_stats = zonal_stats(diff, difference_map_file)
    #         mean_dods = []
    #         for i in diff_stats:
    #             mean_dods.append(i['mean'])
    #         diff['mean_dod'] = mean_dods
    #         diff = diff[diff['mean_dod'] < keep_positive_change_greater_than]

    detected_polygon_gdf = gpd.overlay(detected_polygon_gdf, diff, how="difference")
    return detected_polygon_gdf


def clip_raster_by_buffer(
    difference_map_file, reference_glacier_polygon_file, buffer_distance=2000
):
    """
    Clips difference map bu buffer around reference polygon.
    Difference map must be in UTM.

    Returns:
    numpy array as uint8

    TODO
    - check if difference map in UTM
    """

    gdf = gpd.read_file(reference_glacier_polygon_file)

    buffer = gdf.buffer(buffer_distance)
    source = rasterio.open(difference_map_file, masked=True)
    rio_mask_kwargs = {"filled": False, "crop": True, "indexes": 1}
    masked_array, transform = rasterio.mask.mask(source, buffer)
    array = masked_array.squeeze()
    array = gpoly.core.replace_and_fill_nodata_value(array, source.nodata, 0)
    array = np.uint8(array)

    return array


def convert_glacier_array_to_gdf(array, transform, res, crs):
    """
    Writes array to geotiff then converts to geojson.
    Assumes array originated from geotiff with same shape,
    transform, resolution, and crs.
    """

    if os.path.exists("tmp.tif"):
        os.remove("tmp.tif")
    if os.path.exists("tmp.geojson"):
        os.remove("tmp.geojson")

    with rasterio.open(
        "tmp.tif",
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        nodata=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)

    call = ["gdal_polygonize.py", "tmp.tif", "tmp.geojson"]
    subprocess.call(call)

    os.remove("tmp.tif")
    call = [
        "gdal_rasterize",
        "-burn",
        "1",
        "-tr",
        str(res[0]),
        str(res[1]),
        "-a_nodata",
        "0",
        "-add",
        "tmp.geojson",
        "tmp.tif",
    ]
    subprocess.call(call)

    os.remove("tmp.geojson")
    call = ["gdal_polygonize.py", "tmp.tif", "tmp.geojson"]
    subprocess.call(call)

    detected_polygon = gpd.read_file("tmp.geojson", driver="GeoJSON")

    os.remove("tmp.tif")
    os.remove("tmp.geojson")

    return detected_polygon


def detect_glacier(
    array,
    erode_islands=True,
    erode_islands_search_window=(9, 9),
):
    """
    Finds glacier in uint8 array of difference map values.
    Set erode islands to False if glacier not detected.

    Returns
    numpy array representing glacier

    TODO
    - add qc plots as gallery for each step
    """
    if erode_islands:
        erode_islands_window = np.ones(erode_islands_search_window).astype(int)
        array = opening(array, erode_islands_window)

    # detect edges
    canny = cv2.Canny(array, 50, 150)

    # dilate edges
    dilate_edges_window = np.ones((3, 3)).astype(int)
    multi_dilated = gpoly.core.multi_dil(canny, 4, dilate_edges_window)

    # close areas
    area_closed = area_closing(multi_dilated, 1e6)

    # label areas
    label_im = label(area_closed)
    regions = regionprops(label_im)

    # get stats
    properties = [
        "area",
        "convex_area",
        "bbox_area",
        "extent",
        "mean_intensity",
        "solidity",
        "eccentricity",
        "orientation",
    ]
    df = pd.DataFrame(regionprops_table(label_im, array, properties=properties))

    # get largest region
    max_area_index = df[df["area"] == df["area"].max()].index[0]
    if max_area_index == 0:
        pass
    else:
        max_area_index = max_area_index + 1  # seems to be an indexing bug

    mask = label_im != max_area_index
    masked_array = np.ma.masked_array(label_im, mask=mask)
    array = np.ma.filled(masked_array, fill_value=1)
    array = array.astype(np.uint8)

    return array


def find_largest_intersecting_detected_polygon(
    reference_glacier_polygon_file, detected_polygon_gdf
):
    """
    Extracts largest polygon from multipolygon gdf
    """
    reference_polygon = gpd.read_file(reference_glacier_polygon_file)

    # check if intersects with reference polygon
    detected_polygon_gdf = detected_polygon_gdf[
        detected_polygon_gdf.intersects(reference_polygon.geometry[0])
    ]

    # get largest polygon from multipolygon
    detected_polygon_gdf = detected_polygon_gdf[
        detected_polygon_gdf.area == detected_polygon_gdf.area.max()
    ]

    # use exterior to get rid of islands
    detected_polygon_gdf["geometry"] = Polygon(
        detected_polygon_gdf["geometry"].iloc[0].exterior
    )

    return detected_polygon_gdf


def get_raster_metadata(geotif):
    """
    Returns raster transform, resolution, and crs
    """
    source = rasterio.open(geotif)
    transform = source.transform
    res = source.res
    crs = source.crs

    return transform, res, crs


def merge_with_undetected_high_elevation_areas(
    reference_dem_file,
    reference_glacier_polygon_file,
    detected_polygon_file,
):
    """
    Merges areas from reference glacier polygon in regions
    above detected glacier regions. These undetected regions
    are most likely a result of sparse or missing data in the difference
    map
    """

    reference_glacier_polygon_gdf = gpd.read_file(reference_glacier_polygon_file)
    detected_polygon_gdf = gpd.read_file(detected_polygon_file)

    # get max detected elevation
    max_caputed_elevation = zonal_stats(detected_polygon_file, reference_dem_file)[0][
        "max"
    ]

    # find areas in reference glacier polygon above max captured elevation
    res_union = gpd.overlay(
        reference_glacier_polygon_gdf, detected_polygon_gdf, how="difference"
    )
    res_union = res_union.explode()
    res_union = res_union.reset_index().iloc[:, 2:]
    res_union = res_union[["geometry"]]
    stats = zonal_stats(res_union, reference_dem_file)
    max_elevations = []
    for i in stats:
        max_elevations.append(i["max"])
    res_union["max_elevations"] = max_elevations

    # merge where elevations area higher
    remaining_area = res_union[res_union["max_elevations"] > max_caputed_elevation]
    merged = detected_polygon_gdf.geometry.append(remaining_area.geometry)
    merged = gpd.GeoDataFrame(geometry=merged).reset_index(drop=True)

    # dissolve islands
    merged = merged.dissolve()
    merged = merged.explode().reset_index().iloc[:, 2:]
    geoms = []
    for i in range(0, len(merged)):
        geoms.append(Polygon(merged["geometry"].iloc[i].exterior))
    merged["geometry"] = geoms
    merged = merged.dissolve()

    return merged


def multi_dil(im, num, window):
    for i in range(num):
        im = dilation(im, window)
    return im


def multi_ero(im, num, window):
    for i in range(num):
        im = erosion(im, window)
    return im


def replace_and_fill_nodata_value(array, nodata_value, fill_value):
    if np.isnan(nodata_value):
        masked_array = np.nan_to_num(array, nan=fill_value)
    else:
        mask = array == nodata_value
        masked_array = np.ma.masked_array(array, mask=mask)
        masked_array = np.ma.filled(masked_array, fill_value=fill_value)

    return masked_array


def save_polygon_gdf_to_geojson(
    detected_polygon_gdf, file_path, file_name, extention=".geojson"
):
    Path(file_path).mkdir(parents=True, exist_ok=True)
    file_name = str(Path(file_name).stem)
    out = os.path.join(file_path, file_name + extention)
    detected_polygon_gdf.to_file(out, driver="GeoJSON")
    if os.path.exists(out):
        return out
    else:
        print("failed to create geojson at", out)
