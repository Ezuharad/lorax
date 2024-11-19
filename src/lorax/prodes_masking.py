import rasterio
import geopandas as gpd
from rasterio.mask import mask
import sys

filename = sys.argv[1]

vector_path = 'data/raster_features.geojson'  
raster_path = f'data/{filename}.tif'     
output_raster_path = f'forested/{filename}.tif'  

vector_mask = gpd.read_file(vector_path)

with rasterio.open(raster_path) as src:    
    if vector_mask.crs != src.crs:
        vector_mask = vector_mask.to_crs(src.crs)

    vector_mask_geom = [feature["geometry"] for feature in vector_mask.__geo_interface__["features"]]
    out_image, out_transform = mask(src, vector_mask_geom, crop=False, nodata=0, invert=True)
    # For deforested raster
    # out_image, out_transform = mask(src, vector_mask_geom, crop=True, nodata=0)
    
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": 0  
    })

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

