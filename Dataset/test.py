import rasterio


with rasterio.open('goes_images_northeast/ABI-L1b-RadM/2017/198/12/OR_ABI-L1b-RadM1-M3C01_G16_s20171981200268_e20171981200325_c20171981200367.nc') as dataset:

    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()

    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):

        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            dataset.crs, 'EPSG:4326', geom, precision=6)

        # Print GeoJSON shapes to stdout.
        print(geom)
