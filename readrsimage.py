from osgeo import gdal
import numpy as np

def readrsimage(path):
    """
    Read remote sensing image without geo information.
    
    Return a numpy array.
    """

    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    channels = dataset.RasterCount
    dtype = np.float32

    image = np.zeros((height, width, channels), dtype=dtype)

    for band in range(channels):
        band_data = dataset.GetRasterBand(band + 1)
        image[:, :, band] = band_data.ReadAsArray()

    return image

def readrsimage_with_geoinfo(path):
    """
    introduction
    -----------------------------
    read remote sensing image with geoinfomation.

    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    channels = dataset.RasterCount
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    dtype = np.float32

    image = np.zeros((height, width, channels), dtype=dtype)

    for band in range(channels):
        band_data = dataset.GetRasterBand(band + 1)
        image[:, :, band] = band_data.ReadAsArray()

    return image, projection, geotransform

def writersimage(save_path, image, format = 'GTiff'):
    datatype = gdal.GDT_Float32
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    driver = gdal.GetDriverByName(format)
    ds_to_save = driver.Create(save_path, width, height, channels, datatype)

    for band in range(channels):
        ds_to_save.GetRasterBand(band + 1).WriteArray(image[:, :, band])
        ds_to_save.FlushCache()
    del image
    del ds_to_save     


