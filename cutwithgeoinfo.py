from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import cv2 as cv
import math
import os
from cut import Cut

class CutWithGeoInfo(Cut):

    def __init__(self, mtl_file_path, counter = 0, pan_save_path = "", mul_save_path = ""):
        super(CutWithGeoInfo, self).__init__(mtl_file_path, pan_save_path, mul_save_path)
        self.counter = counter

    def read_with_geoinfo(self):
        BAND_NUMa = 8
        dataset = gdal.Open(self._band_names[0], gdal.GA_ReadOnly)
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        MUL_BAND_NUM = 7
        datatype = np.float32
        
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        del dataset

        image = np.zeros((height, width, MUL_BAND_NUM), dtype = datatype)
        
        for band in range(MUL_BAND_NUM):
            dataset = gdal.Open(self._band_names[band], gdal.GA_ReadOnly)
            band_image = dataset.GetRasterBand(1)
            image[:, :, band] = band_image.ReadAsArray()
            del dataset, band_image

        y0 = self._multi_valid_area_left_top[1]
        y1 = self._multi_valid_area_right_bottom[1]
        x0 = self._multi_valid_area_left_top[0]
        x1 = self._multi_valid_area_right_bottom[0]
        image = image[y0:y1, x0:x1, :]

        x_0_cor = geotransform[0] + x0 * geotransform[1]
        y_0_cor = geotransform[3] + y0 * geotransform[5]

        new_geotransform = (x_0_cor, geotransform[1], 0, y_0_cor, 0, geotransform[5])
     
        return image, projection, geotransform

    def read_pan_image_with_geoinfo(self):
        PAN_INDEX = 7
        dataset = gdal.Open(self._band_names[PAN_INDEX], gdal.GA_ReadOnly)
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        datatype = np.float32
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        pan_image = np.zeros((height, width, 1), dtype = datatype)
        band_data = dataset.GetRasterBand(1)
        pan_image[:, :, 0] = band_data.ReadAsArray()
        
        del dataset, band_data
        y0 = self._pan_valid_area_left_top[1]
        y1 = self._pan_valid_area_right_bottom[1]
        x0 = self._pan_valid_area_left_top[0]
        x1 = self._pan_valid_area_right_bottom[0]

        pan_image = pan_image[y0:y1, x0:x1, :]

        x_0_cor = geotransform[0] + x0 * geotransform[1]
        y_0_cor = geotransform[3] + y0 * geotransform[5]
        
        new_geotransform = (x_0_cor, geotransform[1], 0, y_0_cor, 0, geotransform[5])
        
        return pan_image, projection, geotransform

    @staticmethod
    def write_with_geo(save_path, image, projection, geotransform, format = 'GTiff'):
        datatype = gdal.GDT_Float32
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]

        driver = gdal.GetDriverByName(format)
        ds_to_save = driver.Create(save_path, width, height, channels, datatype)
        ds_to_save.SetGeoTransform(geotransform)
        ds_to_save.SetProjection(projection)

        for band in range(channels):
            ds_to_save.GetRasterBand(band + 1).WriteArray(image[:, :, band])
            ds_to_save.FlushCache()


        del image
        del ds_to_save  

    def cut_with_geo_information(self):
        if self._pan_save_path == "":
            if os.path.exists("./pan") == False:
                os.mkdir("./pan")
            self._pan_save_path = "./pan"

        if self._multi_save_path == "":
            if os.path.exists("./mul") == False:
                os.mkdir("./mul")
            self._multi_save_path = "./mul"

        mul_height = self._multi_valid_area_right_bottom[1] - self._multi_valid_area_left_top[1]
        mul_width = self._multi_valid_area_right_bottom[0] - self._multi_valid_area_left_top[0]
        pan_cut_height = self._pan_band_size[0]
        pan_cut_width = self._pan_band_size[1]
        multi_cut_height = self._multi_band_size[0]
        multi_cut_width = self._multi_band_size[1]

        cut_height_number = math.floor(mul_height / multi_cut_height)
        cut_width_number = math.floor(mul_width / multi_cut_width)


        useful_mul_height = cut_height_number * multi_cut_height
        useful_mul_width = cut_width_number * multi_cut_width
        useful_pan_height = cut_height_number * pan_cut_height
        useful_pan_width = cut_width_number * pan_cut_width

        BAND_NUMBER = 8
        MUL_NUMBER = 7
        PAN_INDEX = 7
        SCALE = 10000

        mul_image, projection, mul_geotransform = self.read_with_geoinfo()
        mul_image = mul_image[:useful_mul_height, :useful_mul_width, :]

        x_origin_point = mul_geotransform[0]
        x_pixel_size = mul_geotransform[1]
        x_ro = mul_geotransform[2]
        y_origin_point = mul_geotransform[3]
        y_ro = mul_geotransform[4]
        y_pixel_size = mul_geotransform[5]

        
        x_offset = 0
        y_offset = 0
        counter_record = self.counter
        for tile_row in range(cut_height_number):
            for tile_col in range(cut_width_number):
                tile = mul_image[tile_row*multi_cut_height:(tile_row+1)*multi_cut_height, tile_col*multi_cut_width:(tile_col+1)*multi_cut_width, :]
                tile = np.float32(tile)/SCALE
                image_name = str(self.counter) + "_m.tif"
                path = os.path.join(self._multi_save_path, image_name)

                tile_x_origin_point = x_origin_point + x_pixel_size * tile_col * multi_cut_width
                tile_y_origin_point = y_origin_point + y_pixel_size * tile_row * multi_cut_height
                tile_geotransform = (tile_x_origin_point, x_pixel_size, x_ro, tile_y_origin_point, y_ro, y_pixel_size)

                CutWithGeoInfo.write_with_geo(path, tile, projection,tile_geotransform)
                self.counter += 1
                del tile
        del mul_image

        pan_image, projection, pan_geotransform = self.read_pan_image_with_geoinfo()

        x_origin_point = pan_geotransform[0]
        x_pixel_size = pan_geotransform[1]
        x_ro = pan_geotransform[2]
        y_origin_point = pan_geotransform[3]
        y_ro = pan_geotransform[4]
        y_pixel_size = pan_geotransform[5]

        self.counter = counter_record
        for tile_row in range(cut_height_number):
            for tile_col in range(cut_width_number):
                tile = pan_image[tile_row*pan_cut_height:(tile_row+1)*pan_cut_height, tile_col*pan_cut_width:(tile_col+1)*pan_cut_width, :]
                image_name = str(self.counter)+'_p.tif'
                path = os.path.join(self._pan_save_path, image_name)
                tile = np.float32(tile)/SCALE
                tile_x_origin_point = x_origin_point + x_pixel_size * tile_col * pan_cut_width
                tile_y_origin_point = y_origin_point + y_pixel_size * tile_row * pan_cut_height
                tile_geotransform = (tile_x_origin_point, x_pixel_size, x_ro, tile_y_origin_point, y_ro, y_pixel_size)        
                CutWithGeoInfo.write_with_geo(path, tile, projection, tile_geotransform)
                self.counter += 1
                del tile

        del pan_image

def main():
    # mtl_file_path = "F:\SRGAN_program\dataset\LC81290352019095LGN00\LC08_L1TP_129035_20190405_20190422_01_T1_MTL.txt"
    # mtl_file_path = "F:\SRGAN_program\dataset\LC81220402017360LGN00\LC08_L1TP_122040_20171226_20180103_01_T1_MTL.txt"
    # mtl_file_path = "F:\SRGAN_program\dataset\LC81230372016205LGN01\LC08_L1TP_123037_20160723_20170322_01_T1_MTL.txt"
    mtl_file_path = "F:\SRGAN_program\dataset\LC81270362013272LGN01\LC08_L1TP_127036_20130929_20170501_01_T1_MTL.txt"
    cuttool = CutWithGeoInfo(mtl_file_path, 0, "F:\SRGAN_program\\newnet\\test_pan", "F:\SRGAN_program\\newnet\\test_mul")
    cuttool.cut_with_geo_information()

if __name__ == "__main__":
    main()    



