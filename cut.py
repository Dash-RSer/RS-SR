from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import cv2 as cv
import math
import os

class Cut(object):
    
    def __init__(self, mtl_file_path, pan_save_path = "", multi_save_path = ""):
        
        self._pan_band_size = (256, 256, 1)
        self._multi_band_size = (128, 128, 1)
        self._mtl_file_name = mtl_file_path
        self._pan_save_path = pan_save_path
        self._multi_save_path = multi_save_path
        self._band_names = []
        self._get_names()
        self._multi_valid_area_left_top = (1300, 1200)
        self._multi_valid_area_right_bottom = (6400, 6400)
        self._pan_valid_area_left_top = (2600, 2400)
        self._pan_valid_area_right_bottom = (12800, 12800)

    
    def _get_names(self):
        path_word_list = self._mtl_file_name.split("_")[:-1]
        base_name = ""
        for string in path_word_list:
            base_name += string + "_"
        BAND_NUMBER = 8
        for band in range(BAND_NUMBER):
            band_name = base_name + "B" + str(band+1) + ".tif"
            self._band_names.append(band_name)

    def read_mul_image(self):
        BAND_NUM = 8
        dataset = gdal.Open(self._band_names[0], gdal.GA_ReadOnly)
        height = dataset.RasterYSize
        width = dataset.RasterXSize

        datatype = np.float32
        MUL_BAND_NUM = 7
        del dataset
        mul_image = np.zeros((height, width, MUL_BAND_NUM), dtype=datatype)
        for band in range(MUL_BAND_NUM):
            dataset = gdal.Open(self._band_names[band], gdal.GA_ReadOnly)
            band_data = dataset.GetRasterBand(1)
            mul_image[:, :, band] = band_data.ReadAsArray()
            del dataset, band_data
        y0 = self._multi_valid_area_left_top[1]
        y1 = self._multi_valid_area_right_bottom[1]
        x0 = self._multi_valid_area_left_top[0]
        x1 = self._multi_valid_area_right_bottom[0]
        mul_image = mul_image[y0:y1, x0:x1, :]

        return mul_image
    
    def read_pan_image(self):
        PAN_INDEX = 7
        dataset = gdal.Open(self._band_names[PAN_INDEX], gdal.GA_ReadOnly)
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        datatype = np.float32
        pan_image = np.zeros((height, width, 1), dtype = datatype)
        band_data = dataset.GetRasterBand(1)
        pan_image[:, :, 0] = band_data.ReadAsArray()
        
        del dataset, band_data
        y0 = self._pan_valid_area_left_top[1]
        y1 = self._pan_valid_area_right_bottom[1]
        x0 = self._pan_valid_area_left_top[0]
        x1 = self._pan_valid_area_right_bottom[0]

        pan_image = pan_image[y0:y1, x0:x1, :]

        return pan_image

    @staticmethod
    def write(path, img):
        datatype = gdal.GDT_Float32

        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        driver = gdal.GetDriverByName("GTiff")
        ds_to_save = driver.Create(path, width, height, channels, datatype)

        for band in range(channels):
            ds_to_save.GetRasterBand(band + 1).WriteArray(img[:, :, band])
            ds_to_save.FlushCache()

        del ds_to_save


    def cut(self):
        if self._pan_save_path == "":
            if os.path.exists("./pan") == False:
                os.mkdir("./pan")
            self._pan_save_path = "./pan"
        if self._multi_save_path == "":
            if os.path.exists("./multi") == False:
                os.mkdir("./multi")
            self._multi_save_path = "./multi"
        
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

        mul_image = self.read_mul_image()
        mul_image = mul_image[:useful_mul_height, :useful_mul_width, :]
        
        counter = 0
        for tile_row in range(cut_height_number):
            for tile_col in range(cut_width_number):
                tile = mul_image[tile_row*multi_cut_height:(tile_row+1)*multi_cut_height, tile_col*multi_cut_width:(tile_col+1)*multi_cut_width, :]
                tile = np.float32(tile)/SCALE
                image_name = str(counter) + "_m.tif"
                path = os.path.join(self._multi_save_path, image_name)
                Cut.write(path, tile)
                counter += 1
                del tile
        del mul_image

        pan_image = self.read_pan_image()
        pan_image = pan_image[:useful_pan_height, :useful_pan_width, :]

        counter = 0
        for tile_row in range(cut_height_number):
            for tile_col in range(cut_width_number):
                tile = pan_image[tile_row*pan_cut_height:(tile_row+1)*pan_cut_height, tile_col*pan_cut_width:(tile_col+1)*pan_cut_width, :]
                image_name = str(counter)+'_p.tif'
                path = os.path.join(self._pan_save_path, image_name)
                tile = np.float32(tile)/SCALE
                Cut.write(path, tile)
                counter += 1
                del tile
        del pan_image

def main():
    mtl_file_path = "F:\SRGAN_program\dataset\LC81290352019095LGN00\LC08_L1TP_129035_20190405_20190422_01_T1_MTL.txt"
    cuttool = Cut(mtl_file_path=mtl_file_path)
    cuttool.cut()


if __name__ == '__main__':
    main()
                        



                        
                

                
                

        
        

        


    

    

