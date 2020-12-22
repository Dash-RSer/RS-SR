import cv2 as cv
import numpy as np
import random
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from show import twopercentlinearstrech, graytwopercentlinearstrech

class ImageGiver(object):
    def __init__(self, pan_path = "./pan", multi_path = "./mul", shuffle = True):
        self._pan_path = pan_path
        self._multi_path = multi_path
        self._multi_imagename_list = []
        self._counter = 0
        self._if_shuffle = shuffle
        self._get_list_names()

    def _get_list_names(self):
        self._multi_imagename_list = os.listdir(self._multi_path)
        if self._if_shuffle:
            random.shuffle(self._multi_imagename_list)
    
    def read_image(self, filename):
        dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        channels = dataset.RasterCount
        datatype = np.float32

        image = np.zeros((height, width, channels), dtype=datatype)
        for band in range(channels):
            band_data = dataset.GetRasterBand(band + 1)
            image[:, :, band] = band_data.ReadAsArray()
        return image

    @staticmethod
    def _get_pan_name(mtl_image_name):
        pan_name = mtl_image_name.split("_")[0] + "_p.tif"
        return pan_name

    def give(self, batchsize = 32):
        
        high_res_pan_images = []
        low_res_pan_images = []
        high_res_mul_images = []
        low_res_mul_images = []
        flag = 0

        for i in range(batchsize):
            i += self._counter
            if i == len(self._multi_imagename_list):
                flag = 1
                break
            multi_image_name = self._multi_imagename_list[i]
            pan_image_name = self._get_pan_name(multi_image_name)


            multi_image_path = os.path.join(self._multi_path, multi_image_name)
            pan_image_path = os.path.join(self._pan_path, pan_image_name)
            
            high_res_mul_image = self.read_image(multi_image_path)
            high_res_pan_image = self.read_image(pan_image_path)

            low_mul_size = (64, 64)
            low_pan_size = (128, 128)
            low_mul_resize = (64, 64, 7)
            low_pan_resize = (128, 128, 1)
            low_res_multi_image =\
                np.reshape(cv.resize(high_res_mul_image, low_mul_size, interpolation=cv.INTER_LINEAR), low_mul_resize)
            low_res_pan_image =\
                np.reshape(cv.resize(high_res_pan_image, low_pan_size, interpolation=cv.INTER_LINEAR), low_pan_resize)

            high_res_pan_images.append(high_res_pan_image)
            high_res_mul_images.append(high_res_mul_image)
            low_res_mul_images.append(low_res_multi_image)
            low_res_pan_images.append(low_res_pan_image)

        high_res_pan_images = np.array(high_res_pan_images)
        high_res_mul_images = np.array(high_res_mul_images)
        low_res_mul_images = np.array(low_res_mul_images)
        low_res_pan_images = np.array(low_res_pan_images)
        
        self._counter += batchsize
        return flag, high_res_pan_images, high_res_mul_images, low_res_mul_images, low_res_pan_images

if __name__ == "__main__":
    pass

    



            




