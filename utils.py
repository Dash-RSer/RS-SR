import numpy as np
from readrsimage import readrsimage_with_geoinfo,readrsimage
import cv2 as cv
from cutwithgeoinfo import CutWithGeoInfo
import os

def make_test_image(pan_image_path, mul_image_path):
    pan_image, pan_prj, pan_geo = readrsimage_with_geoinfo(pan_image_path)
    mul_image, mul_prj, mul_geo = readrsimage_with_geoinfo(mul_image_path)
    pan_resample_image = cv.resize(pan_image, (128, 128), interpolation=cv.INTER_LINEAR)
    mul_resample_image = cv.resize(mul_image, (64, 64), interpolation = cv.INTER_LINEAR)
    new_pan_geo = (pan_geo[0], 30, pan_geo[2], pan_geo[3], pan_geo[4], -30)
    new_mul_geo = (mul_geo[0], 60, mul_geo[2], mul_geo[3], mul_geo[4], -60)
    if os.path.exists("./testimages") == False:
        os.mkdir("./testimages")
    pan_resample_image = np.reshape(pan_resample_image, (128, 128, 1))
    mul_resample_image = np.reshape(mul_resample_image, (64, 64, 7))
    CutWithGeoInfo.write_with_geo("./testimages/pan11.tif", pan_resample_image, pan_prj, new_pan_geo)
    CutWithGeoInfo.write_with_geo("./testimages/mul11.tif", mul_resample_image, mul_prj, new_mul_geo)
    CutWithGeoInfo.write_with_geo("./testimages/label11.tif", mul_image, mul_prj, mul_geo)

def mul_psnr(fake_hr, real_hr):
    """
    Only for 8 bit images.
    """
    # print(fake_hr.shape)
    channels = fake_hr.shape[2]
    fake_hr = fake_hr.astype(np.float32)
    real_hr = real_hr.astype(np.float32)
    
    def single_band_psnr(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        psnr = 10 * np.log10(255 * 255 / mse)
        return psnr
    
    psnr_sum = 0
    for band in range(channels):
        fake_band_img = fake_hr[:, :, band]
        real_band_img = real_hr[:, :, band]
        psnr_sum += single_band_psnr(fake_band_img, real_band_img)
    
    psnr = round(psnr_sum/channels, 2)
    return psnr

def rmse(fake_hr, real_hr):
    """
    Only for 8 bit images.
    """
    # print(fake_hr.shape)
    if len(fake_hr.shape) == 3:
        channels = fake_hr.shape[2]
    else:
        channels = 1
        fake_hr = np.reshape(fake_hr, (fake_hr.shape[0], fake_hr.shape[1], 1))
        real_hr = np.reshape(real_hr, (real_hr.shape[0], real_hr.shape[1], 1))
    fake_hr = fake_hr.astype(np.float32)
    real_hr = real_hr.astype(np.float32)
    
    def single_mse(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        return mse
    
    mse_sum = 0
    for band in range(channels):
        fake_band_img = fake_hr[:, :, band]
        real_band_img = real_hr[:, :, band]
        mse_sum += single_mse(fake_band_img, real_band_img)
    
    rmse = np.sqrt(mse_sum)
    rmse = round(rmse/channels, 2)
    
    return rmse

def MG(image):

    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    MG_sum = 0
    for channel in range(channels):
        channel_mg_sum = 0
        for row in range(1, height):
            for column in range(1, width):
                dy = image[row, column, channel] - image[row-1, column, channel]
                dx = image[row, column, channel] - image[row, column - 1, channel]
                channel_mg_sum += np.sqrt((dx**2 + dy**2)/2)
        channels_mg = channel_mg_sum/((height-1)*(width-1))
        MG_sum += channels_mg
    
    MG = round(MG_sum/channels, 2)
    return MG

def ERGAS(hr_mul, label, lr_mul):
    """
    calc ergas.
    """
    h = 30
    l = 60


    channels = hr_mul.shape[2]

    inner_sum = 0
    for channel in range(channels):
        band_img1 = hr_mul[:, :, channel]
        band_img2 = label[:, :, channel]
        band_img3 = lr_mul[:, :, channel]

        rmse_value = rmse(band_img1, band_img2)
        m = np.mean(band_img3)
        inner_sum += np.power((rmse_value/m), 2)
    mean_sum = inner_sum/channels
    ergas = 100*(h/l)*np.sqrt(mean_sum)

    return ergas
        


def SAM(image1, image2):
    """
    Calculate SAM(spectral angle mapper).

    """
    if image1.shape != image2.shape:
        raise Exception("shape is not the same.")

    height = image1.shape[0]
    width = image1.shape[1]
    channels = image1.shape[2]

    def vector_norm(vector):
        length = len(vector)
        square_sum = 0
        for i in range(length):
            value = np.power(vector[i], 2)
            square_sum += value
        vector_norm = np.sqrt(square_sum)
        return vector_norm

    def vector_inner_poduct(vector1, vector2):
        dot_sum = 0
        length = len(vector1)
        for i in range(length):
            value = vector1[i] * vector2[i]
            dot_sum += value
        return dot_sum
    pixel_num = height * width
    total_sam = 0
    for row in range(height):
        for col in range(width):
            vector1 = image1[row, col, :]
            vector2 = image2[row, col, :]

            u_value = vector_inner_poduct(vector1, vector2)
            d_value = vector_norm(vector1)*vector_norm(vector2)

            single_pixel_sam = np.arccos(u_value/d_value)
            total_sam += single_pixel_sam
    
    mean_sam = total_sam/pixel_num
    return mean_sam

def eachbandrmse(image1, image2):
    channels = image1.shape[2]
    rmse_values = np.zeros(channels, np.float32)
    for i in range(channels):
        rmse_values[i] = rmse(image1[:, :, i],image2[:, :, i])
    return rmse_values


def norm(image):
    max_value = np.max(image)
    min_value = np.min(image)
    image = 255 * (image - min_value)/float(max_value - min_value)
    return image

def main():
    # make_test_image("./testimages/4656_p.tif", "./testimages/4656_m.tif")
    image1 = readrsimage()
    image2 = readrsimage()


if __name__ == "__main__":
    main()