import numpy as np
import tensorflow as tf
from utils import norm, mul_psnr, rmse, MG, SAM, ERGAS, eachbandrmse
from model import GeneratorMixin, DSen2ModelMixin
from show import twopercentlinearstrech
from readrsimage import readrsimage,writersimage
import cv2 as cv
import matplotlib.pyplot as plt
from imagegiver import ImageGiver

class Test(GeneratorMixin, DSen2ModelMixin):
    def __init__(self):
        # config all
        self.pan_lr_shape = (128, 128, 1)
        self.mul_lr_shape = (64, 64, 7)
        self.pan_hr_shape = (256, 256, 1)
        self.mul_hr_shape = (128, 128, 7)
        self.dense_block_num = 8
        self.dense_growth_channels = 32
        self.residual_block_num = 8
        self.elsrgan_model = self.build_generator()
        self.lsrgan_model = self.build_generator()
        self.dsen2_model = self.build_DSen2()
        self.lsrnl2_model = self.build_generator()
        self.lsrnl1_model = self.build_generator()

        self.residual_block_num = 8
        self.test_multi_path = "./test_mul"
        self.test_pan_path = "./test_pan"
        self.save_path = "./mse_images"

        self.dsen2_weight_path = "./DSen_weights/69_w.h5"
        self.lsrnl2_weight_path = "./lsrn_weights/69_w.h5"
        self.lsrnl1_weight_path = "./lsrnl1_weights/69_w.h5"
        self.lsrgan_weight_path = "./lsrgan_weights/59_w.h5"
        self.elsrgan_weight_path = "./elsrgan_weights/69_gen.h5"

    def test(self):
        # load model weigths
        self.dsen2_model.load_weights(self.dsen2_weight_path)
        self.lsrnl2_model.load_weights(self.lsrnl2_weight_path)
        self.lsrnl1_model.load_weights(self.lsrnl1_weight_path)
        self.lsrgan_model.load_weights(self.lsrgan_weight_path)
        self.elsrgan_model.load_weights(self.elsrgan_weight_path)
        MAX_NUM = 300
        giver = ImageGiver(pan_path = self.test_pan_path, 
                        multi_path= self.test_multi_path,
                        shuffle = True)
        
        for iteration in range(MAX_NUM):
            flag, hr_pan, hr_mul, lr_mul, lr_pan = giver.give(1)
            if flag == 1:
                break
            
            pan_img = lr_pan
            mul_img = lr_mul
            mul_img_label = hr_mul

            # reshape for input model to predict
            pan_reshp = np.reshape(pan_img, (1, 128, 128, 1))
            mul_reshp = np.reshape(mul_img, (1, 64, 64, 7))
            # predict
            sr_bicubic_mul = tf.image.resize(mul_reshp, (128, 128), method = tf.image.ResizeMethod.BICUBIC).numpy()
            sr_lsrnl2_mul = self.lsrnl2_model([pan_reshp, mul_reshp])
            sr_lsrnl1_mul = self.lsrnl1_model([pan_reshp, mul_reshp])
            sr_dsen2_mul = self.dsen2_model([pan_reshp, mul_reshp])
            sr_lsrgan = self.lsrgan_model([pan_reshp, mul_reshp])
            sr_elsrgan = self.elsrgan_model([pan_reshp, mul_reshp])

            # back shape
            mul_image = np.reshape(mul_reshp, self.mul_lr_shape)
            sr_bicubic_img = np.reshape(sr_bicubic_mul, self.mul_hr_shape)
            sr_lsrnl2_img = np.reshape(sr_lsrnl2_mul, self.mul_hr_shape)
            sr_lsrnl1_img = np.reshape(sr_lsrnl1_mul, self.mul_hr_shape)
            sr_dsen2_img = np.reshape(sr_dsen2_mul, self.mul_hr_shape)
            sr_lsrgan_img = np.reshape(sr_lsrgan, self.mul_hr_shape)
            sr_elsrgan_img = np.reshape(sr_elsrgan, self.mul_hr_shape)
            mul_img_label = np.reshape(mul_img_label, self.mul_hr_shape)

            #norm for calc psnr, dtype is np.float32.
            norm_mul_img = norm(mul_image)
            norm_bicubic = norm(sr_bicubic_img)
            norm_lsrnl2 = norm(sr_lsrnl2_img)
            norm_lsrnl1 = norm(sr_lsrnl1_img)
            norm_dsen2 = norm(sr_dsen2_img)
            norm_lsrgan = norm(sr_lsrgan_img)
            norm_elsrgan = norm(sr_elsrgan_img)
            norm_label = norm(mul_img_label)
            
            s_label = twopercentlinearstrech(norm_label[:, :, 1:4])

            rmse_bicubic = eachbandrmse(norm_bicubic, norm_label)
            rmse_dsen2 = eachbandrmse(norm_dsen2, norm_label)
            rmse_lsrnl1 = eachbandrmse(norm_lsrnl1, norm_label)
            rmse_lsrnl2 = eachbandrmse(norm_lsrnl2, norm_label)
            rmse_lsrgan = eachbandrmse(norm_lsrgan, norm_label)
            rmse_elsrgan = eachbandrmse(norm_elsrgan, norm_label)

            x = np.array([1, 2, 3, 4, 5, 6, 7],np.float32)
            
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.axis('off')
            plt.imshow(s_label)

            plt.subplot(2,1,2)
            plt.plot(x, rmse_bicubic, label = 'bicubic')
            plt.plot(x, rmse_dsen2, label = 'DSen2')
            plt.plot(x, rmse_lsrnl1, label = 'RS-SRN(l1)')
            plt.plot(x, rmse_lsrnl2, label = 'RS-SRN(l2)')
            plt.plot(x, rmse_elsrgan, label = 'RS-SRGAN(l1)')
            plt.plot(x, rmse_lsrgan, label = 'RS-SRGAN(l2)')          
            plt.xlabel('band')
            plt.ylabel('RMSE')
            plt.legend(fontsize = 5)
            # 50 #141 # 47


            save_path = self.save_path + "/mse_figure_" + str(iteration)
            plt.savefig(save_path, dpi=400)


            # plt.figure()
            # plt.subplots_adjust(left=0, right=1,top=1, bottom=0, wspace = 0.1, hspace=0.1)
            # plt.subplot(2, 4, 1)
            # plt.axis('off')
            # plt.text(35, 75, "")
            # plt.title("LRMS", fontsize = 8)
            # plt.imshow(s_mul)
            
            # plt.subplot(2, 4, 2)
            # plt.axis('off')
            # # plt.text(40, 150, str(rmse_dsen2))
            # plt.title("bicubic", fontsize = 8)
            # plt.imshow(s_bicubic)

            # plt.subplot(2, 4, 3)
            # plt.axis('off')
            # # plt.text(40, 150, str(rmse_dsen2))
            # plt.title("DSen2", fontsize = 8)
            # plt.imshow(s_dsen2)

            # plt.subplot(2, 4, 4)
            # plt.axis('off')
            # # plt.text(40, 150, str(rmse_lsrn))
            # plt.title("LSRN(l1)", fontsize = 8)
            # plt.imshow(s_lsrnl1)

            # plt.subplot(2, 4, 5)
            # plt.axis('off')
            # # plt.text(40, 150, str(rmse_lsrn))
            # plt.title("LSRN(l2)", fontsize = 8)
            # plt.imshow(s_lsrnl2)

            # plt.subplot(2, 4, 6)
            # plt.axis('off')
            # # plt.text(40 , 150, str(rmse_elsrgan))
            # plt.title("LSRGAN(l1)", fontsize = 8)
            # plt.imshow(s_elsrgan)

            # plt.subplot(2, 4, 7)
            # plt.axis('off')
            # # plt.text(40 , 150, str(rmse_elsrgan))
            # plt.title("LSRGAN(l2)", fontsize = 8)
            # plt.imshow(s_lsrgan)

            # plt.subplot(2, 4, 8)
            # plt.axis('off')
            # plt.title("HRMS", fontsize = 10)
            # plt.imshow(s_label)


        


if __name__ == "__main__":
    test = Test()
    test.test()
