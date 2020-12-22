import numpy as np
import tensorflow as tf
from utils import norm, mul_psnr, rmse, MG, SAM, ERGAS
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
        self.save_path = "./test_save_images"

        self.dsen2_weight_path = "./DSen_weights/59_w.h5"
        self.lsrnl2_weight_path = "./lsrn_weights/69_w.h5"
        self.lsrnl1_weight_path = "./lsrnl1_weights/49_w.h5"
        self.lsrgan_weight_path = "./lsrgan_weights/29_w.h5"
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
        rmse_bicubic_sum = 0
        rmse_dsen2_sum = 0
        rmse_lsrnl2_sum = 0
        rmse_lsrnl1_sum = 0
        rmse_lsrgan_sum = 0
        rmse_elsrgan_sum = 0
        
        mg_bicubic_sum = 0
        mg_dsen2_sum = 0
        mg_lsrnl2_sum = 0
        mg_lsrnl1_sum = 0
        mg_lsrgan_sum = 0
        mg_elsrgan_sum = 0
        
        ergas_bicubic_sum = 0
        ergas_dsen2_sum = 0
        ergas_lsrnl2_sum = 0
        ergas_lsrnl1_sum = 0
        ergas_lsrgan_sum = 0
        ergas_elsrgan_sum = 0
        
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
            
            # calculate PSNR
            rmse_bicubic = rmse(norm_bicubic, norm_label)
            rmse_bicubic_sum += rmse_bicubic
            rmse_lsrnl2 = rmse(norm_lsrnl2, norm_label)
            rmse_lsrnl2_sum += rmse_lsrnl2
            rmse_lsrnl1 = rmse(norm_lsrnl1, norm_label)
            rmse_lsrnl1_sum += rmse_lsrnl1
            rmse_dsen2 = rmse(norm_dsen2, norm_label)
            rmse_dsen2_sum += rmse_dsen2
            rmse_lsrgan = rmse(norm_lsrgan, norm_label)
            rmse_lsrgan_sum += rmse_lsrgan
            rmse_elsrgan = rmse(norm_elsrgan, norm_label)
            rmse_elsrgan_sum += rmse_elsrgan
            # calculate MG
            mg_bicubic = MG(norm_bicubic)
            mg_bicubic_sum += mg_bicubic
            mg_dsen2 = MG(norm_dsen2)
            mg_dsen2_sum += mg_dsen2
            mg_lsrnl2 = MG(norm_lsrnl2)
            mg_lsrnl2_sum += mg_lsrnl2
            mg_lsrnl1 = MG(norm_lsrnl1)
            mg_lsrnl1_sum += mg_lsrnl1
            mg_lsrgan = MG(norm_lsrgan)
            mg_lsrgan_sum += mg_lsrgan
            mg_elsrgan = MG(norm_elsrgan)
            mg_elsrgan_sum += mg_elsrgan

            # calculate SAM
            ergas_bicubic = ERGAS(norm_bicubic,norm_label, norm_mul_img)
            ergas_bicubic_sum += ergas_bicubic
            ergas_dsen2 = ERGAS(norm_dsen2,norm_label, norm_mul_img)
            ergas_dsen2_sum += ergas_dsen2
            ergas_lsrnl2 = ERGAS(norm_lsrnl2, norm_label,norm_mul_img)
            ergas_lsrnl2_sum += ergas_lsrnl2
            ergas_lsrnl1 = ERGAS(norm_lsrnl1, norm_label,norm_mul_img)
            ergas_lsrnl1_sum += ergas_lsrnl1
            ergas_lsrgan = ERGAS(norm_lsrgan, norm_label,norm_mul_img)
            ergas_lsrgan_sum += ergas_lsrgan
            ergas_elsrgan = ERGAS(norm_elsrgan, norm_label,norm_mul_img)
            ergas_elsrgan_sum += ergas_elsrgan

            # two percent strech for show
            s_mul = twopercentlinearstrech(mul_image[:, :, 1:4])
            s_bicubic = twopercentlinearstrech(sr_bicubic_img[:, :, 1:4])
            s_dsen2 = twopercentlinearstrech(sr_dsen2_img[:, :, 1:4])
            s_lsrnl2 = twopercentlinearstrech(sr_lsrnl2_img[:, :, 1:4])
            s_lsrnl1 = twopercentlinearstrech(sr_lsrnl1_img[:, :, 1:4])
            s_lsrgan = twopercentlinearstrech(sr_lsrgan_img[:, :, 1:4])
            s_elsrgan = twopercentlinearstrech(sr_elsrgan_img[:, :, 1:4])
            s_label = twopercentlinearstrech(mul_img_label[:, :, 1:4])

            plt.figure()
            plt.subplots_adjust(left=0, right=1,top=1, bottom=0, wspace = 0.1, hspace=0.1)
            plt.subplot(2, 4, 1)
            plt.axis('off')
            plt.text(35, 75, "")
            plt.title("LRMS", fontsize = 8)
            plt.imshow(s_mul)
            
            plt.subplot(2, 4, 2)
            plt.axis('off')
            # plt.text(40, 150, str(rmse_dsen2))
            plt.title("bicubic", fontsize = 8)
            plt.imshow(s_bicubic)

            plt.subplot(2, 4, 3)
            plt.axis('off')
            # plt.text(40, 150, str(rmse_dsen2))
            plt.title("DSen2", fontsize = 8)
            plt.imshow(s_dsen2)

            plt.subplot(2, 4, 4)
            plt.axis('off')
            # plt.text(40, 150, str(rmse_lsrn))
            plt.title("RS-SRN(l1)", fontsize = 8)
            plt.imshow(s_lsrnl1)

            plt.subplot(2, 4, 5)
            plt.axis('off')
            # plt.text(40, 150, str(rmse_lsrn))
            plt.title("RS-SRN(l2)", fontsize = 8)
            plt.imshow(s_lsrnl2)

            plt.subplot(2, 4, 6)
            plt.axis('off')
            # plt.text(40 , 150, str(rmse_elsrgan))
            plt.title("RS-SRGAN(l1)", fontsize = 8)
            plt.imshow(s_elsrgan)

            plt.subplot(2, 4, 7)
            plt.axis('off')
            # plt.text(40 , 150, str(rmse_elsrgan))
            plt.title("RS-SRGAN(l2)", fontsize = 8)
            plt.imshow(s_lsrgan)

            plt.subplot(2, 4, 8)
            plt.axis('off')
            plt.title("HRMS", fontsize = 10)
            plt.imshow(s_label)

            save_path = self.save_path + "/test_" + str(iteration)
            plt.savefig(save_path, dpi=400)
        
        image_num = 300
        mg_bicubic_mean = mg_bicubic_sum/image_num
        mg_dsen2_mean = mg_dsen2_sum/image_num
        mg_lsrnl2_mean = mg_lsrnl2_sum/image_num
        mg_lsrnl1_mean = mg_lsrnl1_sum/image_num
        mg_lsrgan_mean = mg_lsrgan_sum/image_num
        mg_elsrgan_mean = mg_elsrgan_sum/image_num
        
        rmse_bicubic_mean = rmse_bicubic_sum/image_num
        rmse_dsen2_mean = rmse_dsen2_sum/image_num
        rmse_lsrnl2_mean = rmse_lsrnl2_sum/image_num
        rmse_lsrnl1_mean = rmse_lsrnl1_sum/image_num
        rmse_lsrgan_mean = rmse_lsrgan_sum/image_num
        rmse_elsrgan_mean = rmse_elsrgan_sum/image_num
        
        ergas_bicubic_mean = ergas_bicubic_sum/image_num
        ergas_dsen2_mean = ergas_dsen2_sum/image_num
        ergas_lsrnl2_mean = ergas_lsrnl2_sum/image_num
        ergas_lsrnl1_mean = ergas_lsrnl1_sum/image_num
        ergas_lsrgan_mean = ergas_lsrgan_sum/image_num
        ergas_elsrgan_mean = ergas_elsrgan_sum/image_num
        
        print("bicubic_mean_gridient:",mg_bicubic_mean)
        print("dsen2_mean_gridient:",mg_dsen2_mean)
        print("lsrnl2_mean_gridient:",mg_lsrnl2_mean)
        print("lsrnl1_mean_gridient:",mg_lsrnl1_mean)
        print("lsrgan_mean_gridient:",mg_lsrgan_mean)
        print("elsrgan_mean_gridient:",mg_elsrgan_mean)
        
        print("bicubic_mean_rmse",rmse_bicubic_mean)
        print("dsen2_mean_rmse",rmse_dsen2_mean)
        print("lsrnl2_mean_rmse",rmse_lsrnl2_mean)
        print("lsrnl1_mean_rmse",rmse_lsrnl1_mean)
        print("lsrgan_mean_rmse",rmse_lsrgan_mean)
        print("elsrgan_mean_rmse",rmse_elsrgan_mean)
        
        print("bicubic_mean_ergas",ergas_bicubic_mean)
        print("dsen2_mean_ergas",ergas_dsen2_mean)
        print("lsrnl2_mean_ergas",ergas_lsrnl2_mean)
        print("lsrnl1_mean_ergas",ergas_lsrnl1_mean)
        print("lsrgan_mean_ergas",ergas_lsrgan_mean)
        print("elsrgan_mean_ergas",ergas_elsrgan_mean)

if __name__ == "__main__":
    test = Test()
    test.test()



