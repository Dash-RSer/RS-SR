import numpy as np
import tensorflow as tf
from utils import norm, mul_psnr, rmse, MG
from model import GeneratorMixin, DSen2ModelMixin
from show import twopercentlinearstrech
from readrsimage import readrsimage
import cv2 as cv
import matplotlib.pyplot as plt

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
        self.lsrn_model= self.build_generator()
        self.residual_block_num = 8
        self.test_mul_path = "./testimages/mul11.tif"
        self.test_pan_path = "./testimages/pan11.tif"
        self.test_mul_label_path = "./testimages/label11.tif"
        self.save_path = "./misc/t11"
        self.dsen2_weight_path = "./DSen_weights/59_w.h5"
        self.lsrn_weight_path = "./lsrn_weights/69_w.h5"
        self.elsrgan_weight_path = "./elsrgan_weights/69_gen.h5"

    def test(self):
        # load model weigths
        self.dsen2_model.load_weights(self.dsen2_weight_path)
        self.lsrn_model.load_weights(self.lsrn_weight_path)
        self.elsrgan_model.load_weights(self.elsrgan_weight_path)
        
        # read test images
        pan_img = readrsimage(self.test_pan_path)
        mul_img = readrsimage(self.test_mul_path)
        mul_img_label = readrsimage(self.test_mul_label_path)

        # reshape for input model to predict
        pan_reshp = np.reshape(pan_img, (-1, 128, 128, 1))
        mul_reshp = np.reshape(mul_img, (-1, 64, 64, 7))
        # predict
        sr_lsrn_mul = self.lsrn_model([pan_reshp, mul_reshp])
        sr_dsen2_mul = self.dsen2_model([pan_reshp, mul_reshp])
        sr_elsrgan = self.elsrgan_model([pan_reshp, mul_reshp])

        # back shape
        sr_lsrn_img = np.reshape(sr_lsrn_mul, self.mul_hr_shape)
        sr_dsen2_img = np.reshape(sr_dsen2_mul, self.mul_hr_shape)
        sr_elsrgan_img = np.reshape(sr_elsrgan, self.mul_hr_shape)

        #norm for calc psnr, dtype is np.float32.
        norm_mul_img = norm(mul_img)
        norm_lsrn = norm(sr_lsrn_img)
        norm_dsen2 = norm(sr_dsen2_img)
        norm_elsrgan = norm(sr_elsrgan_img)
        norm_label = norm(mul_img_label)
        
        # calculate psnr
        rmse_lsrn = rmse(norm_lsrn, norm_label)
        rmse_dsen2 = rmse(norm_dsen2, norm_label)
        rmse_elsrgan = rmse(norm_elsrgan, norm_label)

        # calculate MG
        mg_dsen2 = MG(norm_dsen2)
        mg_lsrn = MG(norm_lsrn)
        mg_elsrgan = MG(norm_elsrgan)

        print("MG_Dsen2: ",mg_dsen2, " MG_LSRN: ", mg_lsrn, " MG_ELSRGAN:", mg_elsrgan)

        # two percent strech for show
        s_mul = twopercentlinearstrech(mul_img[:, :, 1:4])
        s_dsen2 = twopercentlinearstrech(sr_dsen2_img[:, :, 1:4])
        s_lsrn = twopercentlinearstrech(sr_lsrn_img[:, :, 1:4])
        s_elsrgan = twopercentlinearstrech(sr_elsrgan_img[:, :, 1:4])
        s_label = twopercentlinearstrech(mul_img_label[:, :, 1:4])

        #focus on small area
        """
        unknown
        [20:35, 45:60, :]
        [80:110,80:110, :]
        [60:90,60:90, :]
        [60:90,90:120, :]

        """
        a_mul = s_mul[25:40, 25:40, :]
        a_dsen2 = s_dsen2[50:80,50:80, :]
        a_lsrn = s_lsrn[50:80,50:80, :]
        a_elsrgan = s_elsrgan[50:80,50:80, :]
        a_label = s_label[50:80,50:80, :]

        plt.figure()
        plt.subplots_adjust(left=0, right=1,top=1, bottom=0, wspace = 0, hspace=0)
        plt.subplot(1, 5, 1)
        plt.axis('off')
        plt.text(35, 75, "")
        plt.title("lr", fontsize = 8)
        plt.imshow(s_mul)

        plt.subplot(1, 5, 2)
        plt.axis('off')
        plt.text(40, 150, str(rmse_dsen2))
        plt.title("DSen2", fontsize = 8)
        plt.imshow(s_dsen2)

        plt.subplot(1, 5, 3)
        plt.axis('off')
        plt.text(40, 150, str(rmse_lsrn))
        plt.title("LSRN", fontsize = 8)
        plt.imshow(s_lsrn)


        plt.subplot(1, 5, 4)
        plt.axis('off')
        plt.text(40 , 150, str(rmse_elsrgan))
        plt.title("ELSRGAN(ours)", fontsize = 8)
        plt.imshow(s_elsrgan)

        plt.subplot(1, 5, 5)
        plt.axis('off')
        # plt.text(48, 150, "0")
        plt.title("hr", fontsize = 10)
        plt.imshow(s_label)
        plt.savefig(self.save_path, dpi=400)

        plt.figure()
        plt.subplots_adjust(left=0, right=1,top=1, bottom=0, wspace = 0, hspace=0)
        plt.subplot(1, 5, 1)
        plt.axis('off')
        plt.imshow(a_mul)

        plt.subplot(1, 5, 2)
        plt.axis('off')
        plt.imshow(a_dsen2)

        plt.subplot(1, 5, 3)
        plt.axis('off')
        plt.imshow(a_lsrn)

        plt.subplot(1, 5, 4)
        plt.axis('off')
        plt.imshow(a_elsrgan)

        plt.subplot(1, 5, 5)
        plt.axis('off')
        plt.imshow(a_label)

        plt.savefig(self.save_path+"_detais", dpi=400)

if __name__ == "__main__":
    test = Test()
    test.test()




