import os
import matplotlib.pyplot as plt
from utils import norm
from show import twopercentlinearstrech
from imagegiver import ImageGiver
from utils import mul_psnr

class MakeSampleMixin(object):    
    def make_sample(self, epoch, iteration, save_dir = "./samples"):    
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        sample_img_giver = ImageGiver()
        flag, hr_pan, hr_mul, lr_mul, lr_pan = sample_img_giver.give(1)
        # Because of disign problem, I use this way to do predict. :(
        try:
            fake_hr_mul = self.generator.predict([lr_pan, lr_mul])
        except AttributeError:
            fake_hr_mul = self.model.predict([lr_pan, lr_mul])
        # for show (the datatype is np.int8)
        
        lr_mul_show = twopercentlinearstrech(lr_mul[0,:, :, 1:4])
        fake_hr_mul_show = twopercentlinearstrech(fake_hr_mul[0, :, :, 1:4])
        hr_mul_show = twopercentlinearstrech(hr_mul[0, :, :, 1:4])
        
        # for calc (the datatype is np.float32)
        
        fake_hr_mul_calc = norm(fake_hr_mul[0, :, :, 1:4])
        hr_mul_calc = norm(hr_mul[0, :, :, 1:4])
        psnr = round(mul_psnr(fake_hr_mul_calc, hr_mul_calc), 2)

        print("epoch:{}, iteration:{}, psnr:{}".format(epoch, iteration, psnr))
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('lr')
        plt.imshow(lr_mul_show)
        
        plt.subplot(1, 3, 2)
        plt.title('fake_hr')
        plt.text(0, 140, str(psnr))
        plt.imshow(fake_hr_mul_show)
        
        plt.subplot(1, 3, 3)
        plt.title('real_hr')
        plt.imshow(hr_mul_show)
        
        plt.savefig(save_dir + "/{}_{}".format(epoch, iteration))