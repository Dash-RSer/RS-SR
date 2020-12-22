"""
My new version code for LSRGAN.
Use mixin method for clean.
The main change is using raletivistic discirminator instead of
standard discriminator.
based on esrgan, dsen2.
only for landsat8-oli now.
"""
"""
配置：
使用了Dense Block
使用了l1损失
使用了相对平均判别器
没有使用全变分正则化
"""

import tensorflow as tf
from tensorflow.keras import layers
from layers import DenseBlock, ConvBlock
from imagegiver import ImageGiver
from utils import norm
from show import twopercentlinearstrech, graytwopercentlinearstrech
from utils import norm, mul_psnr
import matplotlib.pyplot as plt
import os
from makesample import MakeSampleMixin
from model import DiscriminatorMixin, GeneratorMixin

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True) # only one gpu i have

class LSRGAN(GeneratorMixin, DiscriminatorMixin, MakeSampleMixin):
    """LSRGAN network.
    
    This is a Generative Adversarial Networks(GANs) for super resolution
    sepecially for Landsat8-OLI sensor.
    
    Only a early version for experiment, I will update it to a
    application version later.

    Arguments:
        pan_path: A string. Your pan images path, it should be cut by the class
         CutWithGeoInfo.
        mul_path: A string. Your multi spectral images path, it should be cut by
         the class CutWithGeoInfo

    """
    def __init__(self, pan_path = "/content/drive/My Drive/lsrgan/newnet/pan", mul_path = "/content/drive/My Drive/lsrgan/newnet/mul"):
        # config parameters here
        self.pan_path = pan_path
        self.mul_path = mul_path
        self.pan_hr_shape = (256, 256, 1)
        self.pan_lr_shape = (128, 128, 1)
        self.mul_hr_shape = (128, 128, 7)
        self.mul_lr_shape = (64, 64, 7)
        self.dense_block_num = 8
        self.dense_growth_channels = 32
        self.batchsize = 8
        self.patch_shape = (8, 8, 1)
        self.epoch = 30
        self.start_epoch = 22
        self.scale = 0.005 # The adjust para in generator's loss
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gen_optimizer = tf.keras.optimizers.Adam(0.0001, clipnorm = 1)
        self.disc_optimizer = tf.keras.optimizers.Adam(0.0001, clipnorm = 1)

    def discriminator_loss(self, real_output, fake_output):
        Ra_loss_rf = tf.math.sigmoid((real_output) - tf.math.reduce_mean(fake_output, axis = 0))
        Ra_loss_fr = tf.math.sigmoid((fake_output) - tf.math.reduce_mean(real_output, axis = 0))
        L_Ra_d = - tf.math.reduce_mean(tf.math.log(Ra_loss_rf + 1e-6 )) - tf.math.reduce_mean(tf.math.log(1- Ra_loss_fr + 1e-6))
        return L_Ra_d
    
    def generator_adversarial_loss(self, real_output, fake_output):
        Ra_loss_rf = tf.math.sigmoid((real_output) - tf.math.reduce_mean(fake_output, axis = 0))
        Ra_loss_fr = tf.math.sigmoid((fake_output) - tf.math.reduce_mean(real_output, axis = 0))
        L_Ra_g = - tf.math.reduce_mean(tf.math.log(1 - Ra_loss_rf + 1e-6)) - tf.math.reduce_mean(tf.math.log(Ra_loss_fr + 1e-6))
        return L_Ra_g


    def train(self):

        gen_l1_loss_fn = tf.keras.losses.MeanAbsoluteError()
        
        # ckpt_dir = "./training_checkpoint"
        # ckpt_prefix = os.path.join(ckpt_dir, "ckpt")
        ckpt = tf.train.Checkpoint(gen_opt = self.gen_optimizer,
                                    disc_opt = self.disc_optimizer,
                                    generator = self.generator, 
                                    discriminator = self.discriminator)
        ckpt_manager = tf.train.CheckpointManager(ckpt, '/content/drive/My Drive/lsrgan/newnet/esrgan_tf_ckpts', max_to_keep = 10)   

        @tf.function
        def train_step(lr_pans, lr_muls, hr_muls):
            
            gen_hr_muls = self.generator([lr_pans, lr_muls])
            with tf.GradientTape() as tape:
                fake_pred = self.discriminator(gen_hr_muls)
                real_pred = self.discriminator(hr_muls)
                disc_loss = self.discriminator_loss(real_pred, fake_pred)

            grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

                
            with tf.GradientTape() as tape:
                gen_hr_muls = self.generator([lr_pans, lr_muls])
                fake_pred = self.discriminator(gen_hr_muls)
                real_pred = self.discriminator(hr_muls)
                gen_adversarial_loss = self.generator_adversarial_loss(real_pred, fake_pred)
                gen_l1_loss = gen_l1_loss_fn(hr_muls, gen_hr_muls)
                gen_loss = gen_l1_loss + self.scale * gen_adversarial_loss

            # print(gen_tv_loss)
            grads = tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

            return disc_loss, gen_loss, gen_adversarial_loss
        
        # check_point
        if self.start_epoch != 0:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from checkpoint.")

        for epoch in range(self.start_epoch+1, self.epoch):
            MAX_NUM = 10000
            giver = ImageGiver(pan_path = self.pan_path, multi_path=self.mul_path)
            
            for iteration in range(MAX_NUM):
                flag, hr_pans, hr_muls, lr_muls, lr_pans = giver.give(self.batchsize)
                if flag == 1:
                    break
                disc_loss, gen_loss, gen_ad_loss= train_step(lr_pans, lr_muls, hr_muls)
                if iteration % 5 == 0:
                    print("epoch:{}, iteration:{}:gen_loss:{},gen_ad_loss:{}, disc_loss:{}".format(epoch, iteration, gen_loss, gen_ad_loss, disc_loss))
                
                if iteration % 100 == 0:
                  self.make_sample(epoch, iteration)
            
            if epoch != 0:
                if os.path.exists("/content/drive/My Drive/lsrgan/newnet/esrgan_weights") == False:
                  os.mkdir("/content/drive/My Drive/lsrgan/newnet/esrgan_weights")
                self.generator.save_weights("/content/drive/My Drive/lsrgan/newnet/esrgan_weights/{}_gen.h5".format(
                    epoch
                ))
                self.discriminator.save_weights("/content/drive/My Drive/lsrgan/newnet/esrgan_weights/{}_disc.h5".format(
                    epoch
                ))
                ckpt_manager.save()                

def main():
    elsrgan = LSRGAN()
    elsrgan.train()

if __name__ == "__main__":
    main()



