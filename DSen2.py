"""
Implementation of DSen2
paper address: https://arxiv.org/abs/1803.04271
Respect to the author for such a good paper!
"""

import tensorflow as tf
from tensorflow.keras import layers
from model import DSen2ModelMixin
from imagegiver import  ImageGiver
from makesample import MakeSampleMixin
import os
from tensorflow.keras.utils import plot_model


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True) # only one gpu i have

class DSen2(DSen2ModelMixin, MakeSampleMixin):
    def __init__(self, pan_path = "./pan", mul_path= "./mul"):
        self.pan_path = pan_path
        self.mul_path = mul_path
        self.pan_hr_shape = (256, 256, 1)
        self.pan_lr_shape = (128, 128, 1)
        self.mul_hr_shape = (128, 128, 7)
        self.mul_lr_shape = (64, 64, 7)
        self.residual_block_num = 8
        self.batchsize = 4
        self.epoch = 30
        self.start_epoch = 0
        self.model = self.build_DSen2()
        self.model_optimizer = tf.keras.optimizers.Adam(0.001, clipnorm = 1)
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()


    def train(self):
        if self.start_epoch != 0:
            weights_path = "./DSen_weights/" + str(self.start_epoch) + "_w.h5"
            self.model.load_weights(weights_path)
        
        @tf.function
        def train_step(lr_pans, lr_muls, hr_muls):
            ckpt = tf.train.Checkpoint(opt = self.model_optimizer,
                                    model = self.model)
            ckpt_manager = tf.train.CheckpointManager(ckpt, './tf_dsen2_ckpts', max_to_keep = 5) 
            with tf.GradientTape() as tape:
                out_hr_muls = self.model([lr_pans, lr_muls])
                loss = self.loss_fn(hr_muls, out_hr_muls)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return loss

        for epoch in range(self.start_epoch, self.epoch):
            MAX_NUM = 10000
            giver = ImageGiver()
            for iteration in range(MAX_NUM):
                flag, hr_pans, hr_muls, lr_muls, lr_pans = giver.give(self.batchsize)
                if flag == 1:
                    break
                loss = train_step(lr_pans, lr_muls, hr_muls)
                if iteration % 5 == 0:
                    print("epoch:{}, iteration:{}, loss:{}".format(epoch, iteration, loss))
                if iteration % 100 == 0:
                    self.make_sample(epoch, iteration, "./DSen_sample")

            if epoch != 0 :
                if os.path.exists("./DSen_weights") == False:
                    os.mkdir("./DSen_weights")
                self.model.save_weights("./DSen_weights/{}_w.h5".format(epoch))

def main():
    dsen2 = DSen2()
    dsen2.train()

if __name__ == "__main__":
    main()


