import tensorflow as tf
from tensorflow.keras import layers
from layers import DenseBlock, ConvBlock, ResidualBlock 
   
class GeneratorMixin(object):    
    
    def build_generator(self):
        input_pan_lyr = tf.keras.Input(self.pan_lr_shape)
        input_mul_lyr = tf.keras.Input(self.mul_lr_shape)

        mul_upsample = tf.keras.layers.UpSampling2D((2, 2))(input_mul_lyr)
        concate = tf.keras.layers.Concatenate()([input_pan_lyr, mul_upsample])
        # pre_conv
        dense_x = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = "same" 
        )(concate)
        input_channels = dense_x.shape[-1]

        for _ in range(self.dense_block_num):
            dense_x = DenseBlock(input_channels, self.dense_growth_channels, 0.2)(dense_x)
        
        adjust_dim_lyr = tf.keras.layers.Conv2D(
            filters = self.mul_hr_shape[2],
            kernel_size = 3,
            strides = 1,
            padding = "same"
        )(dense_x)

        add_feature_lyr = tf.keras.layers.Add()([adjust_dim_lyr, mul_upsample])
        model = tf.keras.Model(
            inputs = [input_pan_lyr, input_mul_lyr],
            outputs = [add_feature_lyr]
        )

        return model

class DiscriminatorMixin(object):
    
    def build_discriminator(self):
        input_lyr = tf.keras.Input(self.mul_hr_shape)
        x = ConvBlock(64)(input_lyr)
        x = ConvBlock(64, 2)(x)
        x = ConvBlock(128)(x)
        x = ConvBlock(128, 2)(x)
        x = ConvBlock(256)(x)
        x = ConvBlock(256, 2)(x)
        x = ConvBlock(512)(x)
        x = ConvBlock(512, 2)(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        out_lyr = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(
            inputs = input_lyr,
            outputs = out_lyr
        )
        return model

class DSen2ModelMixin(object):
    
    def build_DSen2(self):
        input_pan_lyr = tf.keras.Input(self.pan_lr_shape)
        input_mul_lyr = tf.keras.Input(self.mul_lr_shape)

        up_sample_mul = layers.UpSampling2D(size = (2, 2))(input_mul_lyr)
        x = feature_concatenate = layers.Concatenate()([input_pan_lyr, up_sample_mul])
        # make the input for residual block
        # use 128 filters as same as paper
        x = layers.Conv2D(
            filters = 128,
            kernel_size = 3,
            strides = 1,
            padding = "same"
        )(x)
        for _ in range(self.residual_block_num):
            x = ResidualBlock()(x)
        x = layers.Conv2D(
            filters = self.mul_hr_shape[2],
            kernel_size = 1,
            strides = 1,
            padding = "same"
        )(x)
        output_lyr = layers.Add()([up_sample_mul, x])
        model = tf.keras.Model(
            inputs = [input_pan_lyr, input_mul_lyr],
            outputs = output_lyr
        )
        return model

