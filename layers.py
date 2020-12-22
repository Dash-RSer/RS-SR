import tensorflow as tf
from tensorflow.keras import layers

class DenseBlock(layers.Layer):
    """Densebock used in generator

    Argument:
        dense_growth_channels: how many channels grow after one
         conv, leakyrelu, concatenate
    """
    def __init__(self, input_channels, dense_growth_channels, scale):
        super(DenseBlock, self).__init__()
        self.input_channels = input_channels
        self.dense_growth_channels = dense_growth_channels
        self.scale = scale

        self.conv_1 = tf.keras.layers.Conv2D(
            filters = self.dense_growth_channels, 
            kernel_size = 3, 
            strides = 1,
            padding = "same"
        )
        self.concatenate_1 = tf.keras.layers.Concatenate()
        self.conv_2 = tf.keras.layers.Conv2D(
            filters = self.dense_growth_channels, 
            kernel_size = 3, 
            strides = 1,
            padding = "same"
        )
        self.concatenate_2 = tf.keras.layers.Concatenate()
        self.conv_3 = tf.keras.layers.Conv2D(
            filters = self.dense_growth_channels,
            kernel_size = 3,
            strides = 1, 
            padding = "same"
        )
        self.concatenate_3 = tf.keras.layers.Concatenate()
        self.conv_4 = tf.keras.layers.Conv2D(
            filters = self.dense_growth_channels,
            kernel_size = 3,
            strides = 1, 
            padding = "same"            
        )
        self.concatenate_4 = tf.keras.layers.Concatenate()
        self.conv_5 = tf.keras.layers.Conv2D(
            filters = self.input_channels,
            kernel_size = 3,
            strides = 1, 
            padding = "same"   
        )
        self.residual_scale = tf.keras.layers.Lambda(lambda x: x * scale)
        self.add = tf.keras.layers.Add()
        
    
    def call(self, inputs):
        x_0 = inputs
        x = self.conv_1(inputs)
        x = tf.nn.leaky_relu(x)
        x = x_1 = self.concatenate_1([x_0, x])
        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x)
        x = x_2 = self.concatenate_2([x_1, x_0, x])
        x = self.conv_3(x)
        x = tf.nn.leaky_relu(x)
        x = x_3 = self.concatenate_3([x_2, x_1, x_0, x])
        x = self.conv_4(x)
        x = tf.nn.leaky_relu(x)
        x = x_4 = self.concatenate_4([x_3, x_2, x_1, x_0, x])
        x = self.conv_5(x)
        x = self.residual_scale(x)
        x = self.add([x, x_0])
        return x
        
class ConvBlock(tf.keras.layers.Layer):
    """Convolution layer used in discrimanator

    Argument:
        filters: num of filters used in conv.
        strides: strides.
        bn: bool, if use bn in conv block.

    """
    def __init__(self, filters, strides = 1, bn = True):
        super(ConvBlock, self).__init__()
        self. has_bn = True
        self.conv = tf.keras.layers.Conv2D(
            filters = filters,
            kernel_size = 3,
            strides = strides,
            padding = "same"
        )
        self.bn = tf.keras.layers.BatchNormalization(momentum = 0.8)

    def call(self, inputs):
        x = self.conv(inputs)
        x = tf.nn.leaky_relu(x)
        if self.has_bn == True: 
            x = self.bn(x)
        
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters = 128, **kwargs):
        """
        The residual block in DSen2.
        Same as paper at all.
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            filters = filters,
            kernel_size = 3,
            strides = 1,
            padding = "same"
        )
        self.conv2 = layers.Conv2D(
            filters = filters,
            kernel_size = 3,
            strides = 1,
            padding = "same"
        )
        # 0.1 is same as the paper
        self.residual_scale = layers.Lambda(lambda x: x * 0.1)
        self.add = layers.Add()

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.residual_scale(x)
        x = self.add([inputs, x])
        return x
