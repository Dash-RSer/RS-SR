import tensorflow as tf
import numpy as np

def hpf_loss(image, label):
    v = 1/9
    kernel_channel = [[v],]