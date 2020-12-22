import numpy as np
import cv2

def twopercentlinearstrech(image, max_out=255, min_out=0):
    # b, g, r = cv2.split(image)
    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value) 
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)
        return processed_gray
    # r_p = gray_process(r)
    # g_p = gray_process(g)
    # b_p = gray_process(b)
    # result = cv2.merge((b_p, g_p, r_p))
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    result = np.zeros((height, width, channels), dtype = np.float)
    for channel in range(channels):
        result[:, :, channel] = gray_process(image[:, :, channel])
    return np.array(np.uint8(result))

def graytwopercentlinearstrech(image, max_out =255, min_out = 0):
    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value) 
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)
        return processed_gray
    image = gray_process(image)
    return np.uint8(image)