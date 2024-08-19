import numpy as np

def moving_average(data):
    window_size = 200
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
