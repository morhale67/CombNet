import math
import torch


def discretize(my_tensor,n_gray_levels):
    ''' change scale to n gray levels '''
    tensor_0_to_1 = my_tensor / 255
    disc_tensor = (tensor_0_to_1 * n_gray_levels).round() # max is n_gray_levels
    return disc_tensor


def PSNR(image1, image2, m, n, max_i=255):
    # max_i is n_gray_levels
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    y = torch.add(image1.to(dev), (-image2).to(dev))
    y2 = torch.pow(y, 2)
    mse = torch.sum(y2)/(m*n)
    psnr = 10*math.log(max_i/mse, 10)
    return psnr