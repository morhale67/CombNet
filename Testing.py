import torch
from LogFunctions import print_and_log_message
from sup_functions import discretize
from sup_functions import PSNR
import os
import matplotlib.pyplot as plt
import _pickle as cPickle


def test_net(model, test_loader, criterion, batch_size, pic_width, log_path, n_gray_levels, folder_path,
             name_sub_folder='test_images'):
    test_loss, sum_psnr = 0.0, 0.0
    num_samples = len(test_loader.dataset)
    model.eval()
    for x_data, y_label in test_loader:
        output, test_loss, sum_psnr = test_batch(model, x_data, y_label, pic_width, test_loss, sum_psnr,
                                         n_gray_levels, criterion)
    avg_psnr, train_loss = avg_calc(sum_psnr, test_loss, num_samples)
    print_testing_messages(test_loss, num_samples, avg_psnr, log_path)
    save_outputs(output, y_label, pic_width, folder_path, name_sub_folder)

def print_testing_messages(test_loss, num_samples, avg_psnr, log_path):
    print_and_log_message(f'Avarge PSNR for {num_samples} images in test set is {avg_psnr}', log_path)
    print_and_log_message('Test Loss: {:.6f}\n'.format(test_loss), log_path)


def save_outputs(output, y_label, pic_width, folder_path, name_sub_folder):
    in_out_images = zip(output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    images_dir = folder_path + '/' + name_sub_folder
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for i, (out_image, orig_image) in enumerate(in_out_images):
        plt.imsave(images_dir + f'/test_image_{i}_out.jpg', out_image.detach().numpy())
        plt.imsave(images_dir + f'/test_image_{i}_orig.jpg', orig_image.cpu().detach().numpy())
        if i > 18:
            break


def test_batch(model, x_data, y_label, pic_width, test_loss, sum_psnr, n_gray_levels, criterion):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_data, y_label = x_data.to(dev), y_label.to(dev)
    output = model(x_data)
    disc_output = discretize(output, n_gray_levels)
    loss = criterion(output.view(-1, 1, pic_width, pic_width),
                     y_label.view(-1, 1, pic_width, pic_width))  # calculate the loss
    test_loss += loss.item() * x_data.size(0)  # update test loss

    in_out_images = zip(disc_output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    for out_image, orig_image in in_out_images:
        sum_psnr += PSNR(out_image, orig_image, pic_width, pic_width, n_gray_levels)

    return output, test_loss, sum_psnr


def avg_calc(sum_psnr, train_loss, num_samples):
    avg_psnr = sum_psnr / num_samples
    train_loss = train_loss / num_samples
    return avg_psnr, train_loss
