import torch
from LogFunctions import print_and_log_message
from sup_functions import discretize
from sup_functions import PSNR
import time
import os
import matplotlib.pyplot as plt
import _pickle as cPickle


def train_by_parameters(initial_lr, num_dif_lr, div_factor_lr,  train_loader, criterion, batch_size, n_epochs, pic_width,
                        log_path, n_gray_levels, folder_path, model):
    lr_i = initial_lr
    for i in range(num_dif_lr):
        print_and_log_message(f'learning rate: {lr_i}', log_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_i)
        model, optimizer = train_net(model, train_loader, criterion, optimizer, batch_size, n_epochs, pic_width,
                                         log_path, n_gray_levels,folder_path,  name_sub_folder='train_images')
        lr_i = lr_i / div_factor_lr
    return model


def train_net(model, train_loader, criterion, optimizer, batch_size, n_epochs, pic_width, log_path, n_gray_levels,
              folder_path, name_sub_folder, traning=1):
    """ train the network by the model.
        n_epochs - number of times the NN see all the train data """
    model.train()
    start = time.time()
    num_samples = len(train_loader.dataset)
    loss_func, psnr_func = [], []
    for epoch in range(n_epochs):
        train_loss, sum_psnr = 0.0, 0.0
        for x_data, y_label in train_loader:
            output, optimizer, loss_func, sum_psnr, psnr_func = train_batch(model, x_data, y_label, optimizer, pic_width, loss_func, train_loss,
                                                                    sum_psnr, psnr_func, n_gray_levels,criterion, traning)
        avg_psnr, train_loss = avg_calc(sum_psnr, train_loss, num_samples)
        print_training_messages(epoch, train_loss, avg_psnr, start, log_path)
        save_outputs(output, y_label, pic_width, folder_path, name_sub_folder, loss_func, psnr_func)
        start = time.time()
    return model, optimizer


def print_training_messages(epoch, train_loss, avg_psnr, start, log_path):
    end = time.time()
    print_and_log_message('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss), log_path)
    print_and_log_message(f"Time for epoch {epoch + 1} : {round(end - start)} sec", log_path)
    print_and_log_message(f'Wangs average PSNR for epoch {epoch + 1} on training set is {avg_psnr}', log_path)


def save_output_images_vs_original(output, y_label, pic_width, folder_path, name_sub_folder):
    in_out_images = zip(output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    images_dir = folder_path + '/' + name_sub_folder
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for i, (out_image, orig_image) in enumerate(in_out_images):
        # fig, ax = plt.subplots(2)
        # ax[0].imshow(out_image.detach().numpy())
        # ax[1].imshow(orig_image.cpu().detach().numpy())
        plt.imsave(images_dir + f'/train_image_{i}_out.jpg', out_image.detach().numpy())
        plt.imsave(images_dir + f'/train_image_{i}_orig.jpg', orig_image.cpu().detach().numpy())
        if i > 18:
            break


def save_psnr_and_Loss_functions(loss_func, psnr_func, folder_path):
    with open(folder_path + '/Loss_training', "wb") as output_file:
        cPickle.dump(loss_func, output_file)
    with open(folder_path + '/PSNR_training', "wb") as output_file:
        cPickle.dump(psnr_func, output_file)


def save_outputs(output, y_label, pic_width, folder_path, name_sub_folder, loss_func, psnr_func):
    save_output_images_vs_original(output, y_label, pic_width, folder_path, name_sub_folder)
    save_psnr_and_Loss_functions(loss_func, psnr_func, folder_path)


def train_batch(model, x_data, y_label, optimizer, pic_width, loss_func, train_loss, sum_psnr, psnr_func, n_gray_levels, criterion, traning):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_data, y_label = x_data.to(dev), y_label.to(dev)
    optimizer.zero_grad()  # clear the gradients of all optimized variables
    output = model(x_data)  # forward pass: compute predictions
    disc_output = discretize(output, n_gray_levels)
    loss = criterion(output, y_label.view(-1, 1, pic_width, pic_width))  # calculate the loss
    if traning:
        loss.backward()  # backward pass: compute gradient of the loss
        optimizer.step()  # parameter update - perform a single optimization step
    train_loss += loss.item() * x_data.size(0)  # update running training loss
    loss_func.append(loss.item() * x_data.size(0))

    in_out_images = zip(disc_output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    temp_sum_psnr = 0
    for out_image, orig_image in in_out_images:
        temp_sum_psnr += PSNR(out_image, orig_image, pic_width, pic_width, n_gray_levels)
    psnr_func.append(temp_sum_psnr / x_data.size(0))  # per batch size in epoch
    sum_psnr += temp_sum_psnr

    return output, optimizer, loss_func, sum_psnr, psnr_func


def avg_calc(sum_psnr, train_loss, num_samples):
    avg_psnr = sum_psnr / num_samples
    train_loss = train_loss / num_samples
    return avg_psnr, train_loss
