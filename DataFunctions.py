import _pickle as cPickle
import time
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
import numpy as np
import torch


def get_data(generate, log_path, batch_size, pic_width, prc_patterns, m_patterns):
    train_set_file_name = f"train_loader_bs{batch_size}_pw{pic_width}_mp{prc_patterns}.pickle"
    test_set_file_name = f"test_loader_bs{batch_size}_pw{pic_width}_mp{prc_patterns}.pickle"

    if generate:
        start = time.time()
        train_data_minst, test_data_minst = load_data_mnist(batch_size, pic_width)
        train_loader, test_loader, patterns = generate_data(train_data_minst, test_data_minst, pic_width, m_patterns,
                                                            batch_size, patterns='new')
        save_generated_data(train_set_file_name, train_loader, test_set_file_name, test_loader)
        end = time.time()
        data_time_message = f"Generating the data took : {round(end - start)} sec"
    else:
        start = time.time()
        train_loader, test_loader = load_generated_data(train_set_file_name, test_set_file_name)
        end = time.time()
        data_time_message = f"Loading the data took : {round(end - start)} sec"
        patterns =[]

    return train_loader, test_loader, data_time_message, patterns


def save_generated_data(train_set_file_name, train_loader, test_set_file_name, test_loader):
    with open(train_set_file_name, "wb") as output_file:
        cPickle.dump(train_loader, output_file)
    with open(test_set_file_name, "wb") as output_file:
        cPickle.dump(test_loader, output_file)

def load_generated_data(train_set_file_name, test_set_file_name):
    with open(train_set_file_name, "rb") as input_file:
        train_loader = cPickle.load(input_file)
    with open(test_set_file_name, "rb") as input_file:
        test_loader = cPickle.load(input_file)
    return train_loader, test_loader

def load_data_mnist(batch_size, pic_width):
    transform = transforms.Compose([transforms.Resize((pic_width, pic_width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    lambda x: x > 0,
                                    lambda x: x.float()])  # convert data to torch.FloatTensor

    # choose the training and test datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # slice the data
    num_train_samples, num_test_samples = 640, 64
    train_data = Subset(train_data, np.arange(num_train_samples))
    test_data = Subset(test_data, np.arange(num_test_samples))

    pic_width = len(train_data[0][0][0])

    return train_data, test_data


def generate_data(train_data, test_data, pic_width, m_patterns, batch_size, patterns):
    """ Get set of images and return simulated detector data of the images"""
    if patterns == 'new':
        patterns = define_m_random_patterns(pic_width, m_patterns)
    train_detector_data, test_detector_data = create_detector_data(train_data, test_data, patterns, pic_width)

    patterns = define_m_random_patterns(pic_width, m_patterns)
    train_detector_data, test_detector_data = create_detector_data(train_data, test_data, patterns, pic_width)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_detector_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_detector_data, batch_size=batch_size)

    return train_loader, test_loader, patterns


def define_m_random_patterns(pic_width, m):
    """ define the light patterns"""
    patterns = [np.random.rand(pic_width, pic_width) for i in range(m)]
    return patterns

def create_detector_data(train_data, test_data, patterns, pic_width):
    """ create couples of [CGI_sample, original_image]"""
    train_detector_data = []
    for sample in train_data:
        image = sample[0].view(pic_width, pic_width)
        train_detector_data.append([torch.tensor(sample_after_patterns(image, patterns)), torch.flatten(image)])

    test_detector_data = []
    for sample in test_data:
        image = sample[0].view(pic_width, pic_width)
        test_detector_data.append([torch.tensor(sample_after_patterns(image, patterns)), torch.flatten(image)])

    return train_detector_data, test_detector_data

def sample_after_patterns(image, patterns):
    """ calculate CGI_sample from original_image"""
    detector_output = []
    for i, i_pattern in enumerate(patterns):
        image_after_mask = np.array(image) * i_pattern
        detector_output.append(np.float32(sum(sum(image_after_mask))))
    return detector_output

def make_folder(net_name, num_samp, batch_size, n_gray_levels, prc_patterns):
    folder_name = f'{net_name}_NumSamp_{num_samp}_bs_{batch_size}_graylev_{n_gray_levels}_prc{prc_patterns}'
    print(folder_name)
    folder_path = 'Results/' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

