import torch
import torch.nn as nn
from Params import get_run_parameters
from DataFunctions import get_data
from DataFunctions import make_folder
from LogFunctions import print_run_info_to_log
from comb_model import combNet
from Training import train_by_parameters
from Testing import test_net
from os.path import exists

# Initialize Parameters
batch_size, pic_width, prc_patterns, n_gray_levels, m_patterns, initial_lr, div_factor_lr, num_dif_lr, n_epochs,\
                                num_train_samples, num_test_samples, input_shape, classes = get_run_parameters()
folder_path = make_folder('', num_train_samples, batch_size, n_gray_levels, prc_patterns)
log_path = print_run_info_to_log(batch_size, pic_width, prc_patterns, n_gray_levels, m_patterns, initial_lr,
                                 div_factor_lr, num_dif_lr, n_epochs, num_train_samples, num_test_samples)

# Get Data
generate = False
train_loader, test_loader, data_time_message, patterns = get_data(generate, log_path, batch_size, pic_width,
                                                                  prc_patterns, m_patterns)

# Define Model and Loss
model = combNet(n_gray_levels, pic_width, m_patterns, input_shape, classes)
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(dev)
criterion = nn.MSELoss()
model_path = folder_path + '/model.pth'

# Training
if exists(model_path) and not generate:
    model.load_state_dict(torch.load(model_path))  # load saved model
else:
    model = train_by_parameters(initial_lr, num_dif_lr, div_factor_lr, train_loader, criterion, batch_size, n_epochs,
                                pic_width, log_path, n_gray_levels, folder_path, model)
    torch.save(model.state_dict(), model_path) # Save Model

# Testing
test_net(model, test_loader, criterion, batch_size, pic_width, log_path, n_gray_levels, folder_path)


