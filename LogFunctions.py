import logging
import time
from datetime import datetime

def print_run_info_to_log(batch_size, pic_width, prc_patterns, n_gray_levels, m_patterns, initial_lr, div_factor_lr,
                          num_dif_lr, n_epochs, num_train_samples, num_test_samples, folder_path='Logs/'):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    log_name = f"Log_{dt_string}.log"
    print(f'Name of log file: {log_name}')
    log_path = folder_path +'/'+ log_name
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f"This is a summery of the run:")
    logger.info(f'Batch size for this run: {batch_size}')
    logger.info(f'Size of original image: {pic_width}x{pic_width}')
    logger.info(f'Number of patterns: {m_patterns} which is {prc_patterns}% of {pic_width}^2')
    logger.info(f'Number of gray levels in output image: {n_gray_levels}')
    logger.info(
        f"Initial lr: {initial_lr}, Division factor: {div_factor_lr}, {num_dif_lr} divisions with {n_epochs} epochs for each")
    print_and_log_message(f"Number of samples in train is {num_train_samples}", log_path)
    print_and_log_message(f"Number of samples in train is {num_test_samples}", log_path)
    logger.info('***************************************************************************\n\n')
    return log_path

def print_and_log_message(message, log_path):
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    print(message)
    logger.warning(message)


