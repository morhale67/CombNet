def get_run_parameters():
    batch_size = 64
    pic_width = 32
    prc_patterns = 1
    n_gray_levels = 64
    m_patterns = (pic_width ** 2) * prc_patterns // 100

    initial_lr = 10 ** -2
    div_factor_lr = 1
    num_dif_lr = 4
    n_epochs = 1

    input_shape, classes = (pic_width, pic_width, batch_size), 2
    num_train_samples, num_test_samples = 640, 64

    return batch_size, pic_width, prc_patterns, n_gray_levels, m_patterns, initial_lr, div_factor_lr, num_dif_lr,\
           n_epochs, num_train_samples, num_test_samples, input_shape, classes