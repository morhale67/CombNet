import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class combNet(nn.Module):
    def __init__(self, n_gray_levels, pic_width, m_patterns, input_shape=(64, 64, 1), classes=2, kernel_size=3,
                 filter_depth=(64, 128, 256, 512, 0)):
        super(combNet, self).__init__()
        self.n_gray_levels = n_gray_levels
        self.m_patterns = m_patterns
        self.ims = pic_width  # must be even number
        self.input_shape = input_shape
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        self.classes = classes
        self.c_i7 = 768

        # wangs net

        # first
        hidden_1, hidden_2 = self.ims ** 2, (2 * self.ims) ** 2
        self.int_fc1 = nn.Linear(self.m_patterns, hidden_1)
        self.int_fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout = nn.Dropout(0.9)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, int(0.5 * self.ims), 3, padding='same'),
            nn.BatchNorm2d(int(0.5 * self.ims)))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(int(0.5 * self.ims), self.ims, 3, padding='same'),
            nn.BatchNorm2d(self.ims))

        # fork
        self.conv_res = nn.Sequential(
            nn.Conv2d(self.ims, self.ims, 3, padding='same'),
            nn.BatchNorm2d(self.ims))

        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        self.maxpool8 = nn.MaxPool2d(8)
        self.upsample = nn.Upsample(scale_factor=2)

        # final
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(4 * self.ims, 2 * self.ims, 3, padding='same'),
            nn.BatchNorm2d(2 * self.ims))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(2 * self.ims, 2 * self.ims, 3, padding='same'),
            nn.BatchNorm2d(2 * self.ims))

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(2 * self.ims, self.ims, 3, padding='same'),
            nn.BatchNorm2d(self.ims))

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.ims, 1, 3, padding='same'),
            nn.BatchNorm2d(1))

        # Xnet

        # layers
        self.conv_0 = nn.Sequential(
            nn.LazyConv2d(filter_depth[0], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[0]))
        self.conv_1 = nn.Sequential(
            nn.LazyConv2d(filter_depth[1], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[1]))
        self.conv_2 = nn.Sequential(
            nn.LazyConv2d(filter_depth[2], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[2]))
        self.conv_3 = nn.Sequential(
            nn.LazyConv2d(filter_depth[3], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[3]))
        self.conv_for56level = nn.Sequential(
            nn.Conv2d(filter_depth[3], filter_depth[3], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[3]))
        self.conv_for7level = nn.Sequential(
            nn.Conv2d(self.c_i7, filter_depth[1], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[1]))
        self.conv_for8level = nn.Sequential(
            nn.Conv2d(filter_depth[2], filter_depth[1], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[1]))
        self.conv_for11level = nn.Sequential(
            nn.Conv2d(filter_depth[3], filter_depth[3], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[3]))
        self.conv_for12level = nn.Sequential(
            nn.Conv2d(filter_depth[3], filter_depth[2], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[2]))
        self.conv_for13level = nn.Sequential(
            nn.Conv2d(filter_depth[3], filter_depth[1], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[1]))
        self.conv_for14level = nn.Sequential(
            nn.Conv2d(filter_depth[2], filter_depth[0], self.kernel_size, padding='same'),
            nn.BatchNorm2d(filter_depth[0]))
        self.conv_last_xnet = nn.LazyConv2d(classes, (1, 1), padding="valid")

        self.maxpool2x2 = nn.MaxPool2d((2, 2))
        # self.upsample = nn.Upsample(scale_factor=(2, 2))

    def forward(self, x):
        blur_im = self.WangsNet(x)
        x = self.XNet(blur_im)
        return x

    def WangsNet(self, x):
        x = self.first_block(x)
        x = self.fork_block(x)
        x = self.final_block(x)
        # x = self.n_gray_levels * x / 255  # change scale to n gray levels
        return x

    def first_block(self, x):
        batch_size, num_feats = x.shape
        self.int_fc1 = nn.Linear(num_feats, 32 * 32)
        x = F.relu(self.int_fc1(x))
        x = self.dropout(x)

        x = F.relu(self.int_fc2(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block1(x.view(-1, 1, 2 * self.ims, 2 * self.ims)))
        x = self.dropout(x)

        x = F.relu(self.conv_block2(x))
        x = self.dropout(x)

        return x

    def fork_block(self, x):
        # path 1
        x1 = self.res_block(x)

        # path 2
        x2 = self.maxpool2(x)
        x2 = self.res_block(x2)
        x2 = self.upsample(x2)

        # path 3
        x3 = self.maxpool4(x)
        x3 = self.res_block(x3)
        x3 = self.upsample(x3)
        x3 = self.upsample(x3)

        # path 4
        x4 = self.maxpool8(x)
        x4 = self.res_block(x4)
        x4 = self.upsample(x4)
        x4 = self.upsample(x4)
        x4 = self.upsample(x4)

        concat_x = torch.cat((x1, x2, x3, x4), 1)
        return concat_x

    def res_block(self, x):
        """ 4 blue res block, fit to all paths"""
        for _ in range(4):
            y = F.relu(self.conv_res(x))
            f_x = F.relu(self.conv_res(y))
            x = F.relu(x + f_x)
        return x

    def final_block(self, x):
        x = self.maxpool2(x)

        x = F.relu(self.conv_block3(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block4(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block5(x))
        x = self.dropout(x)

        x = F.relu(self.last_layer(x))
        return x

    def XNet(self, x):
        act1, act8, act9, act11 = self.encoder(x)
        x = self.decoder(act1, act8, act9, act11)
        return x

    def encoder(self, x):
        batch1 = self.conv_0(x.view(-1, 1, self.input_shape[0], self.input_shape[1]))
        act1 = F.relu(batch1)
        pool1 = self.maxpool2x2(act1)
        # 100x100
        batch2 = self.conv_1(pool1)
        act2 = F.relu(batch2)
        pool2 = self.maxpool2x2(act2)
        # 50x50
        batch3 = self.conv_2(pool2)
        act3 = F.relu(batch3)
        pool3 = self.maxpool2x2(act3)
        # 25x25
        # Flat
        batch4 = self.conv_3(pool3)
        act4 = F.relu(batch4)
        # 25x25
        batch5 = self.conv_for56level(act4)
        act5 = F.relu(batch5)
        # 25x25
        # Up
        up6 = self.upsample(act5)
        batch6 = self.conv_for56level(up6)
        act6 = F.relu(batch6)
        concat6 = torch.cat((act3, act6), 1)
        # 50x50

        up7 = self.upsample(concat6)
        batch7 = self.conv_for7level(up7)
        act7 = F.relu(batch7)
        concat7 = torch.cat((act2, act7), 1)
        # 100x100

        # Down
        batch8 = self.conv_for8level(concat7)
        act8 = F.relu(batch8)
        pool8 = self.maxpool2x2(act8)
        # 50x50

        batch9 = self.conv_2(pool8)
        act9 = F.relu(batch9)
        pool9 = self.maxpool2x2(act9)

        # 25x25

        # Flat
        batch10 = self.conv_3(pool9)
        act10 = F.relu(batch10)
        # 25x25

        batch11 = self.conv_for11level(act10)
        act11 = F.relu(batch11)

        return act1, act8, act9, act11


    def decoder(self, act1, act8, act9, act11):
        # 25x25
        up12 = self.upsample(act11)
        batch12 = self.conv_for12level(up12)
        act12 = F.relu(batch12)
        concat12 = torch.cat((act9, act12), 1)
        # 50x50

        up13 = self.upsample(concat12)
        batch13 = self.conv_for13level(up13)
        act13 = F.relu(batch13)
        concat13 = torch.cat((act8, act13), 1)
        # 100x100

        up14 = self.upsample(concat13)
        batch14 = self.conv_for14level(up14)
        act14 = F.relu(batch14)
        concat14 = torch.cat((act1, act14), 1)
        # 200x200

        conv15 = self.conv_last_xnet(concat14)
        reshape15 = conv15
        # reshape15 = conv15.view(self.input_shape[2], self.input_shape[0] * self.input_shape[1], self.classes)
        act15 = F.softmax(reshape15, dim=1)

        x = act15[:, 0]

        return x
