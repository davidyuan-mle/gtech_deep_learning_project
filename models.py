import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MultiInputModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiInputModel, self).__init__()
        self.num_classes = 18

        # acceleration 
        self.linear_acc1 = nn.Linear(3, 256)
        self.batchnorm_acc1 = nn.BatchNorm1d(256)
        self.prelu_acc1 = nn.PReLU()
        self.linear_acc2 = nn.Linear(256, 256)
        self.batchnorm_acc2 = nn.BatchNorm1d(256)
        self.prelu_acc2 = nn.PReLU()
        
        # rotation 
        self.linear_rot1 = nn.Linear(4, 256)
        self.batchnorm_rot1 = nn.BatchNorm1d(256)
        self.prelu_rot1 = nn.PReLU()
        self.linear_rot2 = nn.Linear(256, 256)
        self.batchnorm_rot2 = nn.BatchNorm1d(256)
        self.prelu_rot2 = nn.PReLU()

        # temperature 
        self.linear_temp1 = nn.Linear(5, 256)
        self.batchnorm_temp1 = nn.BatchNorm1d(256)
        self.prelu_temp1 = nn.PReLU()
        self.linear_temp2 = nn.Linear(256, 256)
        self.batchnorm_temp2 = nn.BatchNorm1d(256)

        # dem 
        self.linear_dem1 = nn.Linear(7, 256)
        self.batchnorm_dem1 = nn.BatchNorm1d(256)
        self.prelu_dem1 = nn.PReLU()
        self.linear_dem2 = nn.Linear(256, 256)
        self.batchnorm_dem2 = nn.BatchNorm1d(256)
        self.prelu_dem2 = nn.PReLU()

        # tof 
        # conv layers
        # input size is 8 x 8 with 5 channels 
        self.conv1 = nn.Conv2d(5, 64, kernel_size=2, stride=1, padding=0) # 7 x 7 
        self.batchnorm_conv1 = nn.BatchNorm2d(64)
        self.prelu_conv1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0) # 6 x 6 
        self.batchnorm_conv2 = nn.BatchNorm2d(64)
        self.prelu_conv2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0) # 5 x 5 
        self.batchnorm_conv3 = nn.BatchNorm2d(128)
        self.prelu_conv3 = nn.PReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0) # 4 x 4 
        self.batchnorm_conv4 = nn.BatchNorm2d(128)
        self.prelu_conv4 = nn.PReLU()
        self.conv5 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0) # 3 x 3 
        self.batchnorm_conv5 = nn.BatchNorm2d(256)
        self.prelu_conv5 = nn.PReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0) # 2 x 2 
        self.batchnorm_conv6 = nn.BatchNorm2d(256) 
        self.prelu_conv6 = nn.PReLU()

        # flatten 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 1024)
        self.batchnorm_fc1 = nn.BatchNorm1d(1024)
        self.prelu_fc1 = nn.PReLU()

        # mixing layers
        self.linear_mix1 = nn.Linear(2048, 2048)
        self.batchnorm_mix1 = nn.BatchNorm1d(2048)
        self.prelu_mix1 = nn.PReLU()
        self.linear_mix2 = nn.Linear(2048, 1024)
        self.batchnorm_mix2 = nn.BatchNorm1d(1024)
        self.prelu_mix2 = nn.PReLU()
        self.linear_mix3 = nn.Linear(1024, 512)
        self.batchnorm_mix3 = nn.BatchNorm1d(512)
        self.prelu_mix3 = nn.PReLU()
        self.linear_mix4 = nn.Linear(512, 256)
        self.batchnorm_mix4 = nn.BatchNorm1d(256)
        self.prelu_mix4 = nn.PReLU()

        # output layer 
        self.linear_out = nn.Linear(256, self.num_classes)
        
    def forward_non_xof(self, x_a, x_r, x_t, x_dem):
        # acceleration 
        x_a = self.linear_acc1(x_a)
        x_a = self.batchnorm_acc1(x_a)
        x_a = self.prelu_acc1(x_a)
        x_a = self.linear_acc2(x_a)
        x_a = self.batchnorm_acc2(x_a)
        x_a = self.prelu_acc2(x_a)

        # rotation 
        x_r = self.linear_rot1(x_r)
        x_r = self.batchnorm_rot1(x_r)
        x_r = self.prelu_rot1(x_r)
        x_r = self.linear_rot2(x_r)
        x_r = self.batchnorm_rot2(x_r)
        x_r = self.prelu_rot2(x_r)

        # temperature 
        x_t = self.linear_temp1(x_t)
        x_t = self.batchnorm_temp1(x_t)
        x_t = self.prelu_temp1(x_t)
        x_t = self.linear_temp2(x_t)
        x_t = self.batchnorm_temp2(x_t)

        # dem
        x_dem = self.linear_dem1(x_dem)
        x_dem = self.batchnorm_dem1(x_dem)
        x_dem = self.prelu_dem1(x_dem)
        x_dem = self.linear_dem2(x_dem)
        x_dem = self.batchnorm_dem2(x_dem)
        x_dem = self.prelu_dem2(x_dem)

        # concat 
        x = torch.cat((x_a, x_r, x_t, x_dem), dim=1)

        return x 
    

    def forward_xof(self, x_tof):
        # conv layers 
        x_tof = self.conv1(x_tof)
        x_tof = self.batchnorm_conv1(x_tof)
        x_tof = self.prelu_conv1(x_tof)
        x_tof = self.conv2(x_tof)
        x_tof = self.batchnorm_conv2(x_tof)
        x_tof = self.prelu_conv2(x_tof)
        x_tof = self.conv3(x_tof)
        x_tof = self.batchnorm_conv3(x_tof)
        x_tof = self.prelu_conv3(x_tof)
        x_tof = self.conv4(x_tof)
        x_tof = self.batchnorm_conv4(x_tof)
        x_tof = self.prelu_conv4(x_tof)
        x_tof = self.conv5(x_tof)
        x_tof = self.batchnorm_conv5(x_tof)
        x_tof = self.prelu_conv5(x_tof)
        x_tof = self.conv6(x_tof)
        x_tof = self.batchnorm_conv6(x_tof)
        x_tof = self.prelu_conv6(x_tof)

        # flatten 
        x_tof = self.flatten(x_tof)

        # linear layers 
        x_tof = self.fc1(x_tof)
        x_tof = self.batchnorm_fc1(x_tof)
        x_tof = self.prelu_fc1(x_tof)

        return x_tof 

    def forward(self, x_a, x_r, x_t, x_dem, x_tof):
        x_non_xof = self.forward_non_xof(x_a, x_r, x_t, x_dem)
        x_xof = self.forward_xof(x_tof)

        # concat 
        x = torch.cat((x_non_xof, x_xof), dim=1)

        # mixing layers 
        x = self.linear_mix1(x)
        x = self.batchnorm_mix1(x)
        x = self.prelu_mix1(x)
        x = self.linear_mix2(x)
        x = self.batchnorm_mix2(x)
        x = self.prelu_mix2(x)
        x = self.linear_mix3(x)
        x = self.batchnorm_mix3(x)
        x = self.prelu_mix3(x)
        x = self.linear_mix4(x)
        x = self.batchnorm_mix4(x)
        x = self.prelu_mix4(x)

        # output layer 
        x = self.linear_out(x)

        return x 
