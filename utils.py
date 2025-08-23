import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

# encoder 
def BFRB_Encoder(gesture):
    bfrb = {'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch', 'Eyebrow - pull hair', 
    'Eyelash - pull hair', 'Neck - pinch skin', 'Neck - scratch', 'Cheek - pinch skin'}
    non_bfrb = {'Drink from bottle/cup', 'Glasses on/off', 'Pull air toward your face', 'Pinch knee/leg skin', 
    'Scratch knee/leg skin', 'Write name on leg', 'Text on phone', 'Feel around in tray and pull out an object', 'Write name in air', 'Wave hello'}

    labels = [] 
    for g in gesture:
        if g in bfrb:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels) 

# Dataset loader
class MultiInputDataset(Dataset):
    def __init__(self, x_a, x_r, x_t, x_dem, x_tof, y):
        self.x_a = x_a 
        self.x_r = x_r 
        self.x_t = x_t 
        self.x_dem = x_dem 
        self.x_tof = x_tof 
        self.labels = y 

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'x_a': torch.tensor(self.x_a[idx], dtype=torch.float32),
            'x_r': torch.tensor(self.x_r[idx], dtype=torch.float32),
            'x_t': torch.tensor(self.x_t[idx], dtype=torch.float32),
            'x_dem': torch.tensor(self.x_dem[idx], dtype=torch.float32),
            'x_tof': torch.tensor(self.x_tof[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

