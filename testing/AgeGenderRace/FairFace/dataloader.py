from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv2

class FairFaceDataset(Dataset):
    
    def __init__(self, train):
        if train == 0:
            xy = np.loadtxt('fairface_label_train.csv', delimiter=',', dtype=str, skiprows=1)
        else:
            xy = np.loadtxt('fairface_label_val.csv', delimiter=',', dtype=str, skiprows=1)
        self.path = xy[:, [0]]
        self.age = xy[:, [1]]
        self.gender = xy[:, [2]]
        self.race = xy[:, [3]]
        self.n_samples = xy.shape[0]
        
        self.age_mapping = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }
        
        self.gender_mapping = {
            'Male': 0,
            'Female': 1
        }
        
        self.race_mapping = {
            'White': 0,
            'Black': 1,
            'Latino_Hispanic': 2,
            'East Asian': 3,
            'Southeast Asian': 4,
            'Indian': 5,
            'Middle Eastern': 6
        }
        
    def __getitem__(self, index):
        path = self.path[index][0]
        age = self.age_mapping[self.age[index][0]]
        gender = self.gender_mapping[self.gender[index][0]]
        race = self.race_mapping[self.race[index][0]]
        return path, age, gender, race
    
    def __len__(self):
        return self.n_samples

# # Create dataset and dataloaders
# train_dataloader = DataLoader(FairFaceDataset(0), batch_size=64, shuffle=True)
# val_dataloader = DataLoader(FairFaceDataset(1), batch_size=64, shuffle=True)

# # Check the first batch from the train_dataloader
# for batch in train_dataloader:
#     paths, ages, genders, races = batch
#     print('-----------------------------------')
#     print('Paths:', paths)
#     print('Ages:', ages)
#     print('Genders:', genders)
#     print('Races:', races)
#     print('-----------------------------------')
#     break  # Just check the first batch

# # Check the first batch from the val_dataloader
# for batch in val_dataloader:
#     paths, ages, genders, races = batch
#     print('-----------------------------------')
#     print('Paths:', paths)
#     print('Ages:', ages)
#     print('Genders:', genders)
#     print('Races:', races)
#     print('-----------------------------------')
#     break  # Just check the first batch

# print(FairFaceDataset(0).__len__()/64)