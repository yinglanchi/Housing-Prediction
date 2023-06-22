import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import os
import os.path as osp
from os.path import join as pjoin


FILE_DIR = osp.dirname(os.path.abspath(__file__))

DATA_DIR = pjoin(FILE_DIR, "data")
#FILENAME = "kc_house_data.csv"
FILENAME = "cleaned_data_all.csv"

RANDOM_SEED = 1234567
TRAIN_RATIO = 0.8
#VAL_RATIO = 0.1
TEST_RATIO = 1 - TRAIN_RATIO


class HousePriceDataset(Dataset):
    def __init__(self, mode, cfg):
        self.cfg = cfg
        self.data_dict = load_data()
        self.mode = mode
        
    def __len__(self):
        if self.mode == "train":
            return len(self.data_dict["train_labels"])
#         elif self.mode == "val":
#             return len(self.data_dict["val_labels"])
        elif self.mode == "test":
            return len(self.data_dict["test_labels"])
        
    def __getitem__(self, idx):
        data: dict = {}
        if self.mode == "train":
            data["features"] = self.data_dict["train_features"].iloc[idx].to_numpy()
            data["prices"] = self.data_dict["train_labels"].iloc[idx]
#         elif self.mode == "val":
#             data["features"] = self.data_dict["val_features"][idx]
#             data["prices"] = self.data_dict["val_labels"][idx]
        elif self.mode == "test":
            data["features"] = self.data_dict["test_features"].iloc[idx].to_numpy()
            data["prices"] = self.data_dict["test_labels"].iloc[idx]
            
        data["prices"] = np.expand_dims(data["prices"], axis=0)
        
        return data

def load_data():
    """
    :return: data_dict: dict
        "train_features": train_features: [N1, D], N is the number of samples, D is the dimension of features
        "train_labels": train_labels: [N1]
        "val_features": val_features: [N2, D]
        "val_labels": val_labels: [N2]
        "test_features": test_features: [N3, D]
        "test_labels": test_labels: [N3]
    """
    data = pd.read_csv(pjoin(DATA_DIR, FILENAME))
    
    X_data = data.drop('price',1)
    y_data = data['price']
    
#     # TODO: Deal with "date"
#     features = data.drop(columns=["price"]).to_numpy()
#     #features = data.drop(columns=["id", "date", "price", "yr_built", "yr_renovated", "zipcode", "lat", "long"]).to_numpy()
#     #features=data[["floors","waterfront","lat","bedrooms","sqft_basement","view","bathrooms",
#                    #"sqft_living15","sqft_above","grade","sqft_living"]].to_numpy()
#     #print(features)
#     labels = data["price"].to_numpy()
    
#     # randomly split data into train, val and test
#     np.random.seed(RANDOM_SEED)
#     indices = np.random.permutation(len(features))
#     train_indices = indices[:int(len(features) * TRAIN_RATIO)]
#     #val_indices = indices[int(len(features) * TRAIN_RATIO):int(len(features) * (TRAIN_RATIO + VAL_RATIO))]
#     test_indices = indices[int(len(features) * (TRAIN_RATIO)):]

#     train_features = features[train_indices]
#     train_labels = labels[train_indices]
# #     val_features = features[val_indices]
# #     val_labels = labels[val_indices]
#     test_features = features[test_indices]
#     test_labels = labels[test_indices]
    
    train_features, test_features, train_labels, test_labels = train_test_split(X_data, y_data, test_size=0.2, random_state=RANDOM_SEED)
    
    data_dict = {
        "train_features": train_features,
        "train_labels": train_labels,
#         "val_features": val_features,
#         "val_labels": val_labels,
        "test_features": test_features,
        "test_labels": test_labels
    }
    
    return data_dict


def get_dataloader(cfg, mode, shuffle=None):
    if shuffle is None:
        shuffle = (mode == "train")
        
    dataset = HousePriceDataset(mode, cfg)
    batch_size = cfg["batch_size"]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def test():
    data_dict = load_data()
    # print(data_dict["train_features"].shape)
    # print(data_dict["train_labels"].shape)
#     print(data_dict["val_features"].shape)
#     print(data_dict["val_labels"].shape)
    # print(data_dict["test_features"].shape)
    # print(data_dict["test_labels"].shape)
    
    print(data_dict["train_features"].iloc[0].to_numpy())
    
def main():
    test()

if __name__ == "__main__":
    main()
    