import numpy as np
import pickle
import os
import torch

class MNIST:

    def __init__(self, root, split="l_train"):
        self.dataset = self.load_pickle(root,split)

    def load_pickle(self,root,split):
        with open(os.path.join(root, "mnist",f'{split}.pkl'),'rb') as pkl_file:
            data = pickle.load(file=pkl_file) 
            for k,v in data.items():
                if k == 'images':
                    data[k]=torch.from_numpy(v).float()
                else:
                    data[k]=torch.from_numpy(v).long() 
        return data

    def __getitem__(self, idx):

        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]

        if "labels2" in self.dataset:
            label2 = self.dataset["labels2"][idx]
            return image, label, label2
        else:
            return image, label


    def __len__(self):
        return len(self.dataset["images"])