import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, Sampler

class LT_Dataset(Dataset):
    num_classes = 365

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 
    


class LT_Dataset_Eval(Dataset):
    num_classes = 365

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class Places_LT(DataLoader):
    def __init__(self,  data_dir="", batch_size=60, num_workers=40, training=True, train_txt = "./data_txt/Places_LT_v2/Places_LT_train.txt",
                 eval_txt = "./data_txt/Places_LT_v2/Places_LT_val.txt",
                 test_txt = "./data_txt/Places_LT_v2/Places_LT_test.txt"):
        self.num_workers = num_workers
        self.batch_size= batch_size
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        train_dataset = LT_Dataset(data_dir, train_txt, transform=transform_train)
        
        if training:
            dataset = LT_Dataset(data_dir, train_txt, transform=transform_train)
            eval_dataset = LT_Dataset_Eval(data_dir, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        else:
            dataset = LT_Dataset_Eval(data_dir, test_txt, transform=transform_test, class_map=train_dataset.class_map)
            eval_dataset = None
            
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        
        
        self.cls_num_list = train_dataset.cls_num_list
        """
        self.data = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        """
                
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers
        }
        
        #super().__init__(dataset=self.data)
        super().__init__(dataset=self.dataset, **self.init_kwargs)
        
        
    def split_validation(self):
    # If you do not want to validate:
    # return None
    # If you want to validate:\
         
        return DataLoader(dataset=self.eval_dataset, **self.init_kwargs)
    
