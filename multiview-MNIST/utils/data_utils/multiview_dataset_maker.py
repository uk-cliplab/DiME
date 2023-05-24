import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from .custom_transforms import *
import random
import numpy as np
import os

class rot_noisy_multiview_dataset(Dataset):
    """
    Implementation of a multiview dataset. Currently geared for View 1 rotated + View 2 noisy.

    By default, if dataset_id='rot_noisy', returns a multiview dataset of MNIST where View 1
    is rotated by an angle from [-45, 45] degrees and View 2 applies uniform noise from [0, 1].

    Inputs:
        raw_data - raw data to create from. optional if dataset is saved
        dataset_id - name of folder to which the multiview dataset will be stored/loaded from
        is_testset - sets whether this is a testset or no
        check_for_saved - defaults True. If true, will search if dataset set with dataset_id already exists. If so load.
        use_triplet_loss - Unused so far

        rotation_transform - Custom View 1 transform
        noisy_transform - Custom View 2 transform

        rot_angle - Default to 45. If using default rotation transform, applies this angle to sample from [-rot_angle, rot_angle]
        noise_max - Defaults to 1. If using default noisy transform, sample noisy from [0, noise_max].

    Outputs:
        Multiview dataset that can be given to a Dataloader.

        When iterating through dataloader, each batch is of the form ((View 1 batch, View 2 batch),  labels)
    """

    def __init__(self, raw_data=None, rot_angle=45, noise_max=1, use_triplet_loss=False, rotation_transform=None, noisy_transform=None, check_for_saved=True, is_testset=False, dataset_id='rot_noisy', shuffle_view2=True):
        # setup config variables
        self.use_triplet_loss = use_triplet_loss
        self.rot_angle = rot_angle
        self.noise_max = noise_max
        self.raw_data = raw_data
        self.is_testset = is_testset
        self.custom_labels = None
        self.shuffle_view2 = shuffle_view2

        # default View 1 / View 2 transforms
        self.rotation_transform = rotation_transform if rotation_transform is not None else transforms.Compose([
            transforms.RandomRotation(degrees=self.rot_angle, fill=0, interpolation=transforms.InterpolationMode.NEAREST)
        ])
        
        self.noisy_transform = noisy_transform if noisy_transform is not None else transforms.Compose([
            AddUniformNoise(noise_min=0,  noise_max=self.noise_max),
            Clamp(min_val=0, max_val=1)
        ])
        
        # setup folder locations
        if self.is_testset:
            self.label_path = "data/%s/labels_test.npy" % dataset_id
            self.data_path = "data/%s/paired_imgs_test.npy" % dataset_id
        else:
            self.label_path = "data/%s/labels.npy" % dataset_id
            self.data_path = "data/%s/paired_imgs.npy" %dataset_id
        
        # check if data exists and load if allowed
        if check_for_saved and os.path.exists(self.label_path) and os.path.exists(self.data_path):
            self.data, self.labels = self.load_dataset()
        
        # create multiview dataset from scratch
        else:
            if not os.path.exists("data/%s" % dataset_id):
                os.mkdir("data/%s" % dataset_id)
    
            if self.raw_data is None:
                if dataset_id == 'rot_noisy':
                   self.raw_data = self.get_rot_noisy_dataset()
                elif dataset_id == 'cifar10':
                    self.raw_data = self.get_cifar10_dataset()
                else:
                    raise ValueError("Must supply raw dataset because %s does not exist..." % dataset_id)
            self.paired_indices = self.constructIndexPairings()

            self.data, self.labels = self.constructData()
            self.save_dataset()
            
    def constructIndexPairings(self):
        """
        Shuffles View 2 indices so that pairwise View 1 and View 2 belong to same class, but are different instances

        Outputs:
            paired_indices - 2d array of size (N, 2) s.t. above holds. column 0 corresponds to view 1, column 1 to view 2
        """
        if (torch.is_tensor(self.raw_data.targets)):
            targets = np.array([i.item() for i in self.raw_data.targets])
        else:
            targets = np.array([i for i in self.raw_data.targets])

        label_set = set(list(targets))
        
        paired_indices = []
        for lbl in label_set:
            orig_lbl_indices = np.where(targets == lbl)[0]

            shuffled_lbl_indices = np.random.permutation(orig_lbl_indices) if self.shuffle_view2 else orig_lbl_indices

            paired_indices.extend(list(zip(list(orig_lbl_indices), list(shuffled_lbl_indices))))
        
        # shuffle array for good measure
        random.shuffle(paired_indices)
    
        return paired_indices
    
    def constructData(self):
        """
        Applies transforms to data

        Outputs:
            List of data tuples s.t. each tuple is (rotated_instance, noisy_instance)
            1-D list of labels
        """
        num_data = len(self.raw_data)
        data_size = self.raw_data[0][0].shape
        
        data = np.zeros((num_data, 2, *data_size),dtype=np.float32)
        labels = np.zeros(num_data, dtype=int)
        
        for idx in range(num_data):
            rot_idx, noisy_idx = self.paired_indices[idx]
            raw_rot_instance, raw_rot_label = self.raw_data[rot_idx]
            raw_noisy_instance, raw_noisy_label = self.raw_data[noisy_idx]

            assert raw_rot_label == raw_noisy_label

            rotated_instance = self.rotation_transform(raw_rot_instance)
            noisy_instance = self.noisy_transform(raw_noisy_instance)
            
            data[idx, 0, :] = rotated_instance
            data[idx, 1, :] = noisy_instance
            labels[idx] = raw_rot_label
        
        return data, labels
        
    def __safe_print__(self, message):
        """
        Prints a message if class has show_progress set
        """
        if (self.show_progress):
            print(message)
        
    def __len__(self):
        """
        Overload of len() function in superclass

        How many data points are in this dataset
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Overload of getitem() function in superclass

        Gets data at position idx. Of the form (rotated_instance, noisy_instance), label
        """
        label = self.labels[idx]
        rotated_instance = torch.from_numpy(self.data[idx, 0, :])
        noisy_instance = torch.from_numpy(self.data[idx, 1, :])
  
        items = [rotated_instance, noisy_instance]
        return items, label
    
    def save_dataset(self):
        """
        Saves dataset to file.
        """
        assert self.paired_indices is not None
        
        np.save(self.data_path, self.data)
        np.save(self.label_path, self.labels)
        
    def load_dataset(self):
        """
        Load dataset from file
        """
        labels = np.load(self.label_path).astype(np.int)
        data = np.load(self.data_path).astype(np.float32)
        
        return data, labels

    def get_rot_noisy_dataset(self):
        """
        For rot_noisy MNIST, no need to supply raw_data to __init__. We can load it here
        """
        if not self.is_testset:
            return datasets.MNIST(root='data', 
                                   train=True, 
                                   transform=transforms.Compose(
                                       [transforms.ToTensor(),
                                     ]),
                                   download=True)
        else:
            return datasets.MNIST(root='data', 
                                        train=False, 
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                            ]),
                                        download=True)

    def get_custom_labels(self):
        return self.custom_labels if self.custom_labels is not None else self.labels

    def get_cifar10_dataset(self):
        """
        For rot_noisy MNIST, no need to supply raw_data to __init__. We can load it here
        """
        self.rotation_transform = transforms.Compose([])
        
        self.custom_labels = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        if not self.is_testset:
            return datasets.CIFAR10(root='data', 
                                   train=True, 
                                   transform=transforms.Compose(
                                       [transforms.ToTensor(),
                                     ]),
                                   download=True)
        else:
            return datasets.CIFAR10(root='data', 
                                        train=False, 
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                            ]),
                                        download=True)
        
        