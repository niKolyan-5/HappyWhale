from torch.utils.data import DataLoader, Dataset

from PIL import Image
import numpy as np

import pandas as pd
from torchvision import transforms

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import imgaug.augmenters as iaa
import random

from src.utils.pytorch_transforms import get_transforms_compose

import os


class HappyWhaleDataset(Dataset):
    """ Class for train and validation datasets
    Attributes:
        root_path: Path to folder with train images
        df_path: The path to the csv file with information about the data
        sample_type: Dataset type (train, val)
        val_size: Validation dataset size
        random_state: random_state
        transform: Applied transformations for images
        imgaug_cutout: Augmentation CutOut
        """

    def __init__(self, root_path: str,  df_path: pd.DataFrame, sample_type: str, val_size: float, random_state: int,
                 transform: transforms.Compose, imgaug_cutout=False):

        self.df = pd.read_csv(df_path)

        self.root_path = root_path
        self.sample_type = sample_type
        self.val_size = val_size
        self.random_state = random_state
        self.transform = transform
        self.imgaug_cutout = imgaug_cutout

        self.get_transform()
        self.__make_trees(self.df)
        self.__split_data()


    def get_transform(self):
        self.transform = get_transforms_compose(self.transform)

    def __make_trees(self, df: pd.DataFrame):
        """ Method for separating paths to images by each class """

        self.individual_tree = {indiv_name: [] for indiv_name in df.individual_id.unique()}

        for idx, row in tqdm(df.iterrows()):
            self.individual_tree[row.individual_id].append(os.path.join(self.root_path, row.image))

        self.once_labels = [label for label in self.individual_tree if len(self.individual_tree[label]) == 1]


        ## We separately define those classes that contain only one example
        if self.val_size == 0:
            self.train_once = self.once_labels
            self.val_once = []
        else:
            try:
                self.train_once, self.val_once = train_test_split(self.once_labels, test_size=self.val_size,
                                                                  random_state=self.random_state)
            except:
                self.train_once, self.val_once = [], []
                print('No classes with only one example')

    def __split_data(self):
        """ Method for data separation training validation """

        self.images = []
        self.labels = []

        for individual_id, image_list in tqdm(self.individual_tree.items()): ## Итерируемся по всем классам

            ## Defining the training dataset
            if self.sample_type == 'train':
                if individual_id in self.train_once:
                    images = image_list

                elif individual_id in self.val_once:
                    images = []

                elif self.val_size != 0:
                    images, _ = train_test_split(image_list, test_size=self.val_size,
                                                 random_state=self.random_state)

                else:
                    images = image_list

            ## Defining the validation dataset
            elif self.sample_type == 'val':

                if individual_id in self.val_once:
                    images = image_list

                elif individual_id in self.train_once:
                    images = []

                else:
                    _, images = train_test_split(image_list, test_size=self.val_size,
                                                 random_state=self.random_state)

            self.labels.extend([individual_id for _ in images])
            self.images.extend(images)

        ## Creating a dictionary with matching class names to their numeric labels
        self.individual_id2label = {individual_id: idx for idx, individual_id
                                    in enumerate(list(set(self.labels)))}

        self.individual_id2label['new_individual'] = len(self.individual_id2label) ## Creating a separate label for an alternative class

    def change_val_labels(self, train_individual_list: list, train_mapping: dict):
        """ Method for changing the labels of the validation dataset
        Arguments:
            train_individual_list: List of labels from training
            train_mapping: dictionary with the correspondence of class names to their numerical labels from training
        """
        for idx in range(len(self.labels)):
            ## If the label was not found in the training, replace it with the label of an alternative class
            if self.labels[idx] not in train_individual_list:
                self.labels[idx] = 'new_individual'

        self.individual_id2label = train_mapping

    def __len__(self):
        return len(self.images)

    def __imgaug(self, image: Image, p=0.25):
        """ Method for augmentation CutOut """

        if random.random() < p:
            img_arr = np.array(image)
            img_arr = iaa.Cutout(nb_iterations=10, size=0.1).augment_image(img_arr)
            image = Image.fromarray(img_arr.astype('uint8'), 'RGB')

        return image

    def __getitem__(self, idx):
        """ Method for getting a dataset object by its index """

        img = Image.open(self.images[idx]).convert('RGB')
        label = self.individual_id2label[self.labels[idx]]

        if self.imgaug_cutout:
            img = self.__imgaug(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataset(dataset_name: str):
    """ Function for selecting a specific dataset
    Arguments:
        dataset_name: Dataset name
    """
    dataset_mapping = {'happy_whale_dataset': HappyWhaleDataset}

    return dataset_mapping[dataset_name]

def get_data_loader(dataset_name: str, dataset_parameters: dict, batch_size: int, num_workers: int,
                    shuffle: bool, train_labels: list=None, train_mappings: dict=None):
    """ Function for selecting a specific dataloader
    Arguments:
        dataset_name: Dataset name
        dataset_parameters: Dataset parameters
        batch_size: batch size
        num_workers: number od workers
        shuffle: Flag - mixing of data
    :returns
        dataset: Dataset
        dataloader: DataLoader
    """
    dataset = get_dataset(dataset_name)(**dataset_parameters)
    if (train_mappings is not None) and (train_labels is not None):
        dataset.change_val_labels(train_labels, train_mappings)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=shuffle)

    return dataloader, dataset

