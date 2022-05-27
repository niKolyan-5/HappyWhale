import torch

from src.utils.data_loaders import get_data_loader
from src.models.models import get_model
from src.utils.optimizers import get_optimizer, get_scheduler
from src.utils.losses import get_loss_func

import os
from tqdm import tqdm
import numpy as np

from sklearn.neighbors import NearestCentroid
class HappyWhaleModelTrainer:
    """ Class for train HappyWhale models
    Attributes:
        model_config: model parameters
        data_config: data parameters
        train_config: train parameters
    """

    def __init__(self, model_config: dict, data_config: dict, train_config: dict):

        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        self.get_data_loaders()
        self.get_model()
        self.set_train_parameters()



    def get_data_loaders(self):
        """ Get DataLoaders"""

        ## Defining a training dataset
        self.train_loader, self.trainset = get_data_loader(dataset_name=self.data_config['train_data']['dataset_name'],
                                                        dataset_parameters=self.data_config['train_data']['dataset_params'],
                                                        batch_size=self.train_config['train_batch'],
                                                        num_workers=self.train_config['num_workers'],
                                                        shuffle=True)

        ## Defining a validation dataset
        self.val_loader, self.valset = get_data_loader(dataset_name=self.data_config['val_data']['dataset_name'],
                                                      dataset_parameters=self.data_config['val_data']['dataset_params'],
                                                      batch_size=self.train_config['val_batch'],
                                                      num_workers=self.train_config['num_workers'],
                                                      shuffle=False, train_labels=list(set(self.trainset.labels)),
                                                      train_mappings=self.trainset.individual_id2label)

        ## We define a training data set without additional augmentations
        self.train_loader_2, self.trainset_2 = get_data_loader(dataset_name=self.data_config['train_data_2']['dataset_name'],
                                                          dataset_parameters=self.data_config['train_data_2']['dataset_params'],
                                                          batch_size=self.train_config['val_batch'],
                                                          num_workers=self.train_config['num_workers'],
                                                          shuffle=False)


        ## Adding the parameter of the number of classes for the DolgNet model
        if self.model_config['model_name'] == 'dolgnet':
            self.model_config['model_parameters']['dolgnet_parameters']['num_of_classes'] = len(set(self.trainset.labels))



    def set_train_parameters(self):
        """ Method for determining training parameters """

        self.optimizer = get_optimizer(optim_name=self.train_config['optimizer']['name'], model=self.model.model,
                                       optim_parameters=self.train_config['optimizer']['parameters'])
        if self.train_config['criterion']['name'] == 'Focal':
            self.train_config['criterion']['parameters']['device'] = self.model.device
            self.train_config['criterion']['parameters']['class_num'] = len(set(self.trainset.labels))
        self.criterion = get_loss_func(loss_name=self.train_config['criterion']['name'],
                                       loss_params=self.train_config['criterion']['parameters'])

        ## Adding a schedule for learning_rate
        if 'scheduler' in self.train_config:
            if self.train_config['scheduler']['name'] == 'OneCycle':
                self.train_config['scheduler']['parameters']['steps_per_epoch'] = len(self.train_loader)
                self.train_config['scheduler']['parameters']['epochs'] = self.train_config['num_epochs']
            self.scheduler = get_scheduler(scheduler_name=self.train_config['scheduler']['name'],
                                           optimizer=self.optimizer,
                                           scheduler_params=self.train_config['scheduler']['parameters'])

        else:
            self.scheduler = None

    def get_model(self):
        """ Model loading method """

        self.model = get_model(model_name=self.model_config['model_name'],
                               model_parameters=self.model_config['model_parameters'])



    def fit_epoch(self):
        """ Method for training a single learning epoch"""

        self.model.model.train()
        train_loss = []
        for idx, (imgs, labels) in tqdm(enumerate(self.train_loader)):
            imgs, labels = imgs.to(self.model.device), labels.to(self.model.device)

            self.optimizer.zero_grad()
            embs, output = self.model.model(imgs, labels)
            loss = self.criterion(output, labels)

            loss.backward()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss.append(loss.item())
            if idx % 50 == 0:
                print('Train loss:', np.mean(train_loss))

        return np.mean(train_loss)

    def fit_k_nearest_centroids(self, embs: np.ndarray, labels: list):
        """ Method for learning KNN
        Arguments:
            embs: Train embeddings
            labels: Labels
        :returns
            clf.centorids_: Centroids for each classes
            сlf.classes_: Labels
        """

        clf = NearestCentroid(metric='cosine')
        clf.fit(embs, labels)

        return clf.centroids_, clf.classes_


    def compute_val_metrics(self, embeddings: np.ndarray, centroids: np.ndarray, classes: np.ndarray,
                                y_val: list):
        """ Method for calculating metrics on validation
        Arguments:
            embeddings: Validation embeddings
            centroids: Centoids for each classes
            classes: Labels
            y_val: Validation labels
        :returns
           best_map_5: The proportion of correct answers for the top 5 predictions of the model
        """

        confs, preds = self.model.predict(embeddings=embeddings, centroids=centroids, topk=5)

        preds = [[classes[p] for p in pred] for pred in preds]

        best_map_5 = 0
        best_thresh = 0
        weights = np.array([1, 0.8, 0.6, 0.4, 0.2])
        for tresh in [0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8]:  ## Selecting the best threshold for cutting off an alternative class
            precision_5_list = []
            for idx, (pred, conf, true) in enumerate(zip(preds, confs, y_val)):
                y_pred = []
                for p, c in zip(pred, conf): ## We check each prediction
                    if c >= tresh:
                        y_pred.append(p) ## if the similarity is greater than the threshold, we set aside the predicted label
                    else:
                        y_pred.append(self.trainset.individual_id2label['new_individual']) ## otherwise, we replace the prediction with an alternative class label
                pred_mask = np.array(y_pred) == true
                precision_5_list.append(np.sum(weights * pred_mask))

            map_5 = np.mean(precision_5_list)
            print('MAP_5:', map_5, 'Thresh:', tresh)
            if map_5 > best_map_5:
                best_map_5 = map_5
                best_thresh = tresh

        return best_map_5, best_thresh

    def eval_epoch(self):
        """ Method for validating a single epoch """

        self.model.model.eval()

        train_embeddings, y_train = self.model.get_embeddings(dataset=self.trainset_2, data_loader=self.train_loader_2)
        val_embeddings, y_val = self.model.get_embeddings(dataset=self.valset, data_loader=self.val_loader)

        centroids, classes = self.fit_k_nearest_centroids(embs=train_embeddings, labels=y_train)

        map_5, best_thresh = self.compute_val_metrics(embeddings=val_embeddings, centroids=centroids, classes=classes,
                                             y_val=y_val)

        return map_5, best_thresh

    def save_model(self, epoch: int):
        """ Method for saving the model """

        model_path = os.path.join(self.model_config['model_save_path'], self.model_config['model_save_name'])
        if os.path.exists(model_path):
            torch.save(self.model.model, f"{model_path}/{epoch}.pth")
        else:
            os.mkdir(model_path)
            torch.save(self.model.model, f"{model_path}/{epoch}.pth")


    def save_results_log(self, epoch: int, train_loss: float, map_5: float, best_thresh: float):
        """ Method for logging model results """

        log_path = os.path.join(self.train_config['logs_save_path'], f"{self.model_config['model_save_name']}.txt")
        result = f"Epoch={epoch} MAP5={map_5} BestThreshold={best_thresh} TrainLoss={train_loss}"

        with open(log_path, 'a') as f:
            f.write(result)
            f.write('\n')

    def train(self):
        """ Method for performing a model training cycle """

        for epoch in range(1, self.train_config['num_epochs']+1):

            train_loss = self.fit_epoch()
            map_5, best_thresh = self.eval_epoch()

            self.save_model(epoch=epoch)

            self.save_results_log(epoch=epoch, train_loss=train_loss, map_5=map_5, best_thresh=best_thresh)