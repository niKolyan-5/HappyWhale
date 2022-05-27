import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch

from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from src.models.dolgnet_model import DolgNet




class RecognitionDolgnetModel:
    """ RecognitionDolgnetModel class
    Attributes:
        dolgnet_parameters: DolgNet parameters
        gpus: list of GPU device numbers
        emb_size: embedding output size
        weights: path fow model weights (if None use ' ')
        thresh: threshold for separating an alternative class
        """

    def __init__(self, dolgnet_parameters: dict, gpus: list, emb_size: int, weights: str,
                 thresh: float):

        self.dolgnet_parameters = dolgnet_parameters
        self.gpus = gpus
        self.emb_size = emb_size
        self.weights = weights
        self.thresh = thresh

        self.load_model()


    def load_model(self):
        """ Load model  """
        if self.weights != '':
            self.model = torch.load(self.weights, map_location='cpu')
        else:
            self.model = DolgNet(**self.dolgnet_parameters)

        if len(self.gpus) == 1:
            self.device = f'cuda:{self.gpus[0]}'
            self.model = self.model.to(self.device)
        else:
            self.device = 'cuda'
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
            self.model.to(self.device)

    @torch.no_grad()
    def get_embeddings(self, dataset: Dataset, data_loader: DataLoader):
        """ Get embeddings for all data in dataloader
        Arguments:
            dataset: Dataset
            data_loader: Dataloaderr
        :returns
            embeddings: Embeddings for all data
            y_trues: Labels for all data
        """
        embeddings = np.zeros(shape=(len(dataset), self.emb_size))
        idx = 0
        y_trues = []
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(self.device)

            embs = self.model(imgs).to('cpu').detach().numpy()
            y_trues.extend([label.item() for label in labels])

            for emb in embs:
                embeddings[idx] = emb
                idx += 1

        embeddings = np.float32(embeddings)
        embeddings = normalize(embeddings, axis=1, norm='l2')

        return embeddings, y_trues

    def predict(self, embeddings: np.ndarray, centroids: np.ndarray, topk: int):
        """ Predict method
        Arguments:
            embeddings: Embeddings of images
            centroids: centroids of classes
            topk: topk predictions
        :returns
             confs: predictions confidence
             preds: predictions
        """

        similarities = np.abs(cosine_similarity(embeddings, centroids))
        torch_similarities = torch.from_numpy(similarities).float()
        confs, preds = torch.topk(torch_similarities, topk, dim=1)
        confs, preds = confs.numpy(), preds.numpy()

        return confs, preds


def get_model(model_name: str, model_parameters: dict):
    """ Get model function
    Arguments:
        model_name: model name
        model_parameters: model parameters
    """
    model_mapping = {'dolgnet': RecognitionDolgnetModel(**model_parameters)}

    return model_mapping[model_name]