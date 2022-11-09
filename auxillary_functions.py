import torch, torchvision
from torch import Tensor
from torch.utils.data import Sampler, Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Iterator
from abc import abstractmethod

class FewShotDataset(Dataset):
    # Container for datasets to be used with TaskSampler that includes a get_labels method that is important for FSL
    @abstractmethod
    def __get__item(self, item: int) -> Tuple[Tensor, int]:
        raise NotImplementedError("All datasets need a __getitem__method.")
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("All datasets need a __len__ method.")
    
    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplementedError("FewShotDataset requires a get_labels method.")

class TaskSampler(Sampler):
    def __init__(self, dataset:FewShotDataset, n_way: int, n_shot: int, n_query: int, n_tasks: int):
        super().__init__(data_source = None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.items_per_label = dict()
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]
                
    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat([
                torch.tensor(random.sample(self.items_per_label[label], self.n_shot + self.n_query)) 
                for label in random.sample(self.items_per_label.keys(), self.n_way)
            ]).tolist()
    
    def episodic_collate(self, input_data: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collating function for episodic data loaders
        Inputs:
            input_data: List where each element is a tuple containing an image as a torch Tensor and the image label
        Outputs: 
            tuple(Tensor, Tensor, Tensor, Tensor, List[int])
            Where the outputs are:
            - Support images
            - Support labels
            - Query images
            - Query labels
            - Dataset class ids of class sampled in the episode
        """
        
        true_class_ids = list({x[1] for x in input_data})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape((self.n_way, self.n_shot + self.n_query, *all_images.shape[1:]))
        all_labels = torch.tensor([true_class_ids.index(x[1]) for x in input_data]).reshape((self.n_way, self.n_shot + self.n_query))

        support_images = all_images[:, : self.n_shot].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        
        return (support_images, support_labels, query_images, query_labels, true_class_ids)

def plot_images(images: Tensor, title: str, images_per_row: int):
    #Plot images in a grid
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )

def sliding_average(value_list: List[float], window: int) -> float:
    #Computes the average of the latest instances in a list
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()