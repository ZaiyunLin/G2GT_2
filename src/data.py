# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


'''
key part, change to our dataset
understand wrapper
Add tgt processor

'''
from collator import collator
# from custom_dataset import dataloader
from custom_dataset import UsptoDataset

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from functools import partial

dataset = None
def get_dataset(dataset_name = 'uspto',weak_ensemble=0):
    ''' get dataset and its information
    Args:
        dataset_name (str, optional): dataset name. Defaults to 'uspto'.
    Returns:
        dict: dataset information
    '''
    global dataset
    if dataset is not None:
        return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))  
    dataset = {
        'num_class': 1,
        'loss_fn': F.cross_entropy,
        'metric': 'train_loss',
        'metric_mode': 'min',
        'evaluator': 'none',
        'dataset': UsptoDataset(dataset=dataset_name,weak_ensemble=weak_ensemble),
        'max_node': 420, }

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):
    name = "USPTO"

    def __init__(
        self,
        dataset_name: str = 'uspto50k',
        num_workers: int = 0,
        batch_size: int = 1,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        rel_pos_max: int = 1024,
        weak_ensemble = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        print(f' > dataset_name: {dataset_name}')
        self.dataset = get_dataset(self.dataset_name, weak_ensemble=weak_ensemble)
        te, tr, va = self.dataset['dataset'].get_idx_split()
        self.dataset_test = self.dataset['dataset'][:te]
        #self.dataset_train= self.dataset_test
        self.dataset_train = self.dataset['dataset'][te:tr]     
        self.dataset_val = self.dataset['dataset'][tr:va]
        #self.dataset_test  =self.dataset_val
        
        self.num_workers = num_workers
        self.batch_size = batch_size


        self.multi_hop_max_dist = multi_hop_max_dist
        self.rel_pos_max = rel_pos_max
        self.seed = seed
        



    def setup(self, stage: str = None):
        te, tr, va = self.dataset['dataset'].get_idx_split()
        self.dataset_test = self.dataset['dataset'][:te]
        self.dataset_train = self.dataset['dataset'][te:tr]     
        self.dataset_val = self.dataset['dataset'][tr:va]
        
        #self.dataset_val =self.dataset_train
    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=10,
            persistent_workers = True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)['max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            #batch_size=self.batch_size,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            persistent_workers = False,
            collate_fn=partial(collator, max_node=9999, multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        loader.random = False
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=9999, multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(test_dataloader)', len(loader))
        return loader

