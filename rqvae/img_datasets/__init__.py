# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import Subset
import torchvision
from torchvision.datasets import ImageNet
from tqdm import tqdm
from omegaconf import DictConfig

from .lsun import LSUNClass
from .ffhq import ImageFolder, FFHQ
from .transforms import create_transforms

import torchvision.transforms as transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))

class VariableBatchDistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
  def __init__(self, dataset, batch_sizes, num_replicas=None, rank=None, shuffle=True, seed=0):
    super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
    self.buckets = list(dataset.bucket_indices.keys())
    self.bucket_sizes = dataset.get_bucket_sizes()
    # Only print once from rank 0
    if rank == 0:
      print("Bucket sizes:", self.bucket_sizes)
    self.batch_sizes = batch_sizes

  def __iter__(self):
    # deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(self.seed + self.epoch)

    indices = []
    for bucket in self.buckets:
      bucket_indices = self.dataset.bucket_indices[bucket]
      batch_size = self.batch_sizes[bucket]
      
      if self.shuffle:
        # Shuffle indices within each bucket
        indices_bucket = torch.randperm(len(bucket_indices), generator=g).tolist()
        indices_bucket = [bucket_indices[i] for i in indices_bucket]
      else:
        indices_bucket = bucket_indices

      # Add padding to make divisible by (batch_size * num_replicas)
      padding_size = (batch_size * self.num_replicas - len(indices_bucket) % (batch_size * self.num_replicas)) % (batch_size * self.num_replicas)
      if padding_size > 0:
        indices_bucket = indices_bucket + indices_bucket[:padding_size]

      # Verify divisibility before subsampling
      if len(indices_bucket) % (batch_size * self.num_replicas) != 0:
        raise ValueError(f"Bucket {bucket} size {len(indices_bucket)} is not divisible by batch_size ({batch_size}) * num_replicas ({self.num_replicas})")

      # Subsample for this rank
      indices_bucket = indices_bucket[self.rank * batch_size:len(indices_bucket):self.num_replicas * batch_size]
      indices.extend(indices_bucket)

    if self.shuffle:
      # Shuffle all indices
      indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

    return iter(indices)

class ScoreDataset(ImageFolder):
    def __init__(self, root, split='train', **kwargs):
        train_list_file = f'{root}/train.txt'
        val_list_file = f'{root}/test.txt'
        super().__init__(root, train_list_file, val_list_file, split, **kwargs)

class MultiHeightScoreDataset(ScoreDataset):
    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, split, **kwargs)
        
        heights = []
        for sample in tqdm(self.samples, desc='Loading image heights'):
            img = self.loader(sample)
            heights.append(img.height)

        self.heights = heights
        
        # Create bucket indices that will be used by DistributedBucketSampler
        self.bucket_indices = {
            '70_130': [i for i, h in enumerate(heights) if 70 <= h < 130],
            '130_260': [i for i, h in enumerate(heights) if 130 <= h < 260], 
            '260_360': [i for i, h in enumerate(heights) if 260 <= h < 360],
            '360_390': [i for i, h in enumerate(heights) if 360 <= h <= 390]
        }

        # Get indices for images with height >= 260px
        self.big_indices = self.bucket_indices['260_360'] + self.bucket_indices['360_390']

        # Create transforms for each bucket
        self.bucket_transforms = {
            '70_130': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            '130_260': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            '260_360': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            '360_390': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(352),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        }

    def get_bucket_sizes(self):
        return {k: len(v) for k, v in self.bucket_indices.items()}

    def __getitem__(self, index, with_transform=True):
        sample, _ = super().__getitem__(index, with_transform=False)
        height = self.heights[index]
        
        if with_transform:
            if 70 <= height < 130:
                sample = self.bucket_transforms['70_130'](sample)
            elif 130 <= height < 260:
                sample = self.bucket_transforms['130_260'](sample)
            elif 260 <= height < 360:
                sample = self.bucket_transforms['260_360'](sample)
            elif 360 <= height <= 390:
                sample = self.bucket_transforms['360_390'](sample)
                
        return sample, 0
    
    def get_big_items(self, idx):
        # Get the image at specified index
        sample, _ = super().__getitem__(self.big_indices[idx], with_transform=False)
        sample = self.bucket_transforms['260_360'](sample)
        
        return sample, 0

# class Grandstaff(ImageFolder):
#     train_list_file = 'data/Olimpic_grandstaff_128_gray/train.txt'
#     val_list_file = 'data/Olimpic_grandstaff_128_gray/test.txt'

#     def __init__(self, root, split='train', **kwargs):
#         super().__init__(root, Grandstaff.train_list_file, Grandstaff.val_list_file, split, **kwargs)


def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'imagenet':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'imagenet_u':
        root = root if root else 'data/imagenet'

        def target_transform(_):
            return 0
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn, target_transform=target_transform)
        dataset_val = ImageNet(root, split='val', transform=transforms_val, target_transform=target_transform)
    elif config.dataset.type == 'ffhq':
        root = root if root else 'data/ffhq'
        dataset_trn = FFHQ(root, split='train', transform=transforms_trn)
        dataset_val = FFHQ(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        root = root if root else 'data/lsun'
        category_name = config.dataset.type.split('-')[-1]
        dataset_trn = LSUNClass(root, category_name=category_name, transform=transforms_trn)
        dataset_val = LSUNClass(root, category_name=category_name, transform=transforms_val)
    elif config.dataset.type == 'Olimpic_grandstaff_128_gray':
        root = root if root else 'data/Olimpic_grandstaff_128_gray'
        dataset_trn = ScoreDataset(root, split='train', transform=transforms_trn)
        dataset_val = ScoreDataset(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'LSDSQ_flattened_240_gray':
        root = root if root else 'data/LSDSQ_flattened_240_gray'
        dataset_trn = ScoreDataset(root, split='train', transform=transforms_trn)
        dataset_val = ScoreDataset(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'LSD_360anchored_gray':
        root = root if root else 'data/LSD_360anchored_gray'
        assert isinstance(config.experiment.batch_size, DictConfig)
        dataset_trn = MultiHeightScoreDataset(root, split='train', transform=transforms_trn)
        dataset_val = MultiHeightScoreDataset(root, split='val', transform=transforms_val)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val
