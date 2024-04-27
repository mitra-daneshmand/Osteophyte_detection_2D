import gc
import os
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from seed import seed
import datasets
import transforms
from trainer import ModelTrainer
from args import parse_args
from train_utils import estimate_mean_std

from kvs import GlobalKVS
from termcolor import colored
import ssl

import warnings


logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
seed.set_ultimate_seed()
warnings.filterwarnings('ignore')


def main():
    args = parse_args()
    print(colored('Config arguments', 'green'))
    print('network              :', args.backbone)
    print('no epochs            :', args.epochs)
    print('batch size           :', args.batch_size)
    print('learning rate        :', args.learning_rate)
    print('weight_decay         :', args.weight_decay)
    print('target compartment   :', args.target_comp)

    kvs = GlobalKVS()
    kvs.update('args', args)

    metadata = datasets.init_metadata(args.lm, args.target_comp, args.csv_dir)

    save_dir = args.output_dir + '/{}/'.format(args.target_comp)

    test_index = pd.read_csv(os.path.join('sessions_final/FM', 'test.csv'))
    train_val_index = pd.read_csv(os.path.join('sessions_final/FM', 'train_val.csv'))
    test_set, train_val_set = metadata.iloc[test_index.values.flatten()], metadata.iloc[train_val_index.values.flatten()]
    train_val_set = pd.concat([train_val_set, test_set])

    train_val_set.reset_index(inplace=True, drop=True)

    for fold_num in range(5):
        logger.info(f'Training fold {fold_num}')

        train_index = pd.read_csv(os.path.join(save_dir, 'fold_{}_train.csv'.format(str(fold_num))))
        val_index = pd.read_csv(os.path.join(save_dir, 'fold_{}_val.csv'.format(str(fold_num))))
        val_set, train_set = train_val_set.iloc[val_index.values.flatten()], train_val_set.iloc[train_index.values.flatten()]

        torch.cuda.empty_cache()

        kvs.update('cur_fold', fold_num)
        kvs.update('prev_model', None)

        print(colored('====> ', 'blue') + f'Training fold {fold_num}....')

        mean, std = 0.0507, 0.2122  # lat
        # mean, std = 0.0509, 0.2125  # med

        print('Preparing datasets...')
        train_dataset = datasets.KneeDataset(
            dataset=train_set,
            transforms=[
                transforms.HorizontalFlip(prob=0.5),
                transforms.OneOf([
                            transforms.DualCompose([
                                transforms.Scale(ratio_range=(0.7, 0.8), prob=1.),
                                transforms.Scale(ratio_range=(1.5, 1.6), prob=1.),
                            ]),
                            transforms.NoTransform()
                        ]),
                transforms.Crop(output_size=(500, 673)),
                transforms.Rotate(degree_range=[-10., 10.], prob=0.5),
                transforms.Normalize(mean=mean, std=std)
            ]
        )
        val_dataset = datasets.KneeDataset(
            dataset=val_set,
            transforms=[
                transforms.Normalize(mean=mean, std=std),
            ]

        )

        loaders = dict()
        loaders['train'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=8, pin_memory=torch.cuda.is_available())
        loaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=8, pin_memory=torch.cuda.is_available())

        print('Building model and trainer...')
        trainer = ModelTrainer(fold_idx=fold_num)

        tmp = trainer.fit(loaders=loaders)
        metrics_train, fnames_train, metrics_val, fnames_val = tmp


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
