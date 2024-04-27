import os
import gc

import numpy as np
import pandas as pd
import cv2
import torch
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

import transforms
from args import parse_args
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from args import parse_args
from tqdm import tqdm
from seed import seed


# H = 300
# W = 245
H = 500
W = 673
# H = 80
# W = 245

def read_mask(path_file):
    args = parse_args()
    mask = cv2.imread(path_file)

    if args.lm == 'medial':
        mask[(mask != 1) & (mask != 2)] = 0

        ret = np.empty((3, mask.shape[0], mask.shape[1]), dtype=mask.dtype)
        ret[0, :, :] = 0
        ret[1, :, :] = np.isin(mask[:, :, 1], 1).astype(np.uint8)
        ret[2, :, :] = np.isin(mask[:, :, 2], 2).astype(np.uint8)
    else:
        mask[(mask != 100) & (mask != 200)] = 0
        mask[mask == 100] = 1
        mask[mask == 200] = 2
        ret = np.empty((3, mask.shape[0], mask.shape[1]), dtype=mask.dtype)
        ret[0, :, :] = 0
        ret[1, :, :] = np.isin(mask[:, :, 1], 1).astype(np.uint8)
        ret[2, :, :] = np.isin(mask[:, :, 2], 2).astype(np.uint8)

    ret0 = cv2.resize(ret[0, :, :], (W, H))
    ret1 = cv2.resize(ret[1, :, :], (W, H))
    ret2 = cv2.resize(ret[2, :, :], (W, H))
    ret_final = np.empty((3, *(H, W)), dtype=mask.dtype)
    ret_final[0, :, :] = ret0
    ret_final[1, :, :] = ret1
    ret_final[2, :, :] = ret2
    # args = parse_args()
    # tmp = cv2.imread(path_file)
    # mask = np.zeros((H, W, 3))
    # mask[:tmp.shape[0], :tmp.shape[1], :] = tmp
    #
    # # mask = mask[25:mask.shape[0]-25, :, :]
    # # mask[:10, :, :] = 0
    # # # mask[mask.shape[0] - 10:, :, :] = 0
    # '''
    # # if 'L' in path_file:
    # #     mask = cv2.flip(mask, 1)
    # '''
    # '''
    # if args.lm == 'medial':
    #     mask[(mask != 1) & (mask != 2)] = 0
    #
    #     # tmp = mask.copy()
    #     # mask[:, :, 0][(tmp[:, :, 1] == 1)] = 1
    #     # mask[:, :, 1][(tmp[:, :, 1] == 1)] = 100
    #     # mask[:, :, 2][(tmp[:, :, 1] == 1)] = 200
    #
    #     # mask[:, :, 0][(tmp[:, :, 2] == 2)] = 2
    #     # mask[:, :, 1][(tmp[:, :, 2] == 2)] = 150
    #     # mask[:, :, 2][(tmp[:, :, 2] == 2)] = 250
    #
    #     # mask = cv2.resize(mask, (W, H))
    #     # mask = mask.reshape((1, *mask.shape))
    #
    #     # ret = np.empty((mask.shape), dtype=mask.dtype)
    #     # ret[:, :, 0] = 0
    #     # ret[:, :, 1] = np.isin(mask, 1).astype(np.uint8)
    #     # ret[:, :, 2] = np.isin(mask, 2).astype(np.uint8)
    # else:
    #     mask[(mask != 100) & (mask != 200)] = 0
    #     mask[mask == 100] = 1
    #     mask[mask == 200] = 2
    #     # ret = np.empty((mask.shape), dtype=mask.dtype)
    #     # ret[:, :, 0] = 0
    #     # ret[:, :, 1] = np.isin(mask, 1).astype(np.uint8)
    #     # ret[:, :, 2] = np.isin(mask, 2).astype(np.uint8)
    # '''
    # mask[mask == 2] = 1
    # mask[(mask != 0)] = 1
    #
    # ret_tmp = mask.copy()
    # ret0 = cv2.resize(ret_tmp[:, :, 0], (W, H))
    # ret1 = cv2.resize(ret_tmp[:, :, 1], (W, H))
    # ret2 = cv2.resize(ret_tmp[:, :, 2], (W, H))
    # ret_final = np.empty((3, *(H, W)), dtype=mask.dtype)
    # ret_final[0, :, :] = ret0
    # ret_final[1, :, :] = ret1
    # ret_final[2, :, :] = ret2
    #
    # # ret2[ret2==2]=1
    # # ret_final[2, :, :] = ret2
    #
    # # ret2[ret1==2]=1
    # # ret_final[2, :, :] = ret2
    # # ret_final[0, :, :] = 1
    # # ret = (ret1+ret2)+1
    # # ret1[ret1==2]=0
    # # ret1[ret1==3]=0
    # # ret1[ret1 != 1]=0
    # # ret1[ret1 == 1]=3
    # # ret_final[0, :, :] =ret1
    #
    # # ret_final = mask.reshape((1, *mask.shape))
    #
    # # if 'L' in path_file:
    # #     mask = cv2.flip(mask, 1)
    return ret  # ret_final


class KneeDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        if isinstance(ind, torch.Tensor):
            ind = ind.item()
        mask = read_mask(self.dataset['path_mask'].iloc[ind])

        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, 'randomize'):
                    t.randomize()
                mask = t(mask)

        res = {'ID_SIDE': str(self.dataset['ID'].iloc[ind]) + '_' + self.dataset['SIDE'].iloc[ind],
               'mask': mask,
               'Target': self.dataset['Target'].iloc[ind],
               }

        return res


def init_metadata(lm, target_comp, csv_dir):
    # metadata = pd.read_csv(os.path.join(csv_dir, 'OAI_' + tissue + lm + '_imgs_metadata.csv'))
    metadata = pd.read_csv(os.path.join(csv_dir, 'OAI_bl_scaled_All_{}_imgs_metadata.csv'.format(lm)))

    metadata['path_mask'] = ''
    mri_dir = 'data/scaled_All/{}'.format(lm)
    masks_dir = 'xray_bone_segmentation/data/splitted_masks'
    ext_mask_name = '_side.png'
    metadata['path_mask'] = metadata.apply(
        lambda metadata: metadata['imgs'].split(',')[0][:-10].replace(mri_dir, masks_dir), axis=1)
    metadata['path_mask'] = metadata.apply(
        lambda metadata: (metadata['path_mask'] + ext_mask_name).replace('side', metadata['SIDE']), axis=1)

    metadata = multilabel_target(metadata, target_comp)
    del metadata['Target']  # FL
    metadata['Target'] = 0  # FL

    if len(target_comp) > 2:
        metadata['Target'].iloc[metadata[(metadata[target_comp[:2]] > 0) | (metadata[target_comp[2:]] > 0)].index] = 1  # FL
    else:
        metadata['Target'].iloc[metadata[metadata[target_comp] > 0].index] = 1  # FL

    return metadata


def multilabel_target(df, target_comp):
    if len(target_comp) > 2:
        categorical_vars = [target_comp[:2], target_comp[2:]]
    else:
     categorical_vars = [target_comp]

    one_hot_encoder = OneHotEncoder(sparse=False, drop="first")
    encoder_vars_array = one_hot_encoder.fit_transform(df[categorical_vars])
    encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
    encoder_vars_df = pd.DataFrame(encoder_vars_array, columns=encoder_feature_names, dtype=int)
    X_new = pd.concat([df.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1)

    df['Target'] = [x for x in df[categorical_vars].to_numpy()]

    y = df[categorical_vars].to_numpy()
    y_for_stratif = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    df['y_stratif'] = y_for_stratif

    return df


def estimate_mean_std(train_dataset):
    mean_std_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, pin_memory=torch.cuda.is_available())
    means = []
    stds = []
    for sample in tqdm(mean_std_loader, desc='Computing mean and std values:'):
        local_batch, local_labels = sample

        means.append(torch.mean(local_batch, dim=(1, 3, 4)).cpu().numpy())
        stds.append(torch.std(local_batch, dim=(1, 3, 4)).cpu().numpy())

    mean = np.mean(np.concatenate(means, axis=0), axis=0)
    std = np.mean(np.concatenate(stds, axis=0), axis=0)

    return mean, std
