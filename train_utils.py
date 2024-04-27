from model import KneeNet, PretrainedModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from args import parse_args
from model import resnet10_2D
from model import Custom_VGG


def init_model():
    args = parse_args()

    if args.pretrain:
        net = PretrainedModel(args.backbone, args.dropout_rate, 1, True)
    else:
        net = KneeNet(args.backbone, args.dropout_rate, args.pretrain)
    return net


def estimate_mean_std(dataset):
    args = parse_args()
    mean_std_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=torch.cuda.is_available())
    len_inputs = len(mean_std_loader.sampler)
    mean = 0
    std = 0
    for sample in tqdm(mean_std_loader, desc='Computing mean and std values:'):
        local_batch, local_labels = sample['mask'], sample['Target']

        for j in range(local_batch.shape[0]):
            mean += local_batch.float()[j, :, :, :].mean()
            std += local_batch.float()[j, :, :, :].std()

    mean /= len_inputs
    std /= len_inputs

    return mean, std
