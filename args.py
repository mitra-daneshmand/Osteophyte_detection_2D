import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('--backbone', type=str, choices=['se_resnet50',
                                                         'resnet18',
                                                         'inceptionv4',
                                                         'se_resnext50_32x4d',
                                                         'se_resnext101_32x4d'
                                                         ], default='resnet18')

    parser.add_argument('-out', '--output_dir', default='sessions',
                        help='Complete path for the model weights and other results. (default: sessions/)')

    parser.add_argument('--pretrain', default=True,
                        help='pretraining or not: False (default), True')

    parser.add_argument('-csv', '--csv_dir', default='../data/CSVs/',
                        help='CSV files directory: ../data/CSVs/ (default)')

    parser.add_argument('--n_classes', default=1, type=int,
                        help='Number of classes')

    parser.add_argument('--ft_portion', default='complete', type=str,
                        help='The portion of the model to apply fine tuning, Options: complete, 6_layers')

    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help='Number of epochs to train the model. (default: 20)')

    parser.add_argument('-bs', '--batch_size', default=64, type=int,
                        help='Training batch size. (default: 16)')

    parser.add_argument('--dropout_rate', type=float, default=0.5)

    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
                        help='Learning rate. (default: 1e-3)')

    parser.add_argument('-wd', '--weight_decay', default=1e-5,
                        help='Weight decay for training. Options: 1e-5 (default), 1e-4')

    parser.add_argument('-tar_comp', '--target_comp', default='FL',
                        help='Knee area for OST progression definition. Options: FM, FL, TL, TM')

    parser.add_argument('-lm', '--lm', default='lateral',
                        help='lateral or medial comps. Options: (default), medial, lateral')

    args = parser.parse_args()
    return args
