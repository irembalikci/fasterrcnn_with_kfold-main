"""
USAGE

# training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --mosaic 0 --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Distributed training:
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data data_configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16
"""
from torch_utils.engine import (
    train_one_epoch, evaluate, utils
)
from torch.utils.data import (
    distributed, RandomSampler, SequentialSampler
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, Averager, 
    save_model, save_loss_plot,
    show_tranformed_image,
    save_mAP, save_model_state, SaveBestModel,
    yaml_save, init_seeds, EarlyStopping
)
from utils.logging import (
    set_log, coco_log,
    set_summary_writer, 
    tensorboard_loss_log, 
    tensorboard_map_log,
    csv_log,
    wandb_log, 
    wandb_save_model,
    wandb_init
)

import torch
import argparse
import yaml
import numpy as np
import torchinfo
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  # Or mAP, or any other metric you're using

torch.multiprocessing.set_sharing_strategy('file_system')

RANK = int(os.getenv('RANK', -1))

# For same annotation colors each time.
np.random.seed(42)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='name of the model'
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default='cuda',
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', 
        default=5,
        type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers', 
        default=4,
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=4, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '--lr', 
        default=0.001,
        help='learning rate for the optimizer',
        type=float
    )
    parser.add_argument(
        '-ims', '--imgsz',
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-n', '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', 
        dest='vis_transformed', 
        action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '--mosaic', 
        default=0.0,
        type=float,
        help='probability of applying mosaic, (default, always apply)'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', 
        dest='use_train_aug', 
        action='store_true',
        help='whether to use train augmentation, blur, gray, \
              brightness contrast, color jitter, random gamma \
              all at once'
    )
    parser.add_argument(
        '-ca', '--cosine-annealing', 
        dest='cosine_annealing', 
        action='store_true',
        help='use cosine annealing warm restarts'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None, 
        type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training', 
        dest='resume_training', 
        action='store_true',
        help='whether to resume training, if true, \
            loads previous training plots and epochs \
            and also loads the otpimizer state dictionary'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '--world-size', 
        default=1, 
        type=int, 
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url used to set up the distributed training'
    )
    parser.add_argument(
        '-dw', '--disable-wandb',
        dest="disable_wandb",
        action='store_true',
        help='whether to use the wandb'
    )
    parser.add_argument(
        '--sync-bn',
        dest='sync_bn',
        help='use sync batch norm',
        action='store_true'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='use automatic mixed precision'
    )
    parser.add_argument(
        '--patience',
        default=10,
        help='number of epochs to wait for when mAP does not increase to \
              trigger early stopping',
        type=int
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int ,
        help='golabl seed for training'
    )
    parser.add_argument(
        '--project-dir',
        dest='project_dir',
        default=None,
        help='save resutls to custom dir instead of `outputs` directory, \
              --project-dir will be named if not already present',
        type=str
    )

    args = vars(parser.parse_args())
    return args

def main(args):
    # Initialize distributed mode.
    utils.init_distributed_mode(args)

    # Initialize W&B with project name.
    if not args['disable_wandb']:
        wandb_init(name=args['name'])
        
    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    init_seeds(args['seed'] + 1 + RANK, deterministic=True)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
    TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
    VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
    VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    DEVICE = torch.device(args['device'])
    BATCH_SIZE = args['batch']
    OUT_DIR = set_training_dir(args['name'], args['project_dir'])
    
    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=args['seed'])  # Adjust n_splits as needed
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(np.arange(len(TRAIN_DIR_IMAGES)))):
        print(f"Training Fold {fold+1}")
        
        # Create train/valid datasets for this fold
        train_dataset = create_train_dataset(
            TRAIN_DIR_IMAGES[train_idx], 
            TRAIN_DIR_LABELS[train_idx],
            args['imgsz'], 
            CLASSES,
            use_train_aug=args['use_train_aug'],
            mosaic=args['mosaic'],
            square_training=args['square_training']
        )
        
        valid_dataset = create_valid_dataset(
            VALID_DIR_IMAGES[valid_idx], 
            VALID_DIR_LABELS[valid_idx], 
            args['imgsz'], 
            CLASSES,
            square_training=args['square_training']
        )

        if args['distributed']:
            train_sampler = distributed.DistributedSampler(train_dataset)
            valid_sampler = distributed.DistributedSampler(valid_dataset, shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = SequentialSampler(valid_dataset)

        train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=train_sampler)
        valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler)
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}\n")

        # Re-initialize the model for each fold
        if args['weights'] is None:
            print('Building model from scratch...')
            model = create_model(args['model'], num_classes=NUM_CLASSES, pretrained=True)
        else:
            print('Loading pretrained weights...')
            checkpoint = torch.load(args['weights'], map_location=DEVICE)
            build_model = create_model[args['model']]
            model = build_model(num_classes=NUM_CLASSES)
            model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(DEVICE)

        # Re-initialize the optimizer for each fold
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)

        # Optional: If using a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Train and evaluate the model for the current fold
        train_loss_hist = Averager()
        for epoch in range(args['epochs']):
            train_loss_hist.reset()
            train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=100, scheduler=scheduler)
            
            # Evaluate on validation data for the current fold
            stats, val_pred_image = evaluate(model, valid_loader, device=DEVICE)
            val_map = stats[0]  # Change this to the metric you want to use (mAP or accuracy)
            
        fold_scores.append(val_map)

    # After all folds
    mean_score = np.mean(fold_scores)
    print(f"KFold Cross-Validation Scores: {fold_scores}")
    print(f"Mean Score: {mean_score}")

    # Save models to Weights&Biases (Optional)
    if not args['disable_wandb']:
        wandb_save_model(OUT_DIR)



if __name__ == '__main__':
    args = parse_opt()
    main(args)

