#!/usr/bin/env python
# coding: utf-8

# Training code based on code for the book NLP with PyTorch by Rao & McMahan
# The code has been adapted to train on speech data
# Author: Badr M. Abdullah @  LSV, LST department Saarland University
# Follow me on Twitter @badr_nlp

import os
import yaml
import sys

# NOTE: import torch before pandas, otherwise segementation fault error occurs
# The couse of this problem is UNKNOWN, and not solved yet
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from kaldiio import WriteHelper
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kaldiio
from nn_speech_models import *

# Training Routine
# Helper functions
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args['training_hyperparams']['learning_rate'],
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args['model_state_file']}


def update_train_state(args, model, train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    # save model
    torch.save(model.state_dict(),
        train_state['model_filename'] + \
        str(train_state['epoch_index'] + 1) + '.pth')

    # save model after first epoch
    if train_state['epoch_index'] == 0:
        train_state['stop_early'] = False
        train_state['best_val_accuracy'] = train_state['val_acc'][-1]

    # after first epoch check early stopping criteria
    elif train_state['epoch_index'] >= 1:
        acc_t = train_state['val_acc'][-1]

        # if acc decreased, add one to early stopping criteria
        if acc_t <= train_state['best_val_accuracy']:
            # Update step
            train_state['early_stopping_step'] += 1

        else: # if acc improved
            train_state['best_val_accuracy'] = train_state['val_acc'][-1]

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        early_stop = train_state['early_stopping_step'] >= args['training_hyperparams']['early_stopping_criteria']

        train_state['stop_early'] = early_stop

    return train_state


def compute_accuracy(y_pred, y_target):
    #y_target = y_target.cpu()
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def compute_binary_accuracy(y_pred, y_target):
    y_target = y_target.cpu().long()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long() #.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def get_predictions(y_pred, y_target):
    """Return indecies of predictions. """

    _, y_pred_indices = y_pred.max(dim=1)

    pred_labels = y_pred_indices.tolist()
    true_labels = y_target.tolist()

    return (true_labels, pred_labels)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# obtain user input
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_pah = sys.argv[1] #'/LANG-ID-X/config_1.yml'

config_args = yaml.safe_load(open(config_file_pah))


config_args['model_id'] = '_'.join(str(ip) for ip in
    [
        config_args['model_arch']['nn_model'],
        #config_args['model_arch']['bottleneck_size'],
        #config_args['model_arch']['output_dim'],
        #config_args['model_arch']['signal_dropout_prob'],
        config_args['input_signal_params']['feature_type'],
        config_args['input_signal_params']['experiment_type']
    ]
)

if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'],
        '_'.join([config_args['model_state_file'],
        config_args['model_id']])
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))


# Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))

# Set seed for reproducibility
set_seed_everywhere(config_args['seed'], config_args['cuda'])

# handle dirs
handle_dirs(config_args['model_save_dir'])

##### HERE IT ALL STARTS ...
# source vectorizer ...
source_speech_df = pd.read_csv(config_args['source_speech_metadata'], encoding='utf-8')

source_label_set=config_args['source_language_set'].split()

# make sure no utterances with 0 duration such as
#source_speech_df = source_speech_df[(source_speech_df.duration!=0)]

source_speech_df = source_speech_df[(source_speech_df['language'].isin(source_label_set))]


len(source_speech_df), source_label_set

# source vectorizer ...
target_speech_df = pd.read_csv(config_args['target_speech_metadata'], encoding='utf-8')

target_label_set=config_args['target_language_set'].split()

# make sure no utterances with 0 duration such as
#target_speech_df = target_speech_df[(target_speech_df.duration!=0)]

target_speech_df = target_speech_df[(target_speech_df['language'].isin(target_label_set))]

len(target_speech_df), target_label_set


cmvn_stats = kaldiio.load_mat(config_args['source_cmvn'])
mean_stats = cmvn_stats[0,:-1]
count = cmvn_stats[0,-1]
offset = np.expand_dims(mean_stats,0)/count

source_speech_vectorizer = LID_Vectorizer(
    data_dir=config_args['source_data_dir'],
    speech_df=source_speech_df,
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['source_language_set'].split(),
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    num_frames=config_args['input_signal_params']['num_frames'],
    feature_dim=config_args['model_arch']['feature_dim'],
    start_idx=config_args['input_signal_params']['start_index'],
    end_idx=config_args['input_signal_params']['end_index'],
    cmvn = offset
)
print(source_speech_vectorizer.index2lang)

cmvn_stats = kaldiio.load_mat(config_args['target_cmvn'])
mean_stats = cmvn_stats[0,:-1]
count = cmvn_stats[0,-1]
offset = np.expand_dims(mean_stats,0)/count

target_speech_vectorizer = LID_Vectorizer(
    data_dir=config_args['target_data_dir'],
    speech_df=target_speech_df,
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['target_language_set'].split(),
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    num_frames=config_args['input_signal_params']['num_frames'],
    feature_dim=config_args['model_arch']['feature_dim'],
    start_idx=config_args['input_signal_params']['start_index'],
    end_idx=config_args['input_signal_params']['end_index'],
    cmvn = offset
)
print(target_speech_vectorizer.index2lang)


# data loaders ....
source_speech_dataset = LID_Dataset(source_speech_df, source_speech_vectorizer)
target_speech_dataset = LID_Dataset(target_speech_df, target_speech_vectorizer)


if config_args['model_arch']['nn_model'] == 'ConvNet_DA':
    nn_LID_model_DA = ConvNet_LID_DA(
        feature_dim=config_args['model_arch']['feature_dim'],
        bottleneck=config_args['model_arch']['bottleneck'],
        bottleneck_size=config_args['model_arch']['bottleneck_size'],
        output_dim=config_args['model_arch']['output_dim'],
        dropout_frames=config_args['model_arch']['frame_dropout'],
        dropout_features=config_args['model_arch']['feature_dropout'],
        signal_dropout_prob=config_args['model_arch']['signal_dropout_prob'],
        num_channels=config_args['model_arch']['num_channels'],
        num_classes= len(source_label_set),   # or config_args['model_arch']['num_classes'],
        filter_sizes=config_args['model_arch']['filter_sizes'],
        stride_steps=config_args['model_arch']['stride_steps'],
        pooling_type=config_args['model_arch']['pooling_type']
    )

nn_LID_model_DA.load_state_dict(torch.load(config_args['best_model']))

# test model
#x_in = torch.rand(1, 13, 384)
#nn_LID_model_DA.forward(x_in)


print(nn_LID_model_DA)



loss_func_cls = nn.CrossEntropyLoss()
loss_func_dmn  = nn.CrossEntropyLoss()

optimizer = optim.Adam(nn_LID_model_DA.parameters(),lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = make_train_state(config_args)

# this line was added due to RunTimeError
# NOTE: uncomment this
nn_LID_model_DA.cuda()


src_val_balanced_acc_scores = []
tgt_val_balanced_acc_scores = []

num_epochs = config_args['training_hyperparams']['num_epochs']
batch_size = config_args['training_hyperparams']['batch_size']

with WriteHelper('ark,scp:/home/gnani/da-lang-id/davector.ark,/home/gnani/da-lang-id/davector.scp') as writer:
    print('writing embeddings has started.')
    for epoch_index in range(1):
        ##### TRAINING SECTION
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset
        # setup: batch generator, set loss and acc to 0, set train mode on
        source_speech_dataset.set_mode('TRA')
        source_total_batches = source_speech_dataset.get_num_batches(batch_size)

        source_batch_generator = generate_batches(
            source_speech_dataset,
            batch_size=256,
            device=config_args['device']
        )

        target_speech_dataset.set_mode('TRA')
        target_total_batches = target_speech_dataset.get_num_batches(batch_size)

        target_batch_generator = generate_batches(
            target_speech_dataset,
            batch_size=batch_size,
            device=config_args['device']
        )


        max_batches = min(source_total_batches, target_total_batches)
        #print(source_total_batches, target_total_batches)

        running_cls_loss = 0.0
        running_dmn_loss = 0.0

        running_cls_acc = 0.0
        running_src_dmn_acc = 0.0
        running_tgt_dmn_acc = 0.0

        nn_LID_model_DA.eval()

        
        for batch_index, src_batch_dict in enumerate(source_batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients

            # step 2. training progress and GRL lambda
            p = float(batch_index + epoch_index * max_batches) / (num_epochs * max_batches)
            _lambda = 2. / (1. + np.exp(-5 * p)) - 1

            #print("src_batch_dict['x_data']:",src_batch_dict['x_data'].shape)
            bs = src_batch_dict['x_data'].shape[0] 
            # step 3. forward pass and compute loss on source domain
            src_dmn_trues = torch.zeros(bs, dtype=torch.long, device=config_args['device']) # generate source domain labels
            src_cls_trues = src_batch_dict['y_target']
            emb = nn_LID_model_DA.emb(x_in=src_batch_dict['x_data'])
            #print(src_batch_dict)
            emb = emb.cpu()
            for utt, embedding in zip(src_batch_dict['uttr_id'],emb):
                writer(utt, embedding.numpy())
            print(batch_index)        

