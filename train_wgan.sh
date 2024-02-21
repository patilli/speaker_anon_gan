#!/usr/bin/env bash

GPU_IDS=1
DIR_N=run_name_v1

DATA_DIR=data/reference_embeddings_as_list.pt
CONFIG=./configs/gan_anon.json
NAME=$DIR_N

eval "$(conda shell.bash hook)"
conda activate my_env

CUDA_VISIBLE_DEVICES=$GPU_IDS nohup python -u train_anon_gan.py\
                                           --data_dir $DATA_DIR\
                                           --config $CONFIG\
                                           --comet_ml_experiment_name $NAME\
                                            > $DIR_N.out 2>&1 &
