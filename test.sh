#!/bin/bash
python run.py \
    --ckpt /content/Action_Recognition/models/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 51 \
    --model_type lrcn \
    --batch_size 4 \
    --mode eval
