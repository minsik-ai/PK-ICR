#!/bin/sh

python finetune/grounding_finetune.py \
    --input_path data/lines_train.json \
    --output_path finetuned_persona_model \
    --hard_neg_only
