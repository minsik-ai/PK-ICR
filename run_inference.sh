#!/bin/sh

python retrieve/retrieve_grounding.py \
    --add_grounding \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_sbert_persona_reselect_with_aug_finetune_0_5.json \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --persona_reselect_finetune \
    --hard_neg_comp
