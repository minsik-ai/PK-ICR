# PK-ICR: Persona-Knowledge Interactive Multi-Context Retrieval for Grounded Dialogue
This repository contains training and evaluation code for our **EMNLP 2023 Oral** paper [Persona-Knowledge Interactive Multi-Context Retrieval for Grounded Dialogue](https://arxiv.org/abs/2302.06674).

## Datasets

To execute the scripts, train and test datasets in JSONL form is required.

```
{ ... } # turn 1
{ ... } # turn 2
...

```

Data is available at following : [train_data](https://drive.google.com/file/d/1YmEW12HqjAjlEfZ05g8VLRux8kyUjdcI/view?usp=sharing), [test data](https://codalab.lisn.upsaclay.fr/my/datasets/download/341d58e9-34a9-4dcc-ab16-0720d11d4b37)

## Pipeline

### 1. Preprocess: Convert JSON to JSONL

```bash
python retrieve/proc_lines.py \
    --input_path data/train.json \
    --output_path data/lines_train.json

python retrieve/proc_lines.py \
    --input_path data/test_public.json \
    --output_path data/lines_test_public.json
```

### 2. Finetune Persona Model

Cross-encoder:
```bash
python finetune/persona_finetune.py \
    --input_path data/lines_train.json \
    --output_path finetuned_persona_cross \
    --hard_neg_only
```

Bi-encoder:
```bash
python finetune/persona_finetune.py \
    --input_path data/lines_train.json \
    --output_path finetuned_persona_bi \
    --hard_neg_only \
    --bi_enc
```

### 3. Run Grounding Retrieval

#### Unfinetuned models

Bi-encoder:
```bash
python retrieve/retrieve_grounding.py \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_bi.json \
    --enc bi \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --hard_neg_comp
```

Cross-encoder:
```bash
python retrieve/retrieve_grounding.py \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_cross.json \
    --enc cross \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --hard_neg_comp
```

DPR:
```bash
python retrieve/retrieve_grounding.py \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_dpr.json \
    --enc dpr \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --hard_neg_comp
```

NSP:
```bash
python retrieve/retrieve_grounding.py \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_nsp.json \
    --enc nsp \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --hard_neg_comp
```

#### Finetuned persona model

Cross-encoder:
```bash
python retrieve/retrieve_grounding.py \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_finetuned_cross.json \
    --enc cross \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --persona_finetune_model finetuned_persona_cross \
    --hard_neg_comp
```

Bi-encoder:
```bash
python retrieve/retrieve_grounding.py \
    --input_path data/lines_test_public.json \
    --output_path data/lines_aug_test_finetuned_bi.json \
    --enc bi \
    --method sbert \
    --p_thres 0.5 \
    --permutate_persona_knowledge \
    --persona_reselect \
    --persona_aug \
    --persona_finetune_model finetuned_persona_bi \
    --hard_neg_comp
```

### 4. Prepare Generation Input

```bash
python retrieve/gen_input_prep.py \
    --input_path data/lines_aug_test_finetuned_cross.json \
    --output_path data/new_aug_test.json
```

### 5. Prepare Submission

```bash
python submit_proc.py \
    --ground_in data/lines_aug_test_finetuned_cross.json \
    --ground_out answer_grounding.json
```

### Evaluation

[Evaluation](https://codalab.lisn.upsaclay.fr/competitions/3754#results)

## Cite as

```
@inproceedings{oh2023pkicr,
      title={PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue}, 
      author={Minsik Oh and Joosung Lee and Jiwei Li and Guoyin Wang},
      year={2023},
      booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      publisher={Association for Computational Linguistics},
      url={https://arxiv.org/abs/2302.06674}
}
```
