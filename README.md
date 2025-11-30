# PK-ICR: Persona-Knowledge Interactive Multi-Context Retrieval for Grounded Dialogue
This repository contains training and evaluation code for our EMNLP 2023 paper [Persona-Knowledge Interactive Multi-Context Retrieval for Grounded Dialogue](https://arxiv.org/abs/2302.06674).

## Datasets

To execute the scripts, train and test datasets in JSONL form is required.

```
{ ... } # turn 1
{ ... } # turn 2
...

```

Data is available at following : [train_data](https://drive.google.com/file/d/1YmEW12HqjAjlEfZ05g8VLRux8kyUjdcI/view?usp=sharing), [test data](https://codalab.lisn.upsaclay.fr/my/datasets/download/341d58e9-34a9-4dcc-ab16-0720d11d4b37)

## Replicating Results

```
# Finetune Persona model
sh run_finetune.sh

# Infer both Knowledge and Persona.
sh run_inference.sh
```

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
