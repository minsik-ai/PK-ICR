import logging
import math
from argparse import ArgumentParser
from pprint import pformat

from datasets import Dataset
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn
from torch.utils.data import DataLoader

from utils.data_utils import choose_top_idx_tfidf
from tqdm import tqdm
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer, losses
import json

BATCH_SIZE = 32
NUM_EPOCHS = 2

# To be used in conjunction with retrieve_grounding.py

# Bi Encoder
bi = SentenceTransformer("msmarco-distilbert-base-tas-b", device="cuda")

# Cross Encoder
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda")

def sbert_score(q, a):
    score = cross.predict([[q, a]], activation_fct=nn.Sigmoid(), show_progress_bar=False)
    return score[0]

def run():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--question_only", action="store_true", help="Question only")
    parser.add_argument("--persona_aug", action="store_true", help="Augment Persona")
    parser.add_argument("--hard_neg_only", action='store_true', help="Reduce number of negatives")
    parser.add_argument("--bi_enc", action="store_true", help="Whether to use bi encoder.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    logger.info(f"File in : {args.input_path}")
    assert "lines" in args.input_path

    # Dataset creation
    input_dict = {
        "texts":[],
        "label":[]
    }
    print("data creating")
    with open(args.input_path, 'r') as in_f:
        for line in tqdm(in_f):
            # print(line)
            # print("Start dialog")
            in_json = json.loads(line)
            utt_jsons = in_json["utterance"]
            for utt_json in utt_jsons:
                q_key = [key for key in utt_json.keys() if key.startswith("dialog")]
                question = utt_json[q_key[0]][-2]

                # import pdb; pdb.set_trace()
                k_cands = utt_json["knowledge_candidates"]
                p_cands = utt_json["persona_candidate"]

                # Context Augmentation
                p_cands_aug = [f"{p} {question}" for p in p_cands]
                # print("Start choosing")
                k_ans_idx = utt_json["knowledge_answer_index"]
                p_grounds = utt_json["persona_grounding"]

                p_target = p_cands_aug if args.persona_aug else p_cands
                if args.question_only:
                    for p_idx, p_cand in enumerate(p_target):
                        input_dict["texts"].append([question, p_cand])
                        input_dict["label"].append(1.0 if p_grounds[p_idx] else 0.0)
                # TODO : Reduce negatives? Contrastive loss fixed batch?
                else:
                    for p_idx, p_cand in enumerate(p_target):
                        for k_idx, k_cand in enumerate(k_cands):
                            if args.hard_neg_only and k_idx != k_ans_idx:
                                continue
                            x_value = [p_cand, k_cand]
                            y_value = 1.0 if p_grounds[p_idx] and k_idx == k_ans_idx else 0.0
                            input_dict["texts"].append(x_value)
                            input_dict["label"].append(y_value)
    print("data created")
    print("prepping")

    dataset = Dataset.from_dict(input_dict)
    dataset = dataset.train_test_split(test_size=0.1)

    pos_train = 0
    neg_train = 0
    pos_val = 0
    neg_val = 0
    train_examples = []
    for line in tqdm(dataset["train"]):
        train_examples.append(InputExample(texts=line['texts'], label=line['label']))
        if line['label'] == 1.0:
            pos_train += 1
            # print(line['texts'])
        else:
            neg_train += 1
    print(f"train stats : pos - {pos_train}, neg - {neg_train}")

    train_dl = DataLoader(train_examples, batch_size=BATCH_SIZE)

    test_examples = []
    for line in tqdm(dataset["test"]):
        # import pdb; pdb.set_trace()
        test_examples.append(InputExample(texts=line['texts'], label=line['label']))
        if line['label'] == 1.0:
            pos_val += 1
            # print(line['texts'])
        else:
            neg_val += 1
    print(f"val stats : pos - {pos_val}, neg - {neg_val}")

    eval_steps = 1000
    warmup_steps = math.ceil(len(train_dl) * NUM_EPOCHS * 0.1)

    print("training starts")
    if not args.bi_enc:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_examples, name='test')
        cross.fit(train_dataloader=train_dl,
                  evaluator=evaluator,
                  evaluation_steps=eval_steps,
                  epochs=NUM_EPOCHS,
                  warmup_steps=warmup_steps,
                  output_path=args.output_path)
    else:
        train_loss = losses.CosineSimilarityLoss(bi)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='test')
        bi.fit(train_objectives=[(train_dl, train_loss)],
               evaluator=evaluator,
               epochs=NUM_EPOCHS,
               warmup_steps=warmup_steps,
               output_path=args.output_path)

if __name__ == "__main__":
    run()
