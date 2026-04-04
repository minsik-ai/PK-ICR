import collections
import logging
from argparse import ArgumentParser
from pprint import pformat

import torch
from torch import nn
from transformers import AutoTokenizer, MobileBertForNextSentencePrediction

from utils.data_utils import choose_top_idx_tfidf
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import json

bert_token = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
bert_nsp = MobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")
bert_nsp.to("cuda")

bi = SentenceTransformer("msmarco-distilbert-base-tas-b", device="cuda")
cross_knowledge = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda")
bi_persona_norm = None
cross_persona_norm = None

dpr_q = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base", device="cuda")
dpr_ctx = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base", device="cuda")

def choose_top_idx_sbert(knowledge, question, enc):
    results = []
    if enc == "bi":
        q_emb = bi.encode(question, convert_to_tensor=True, show_progress_bar=False)
        c_embs = bi.encode(knowledge, convert_to_tensor=True, show_progress_bar=False)
        for i, c_emb in enumerate(c_embs):
            score = util.dot_score(q_emb, c_emb).cpu().numpy()[0][0]
            results.append((i, score))
    elif enc == "dpr":
        dpr_ks = [f"Title [SEP] {k}"for k in knowledge]
        q_emb = dpr_q.encode(question, convert_to_tensor=True, show_progress_bar=False)
        c_embs = dpr_ctx.encode(dpr_ks, convert_to_tensor=True, show_progress_bar=False)
        for i, c_emb in enumerate(c_embs):
            score = util.dot_score(q_emb, c_emb).cpu().numpy()[0][0]
            results.append((i, score))
    elif enc == "cross":
        pairs = [[question, cand] for cand in knowledge]
        model = cross_knowledge
        scores = model.predict(pairs, activation_fct=nn.Sigmoid(), show_progress_bar=False)
        results = [(i, score) for i, score in enumerate(scores)]
    elif enc == "nsp":
        pairs = [[question, cand] for cand in knowledge]
        q_sents = [p[0] for p in pairs]
        a_sents = [p[1] for p in pairs]
        encoding = bert_token(q_sents, a_sents, return_tensors="pt", padding=True, max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            outs = bert_nsp(**encoding)
            probs = torch.nn.Softmax(dim=1)(outs.logits)
        results = [(i, ps[1]) for i, ps in enumerate(probs)]
    else:
        raise ValueError(enc)

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[0]

def sbert_score(q, a, enc, finetune_model=None):
    global bi_persona_norm, cross_persona_norm
    if enc == "bi":
        if finetune_model and not bi_persona_norm:
            bi_persona_norm = SentenceTransformer(finetune_model, device="cuda")
        q_emb = (bi_persona_norm if finetune_model else bi).encode(q, convert_to_tensor=True, show_progress_bar=False)
        a_emb = (bi_persona_norm if finetune_model else bi).encode(a, convert_to_tensor=True, show_progress_bar=False)
        return util.dot_score(q_emb, a_emb).cpu().numpy()[0][0]
    elif enc == "dpr":
        if finetune_model and not bi_persona_norm:
            bi_persona_norm = SentenceTransformer(finetune_model, device="cuda")
        q_emb = (bi_persona_norm if finetune_model else dpr_q).encode(q, convert_to_tensor=True, show_progress_bar=False)
        a_emb = (bi_persona_norm if finetune_model else dpr_ctx).encode(f"Title [SEP] {a}", convert_to_tensor=True, show_progress_bar=False)
        return util.dot_score(q_emb, a_emb).cpu().numpy()[0][0]
    elif enc == "cross":
        if finetune_model and not cross_persona_norm:
            cross_persona_norm = CrossEncoder(finetune_model, device="cuda")
        score = (cross_persona_norm if finetune_model else cross_knowledge).predict([[q, a]], activation_fct=nn.Sigmoid(), show_progress_bar=False)
        return score[0]
    elif enc == "nsp":
        encoding = bert_token(q, a, return_tensors="pt", padding=True, max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            outs = bert_nsp(**encoding)
            probs = torch.nn.Softmax(dim=1)(outs.logits)
        return probs[0][1]
    else:
        raise ValueError(enc)

def run():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file (must contain 'lines' in filename)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output augmented JSONL file")
    parser.add_argument("--enc", type=str, required=True, help="Encoder type for retrieval: cross, bi, nsp, or dpr")
    parser.add_argument("--method", type=str, required=False, help="Retrieval method: 'tfidf' for TF-IDF, otherwise uses SBERT encoder specified by --enc")
    parser.add_argument("--persona_finetune_model", type=str, required=False, help="Path to finetuned persona model (not supported with nsp encoder)")
    parser.add_argument("--persona_aug", action='store_true', help="Augment persona candidates by appending the question text")
    parser.add_argument("--question_only", action='store_true', help="Select persona using question similarity only, ignoring knowledge (mutually exclusive with --persona_aug)")
    parser.add_argument("--permutate_persona_knowledge", action='store_true', help="Score all persona-knowledge pairs and select the best joint match")
    parser.add_argument("--persona_reselect", action="store_true", help="Re-rank personas against the selected knowledge using --persona_finetune_model")
    parser.add_argument("--hard_neg_comp", action="store_true", help="Compute relative rank statistics of question vs persona similarities for analysis")
    parser.add_argument("--k_thres", type=float, default=0.0, help="Knowledge similarity threshold (default: 0.0)")
    parser.add_argument("--p_thres", type=float, default=0.0, help="Persona similarity threshold for grounding label (default: 0.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    assert args.enc in ["cross", "bi", "nsp", "dpr"]
    if args.enc == "nsp":
        assert args.persona_finetune_model is None

    logger.info("ADD GROUNDING TFIDF, only add lines")
    logger.info(f"File in : {args.input_path}")
    assert "lines" in args.input_path
    assert not (args.question_only and args.persona_aug)

    p_thres = args.p_thres
    p_rel = 0
    p_nonrel = 0

    with open(args.input_path, 'r') as in_f:
        with open(args.output_path, 'w') as out_f:
            rel_rank_bins = collections.defaultdict(int)
            rel_rank_bins_half = collections.defaultdict(int)
            q_sim_avg = 0
            for line in tqdm(in_f):
                in_json = json.loads(line)
                utt_jsons = in_json["utterance"]
                for utt_json in utt_jsons:
                    q_key = [key for key in utt_json.keys() if key.startswith("dialog")]
                    question = utt_json[q_key[0]]

                    k_cands = utt_json["knowledge_candidate"]
                    p_cands = utt_json["persona_candidate"]

                    # Context Augmentation
                    p_cands_aug = [f"{p} {question}" for p in p_cands]

                    p_targets = p_cands_aug if args.persona_aug else p_cands

                    if args.permutate_persona_knowledge:
                        k_list = []
                        for p_idx, p_cand in enumerate(p_targets):
                            k_idx, k_sim = choose_top_idx_tfidf(k_cands, p_cand) if args.method == "tfidf" \
                                else choose_top_idx_sbert(k_cands, p_cand, enc=args.enc)
                            k_list.append(((p_idx, k_idx), k_sim))
                        sorted_k_list = sorted(k_list, key=lambda val: val[1], reverse=True)
                        choice = sorted_k_list[0]
                        p_idx, k_idx = choice[0]
                        if not args.persona_reselect:
                            p_sim = choice[1]
                        else:
                            p_s = []
                            for p_idx, p_cand in enumerate(p_targets):
                                p_sim = sbert_score(p_cand, k_cands[k_idx], enc=args.enc, finetune_model=args.persona_finetune_model)
                                p_s.append((p_idx, p_sim))
                            sorted_p_list = sorted(p_s, key=lambda val: val[1], reverse=True)
                            p_idx, p_sim = sorted_p_list[0]
                            if args.hard_neg_comp:
                                q_sim = sbert_score(question, k_cands[k_idx], enc=args.enc, finetune_model=args.persona_finetune_model)
                                p_s_new = p_s.copy()

                                p_s.append((-1, q_sim))
                                sorted_p_list = sorted(p_s, key=lambda val: val[1], reverse=True)
                                non_persona_ind = next(i for i, v in enumerate(sorted_p_list) if v[0] == -1)
                                rel_rank = non_persona_ind - 1
                                rel_rank_bins[rel_rank] += 1
                                q_sim_avg += q_sim

                                # Half
                                p_s_new.append((-1, 0.5))
                                sorted_p_new_list = sorted(p_s_new, key=lambda val: val[1], reverse=True)
                                half_ind = next(i for i, v in enumerate(sorted_p_new_list) if v[0] == -1)
                                rel_half_rank = half_ind - 1
                                rel_rank_bins_half[rel_half_rank] += 1
                                print(f"Relative Rank : {rel_rank}, Half Rank : {rel_half_rank}, Similarity : {q_sim}")
                    else:
                        k_idx, k_sim = choose_top_idx_tfidf(k_cands, question) if args.method == "tfidf" \
                            else choose_top_idx_sbert(k_cands, question, enc=args.enc)
                        if not args.question_only:
                            p_idx, p_sim = choose_top_idx_tfidf(p_cands, k_cands[k_idx]) if args.method == "tfidf" \
                                else choose_top_idx_sbert(p_targets, k_cands[k_idx], enc=args.enc)
                        else:
                            tqdm.write("Persona QUESTION ONLY")
                            p_s = []
                            for p_idx, p in enumerate(p_targets):
                                p_sim = sbert_score(question, p, enc=args.enc, finetune_model=args.persona_finetune_model)
                                p_s.append((p_idx, p_sim))
                            sorted_p_list = sorted(p_s, key=lambda val: val[1], reverse=True)
                            p_idx, p_sim = sorted_p_list[0]

                    utt_json["knowledge_answer_idx"] = k_idx
                    utt_json["persona_grounding"] = [("true" if (i == p_idx and p_sim > p_thres) else "false") for i in range(len(p_targets))]

                    if p_sim > p_thres:
                        p_rel += 1
                    else:
                        p_nonrel += 1

                    tqdm.write("")
                    tqdm.write(f"Knowledge Candidates : {k_cands}")
                    tqdm.write("")
                    tqdm.write(f"Persona Candidates : {p_cands}")
                    tqdm.write("")
                    tqdm.write(f"Question : {question}")
                    tqdm.write("")
                    tqdm.write(f"Knowledge : {k_cands[k_idx]}, {k_sim}")
                    tqdm.write("")
                    tqdm.write(f"Persona : {p_cands[p_idx]}, {p_sim}")

                    tqdm.write("--------")
                out_f.write(json.dumps(in_json))
                out_f.write("\n")

    print("Lines Added")
    print(f"Persona rel : {p_rel}, non-rel : {p_nonrel} from thres : {p_thres}")
    if args.hard_neg_comp:
        print(f"Rel rank bins : {rel_rank_bins}")
        print(f"Rel rank half bins : {rel_rank_bins_half}")
        rel_rank_avg = 0
        rel_rank_cnt = 0
        for rank, cnt in rel_rank_bins.items():
            rel_rank_avg += rank * cnt
            rel_rank_cnt += cnt
        rel_rank_avg /= rel_rank_cnt
        rel_half_rank_avg = 0
        for rank, cnt in rel_rank_bins_half.items():
            rel_half_rank_avg += rank * cnt
        rel_half_rank_avg /= rel_rank_cnt

        q_sim_avg /= rel_rank_cnt
        print(f"Rel rank avg : {rel_rank_avg}, half rank avg : {rel_half_rank_avg}")
        print(f"Q sim avg : {q_sim_avg}")

if __name__ == "__main__":
    run()
