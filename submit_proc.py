import logging
from argparse import ArgumentParser
from pprint import pformat
from utils.data_utils import choose_top_idx_tfidf
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

def run():
    parser = ArgumentParser()
    parser.add_argument("--ground_in", type=str, required=True)
    parser.add_argument("--ground_out", type=str, required=True)
    parser.add_argument("--gen_in", type=str, required=False)
    parser.add_argument("--gen_out", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    with open(args.ground_in, 'r') as in_f:
        with open(args.ground_out, 'w') as out_f:
            # TODO : Consider adding list as post-proc to avoid memory usage
            out_json = []
            for line in tqdm(in_f):
                in_json = json.loads(line)
                dialog_id = in_json["dialogID"]
                utt_jsons = in_json["utterance"]
                line_json = {dialog_id:[]}
                vals = line_json[dialog_id]
                for utt_json in utt_jsons:
                    k_idx = utt_json["knowledge_answer_idx"]
                    p_ground = utt_json["persona_grounding"]

                    item = {}
                    item["pg"] = [1 if b == "true" else 0 for b in p_ground]
                    item["kg"] = k_idx
                    vals.append(item)

                    tqdm.write("")
                    tqdm.write(f"dialog_id : {dialog_id}")
                out_json.append(line_json)
            out_f.write(json.dumps(out_json, indent=4))
    print("Finished : validate ground file.")

    if not args.gen_in:
        print("-------------- DUMMY GEN ---------------")
        with open(args.ground_in, 'r') as in_f:
            with open(args.gen_out, 'w') as out_f:
                # TODO : Consider adding list as post-proc to avoid memory usage
                out_json = []
                for line in tqdm(in_f):
                    in_json = json.loads(line)
                    dialog_id = in_json["dialogID"]
                    utt_jsons = in_json["utterance"]
                    line_json = {dialog_id: []}
                    vals = line_json[dialog_id]
                    for utt_json in utt_jsons:
                        k_idx = utt_json["knowledge_answer_idx"]
                        k_cands = utt_json["knowledge_candidate"]
                        item = {}

                        # DO NOT USE CAND DIRECTLY NOW
                        item["generation"] = ""
                        vals.append(item)

                        tqdm.write("")
                        tqdm.write(f"dialog_id : {dialog_id}")
                    out_json.append(line_json)
                out_f.write(json.dumps(out_json, indent=4))
    else:
        print("-------------- REAL GEN ---------------")
        with open(args.gen_in, 'r') as in_f:
            with open(args.gen_out, 'w') as out_f:
                # Single Line
                for line in in_f:
                    out_f.write(line)

    print("Finished : validate gen file.")

if __name__ == "__main__":
    run()