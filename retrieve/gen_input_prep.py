import json
import logging
from argparse import ArgumentParser
from pprint import pformat


def run():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    logger.info("Gen Input Prep Mode")
    logger.info(f"File in : {args.input_path}")
    out_json = []
    with open(args.input_path, 'r') as in_f:
        for line in in_f:
            dialog = json.loads(line)
            for utt in dialog["utterance"]:
                q_key = [key for key in utt.keys() if key.startswith("dialog")]
                dial_key = q_key[0]
                # No History Mode
                utt[dial_key] = [utt[dial_key], "There is no gold."]

                utt["knowledge_candidates"] = utt["knowledge_candidate"]
                utt["knowledge_answer_index"] = utt["knowledge_answer_idx"]
                del utt["knowledge_answer_idx"]
                del utt["knowledge_candidate"]
                utt["filtered_triple_candidates"] = [[], [], [], [], []]
                utt["entities_in_kg"] = [[{}], [{}], [{}], [{}], [{}]]
            out_json.append(dialog)

    print(f"Dialog count : {len(out_json)}")

    with open(args.output_path, 'w') as out_f:
        out_f.write(json.dumps(out_json))

    logger.info("Line Write Complete")
    logger.info(f"File out : {args.output_path}")


if __name__ == "__main__":
    run()
