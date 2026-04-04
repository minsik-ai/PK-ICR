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

    logger.info("Line Split Mode")
    logger.info(f"File in : {args.input_path}")
    with open(args.input_path, 'r') as in_f:
        in_json = json.load(in_f)

    print(f"Dialog count : {len(in_json)}")

    with open(args.output_path, 'w') as out_f:
        for dialog in in_json:
            out_f.write(json.dumps(dialog))
            out_f.write("\n")

    logger.info("Line Write Complete")
    logger.info(f"File out : {args.output_path}")


if __name__ == "__main__":
    run()
