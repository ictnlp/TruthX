import argparse
import os
import torch
import json
from pathlib import Path
import tqdm
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


from fastchat.model import add_model_args
from llm import llm
from utils import read_csv


@torch.inference_mode()
def main(args):

    df = read_csv("data/TruthfulQA.csv")
    data = list(df.T.to_dict().values())

    # Load model
    lm = llm(args)

    out_dir = Path("/".join(args.output_file.split("/")[:-1]))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(args.output_file, "w") as file:
        i = 0
        for item in tqdm.tqdm(data):
            question = item["Question"]
            answer = lm.tfqa_generate(question)
            res = {
                "id": i,
                "Question": question,
                "Answer": answer,
            }
            json.dump(res, file, ensure_ascii=False)
            print(res)
            file.write("\n")
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    parser.add_argument("--output-file", type=str, default="exp/res.jsonl")
    parser.add_argument("--fewshot-prompting", type=bool, default=False)
    args = parser.parse_args()

    main(args)
