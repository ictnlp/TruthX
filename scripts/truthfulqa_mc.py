import argparse
import os
import torch
import json
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
import warnings
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


from utils import load_df_from_tsv, read_csv, load_questions, save_questions
from fastchat.model import add_model_args
from llm import llm

from TruthfulQA.truthfulqa.utilities import (
    format_prompt,
    format_prompt_with_answer_strings,
    split_multi_answer,
    format_best,
    find_start,
)
from TruthfulQA.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL
from TruthfulQA.truthfulqa.models import set_columns, MC_calcs
from TruthfulQA.truthfulqa.evaluate import format_frame


@torch.inference_mode()
def main(args):

    questions = load_questions("data/TruthfulQA.csv")
    out_dir = Path(args.output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    lm = llm(args)
    tag = lm.name
    mc_results = run_mc_probs(
        questions, tag, lm, out_file=out_dir.joinpath("answer.jsonl")
    )
    save_questions(mc_results, out_dir.joinpath("answer.csv"))
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(
        columns={"level_0": "Model", "level_1": "Metric", 0: "Value"}
    )
    results = pd.pivot_table(results, "Value", "Model", "Metric")
    results.to_csv(out_dir.joinpath("metrics.csv"))


def run_mc_probs(frame, tag, lm, device=None, cache_dir=None, out_file=None):

    set_columns(tag, frame)

    with torch.no_grad():
        with open(out_file, "w") as file:
            for idx in tqdm.tqdm(frame.index):
                if pd.isnull(frame.loc[idx, "{0} lprob max".format(tag)]):

                    # check that answer exists
                    if pd.isnull(frame.loc[idx, INCORRECT_COL]):
                        warnings.warn(
                            "References missing for {0}!".format(idx), stacklevel=2
                        )
                        continue
                    if not len(frame.loc[idx, INCORRECT_COL]):
                        warnings.warn(
                            "References missing for {0}!".format(idx), stacklevel=2
                        )
                        continue

                    # reference answers
                    ref_best = format_best(frame.loc[idx, BEST_COL])
                    ref_true = split_multi_answer(frame.loc[idx, ANSWER_COL])
                    ref_false = split_multi_answer(frame.loc[idx, INCORRECT_COL])

                    scores_true = []
                    scores_false = []

                    for temp_ans in ref_true:
                        log_probs = lm.get_lprobs(
                            frame.loc[idx, "Question"], temp_ans, temperature=1.0
                        )
                        scores_true.append(log_probs)

                    for temp_ans in ref_false:
                        log_probs = lm.get_lprobs(
                            frame.loc[idx, "Question"], temp_ans, temperature=1.0
                        )
                        scores_false.append(log_probs)

                    res = {
                        "id": idx,
                        "Question": frame.loc[idx, "Question"],
                        "ref_best": ref_best,
                        "ref_true": ref_true,
                        "ref_false": ref_false,
                        "scores_true": scores_true,
                        "scores_false": scores_false,
                    }
                    json.dump(res, file, ensure_ascii=False)
                    file.write("\n")
                    MC_calcs(
                        tag, frame, idx, scores_true, scores_false, ref_true, ref_best
                    )

    if device:
        torch.cuda.empty_cache()

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    parser.add_argument("--output-dir", type=str, default="exp")
    parser.add_argument("--fewshot-prompting", type=bool, default=False)
    args = parser.parse_args()

    main(args)
