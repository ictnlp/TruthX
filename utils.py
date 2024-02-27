import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def read_csv(file_name):
    try:
        df = pd.read_csv(file_name)

        # 返回读取的DataFrame
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def load_df_from_tsv(path: Union[str, Path], sep="\t") -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep=sep,
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def load_tsv_to_dicts(path: Union[str, Path]) -> List[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        rows = [dict(e) for e in reader]
    return rows


def load_questions(filename="questions.csv"):
    """Loads csv of questions into a pandas dataframe"""

    questions = pd.read_csv(filename)
    questions.dropna(axis=1, how="all", inplace=True)  # drop all-null columns

    return questions


def save_questions(questions, filename="answers.csv"):
    """Saves dataframe of questions (with model answers) to csv"""

    questions.to_csv(filename, index=False)
