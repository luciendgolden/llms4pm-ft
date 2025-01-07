from typing import Optional
from datasets.arrow_dataset import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import ast
import sys

def remove_duplicates(pair_df):
    """
    Removes duplicate rows in the DataFrame based on specific columns.
    Additional columns like 'trace', 'eventually_follows', and 'prefix'
    are considered if present in the DataFrame.
    """
    columns = ["revision_id", "model_id", "unique_activities"]
    if "trace" in pair_df.columns:
        columns.append("trace")
    if "eventually_follows" in pair_df.columns:
        columns.append("eventually_follows")
    if "prefix" in pair_df.columns:
        columns.append("prefix") # update this to consider multiple possible options?
    pair_df = pair_df.drop_duplicates(subset=columns)
    return pair_df


def setify(x: str):
    """
    Converts a string representation of a set into an actual Python set.
    Ensures the result is a set, otherwise raises an AssertionError.
    """
    set_: set[str] = eval(x)
    assert isinstance(set_, set), f"Conversion failed for {x}"
    return set_

def stratified_sample(df, label_col, frac, random_state=42) -> pd.DataFrame:
    """
    Performs stratified sampling to reduce the dataset size by a given fraction.
    """
    stratified_df, _ = train_test_split(
        df, 
        stratify=df[label_col], 
        test_size=1-frac, 
        random_state=random_state
    )
    return stratified_df

def split_by_model(df, task, pkl_path="data/train_val_test.pkl", frac: Optional[float] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train, validation, and test subsets based on IDs.
    Only includes rows with more than one unique activity.
    """
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    df["num_unique_activities"] = df["unique_activities"].apply(len)

    # only keep rows with at least 2 activities
    df = df[df["num_unique_activities"] > 1]

    # Load pre-defined train/val/test IDs
    with open(pkl_path, "rb") as file:
        train_ids, val_ids, test_ids = pickle.load(file)
        
    # Split the DataFrame into train, validation, and test sets
    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]

    # Perform stratified sampling for tasks with binary labels
    # if frac is not None, otherwise perform random sampling
    if frac is not None and 0 < frac < 1:
        if task in ["TRACE_ANOMALY", "OUT_OF_ORDER"]:
            train_df = stratified_sample(train_df, label_col="ds_labels", frac=frac)
            val_df = stratified_sample(val_df, label_col="ds_labels", frac=frac)
            test_df = stratified_sample(test_df, label_col="ds_labels", frac=frac)
        else:
            # For S-NAP, perform random sampling
            train_df = train_df.sample(frac=frac, random_state=42)
            val_df = val_df.sample(frac=frac, random_state=42)
            test_df = test_df.sample(frac=frac, random_state=42)

    return train_df, val_df, test_df

def next_label(row):
    sorted_acts = sorted(list(row["unique_activities"]))
    if row["next"] not in sorted_acts:
        return 0
    return sorted_acts.index(row["next"]) + 1

def load_dataset(file_name: str, task: str, frac: Optional[float]) -> tuple[Dataset, Dataset, Dataset]:
    """
    Dynamically loads and processes a dataset based on the file name and task.
    """
    df = pd.read_csv(file_name)

    if task == "TRACE_ANOMALY":
        # T-SAD
        df["ds_labels"] = (~df["anomalous"]).astype(bool)  # Invert labels
        df["trace"] = df["trace"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["trace"] = df["trace"].apply(lambda x: tuple(x))
        df = remove_duplicates(df)
        df["unique_activities"] = df["unique_activities"].apply(setify)
        columns = ["model_id", "revision_id", "unique_activities", "trace", "ds_labels"]
        df = df.loc[:, columns]

    elif task == "OUT_OF_ORDER":
        # A-SAD
        df["ds_labels"] = (~df["out_of_order"]).astype(bool)  # Invert labels
        df = remove_duplicates(df)
        df["unique_activities"] = df["unique_activities"].apply(setify)
        columns = ["model_id", "revision_id", "unique_activities", "ds_labels", "eventually_follows"]
        df = df.loc[:, columns]

    elif task == "NEXT_ACTIVITY":
        # S-NAP
        df["prefix"] = df["prefix"].apply(lambda x: tuple(x))
        df = remove_duplicates(df)
        df["prefix"] = df["prefix"].apply(lambda x: list(x))
        df["unique_activities"] = df["unique_activities"].apply(setify)
        mask = ~(df["next"] == "[END]")
        df = df[mask]
        df["ds_labels"] = df.apply(next_label, axis=1)
        columns = ["model_id", "revision_id", "trace", "prefix", "next", "unique_activities", "ds_labels"]
        df = df.loc[:, columns]
    else:
        raise ValueError(f"Unsupported task: {task}")

    # split into train/val/test
    train_df, val_df, test_df = split_by_model(df, task=task, frac=frac)

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )

def print_stats(split_name, split_df):
        total_samples = len(split_df)
        label_counts = pd.Series(split_df['ds_labels']).value_counts()
        label_percentages = label_counts / total_samples * 100
        print(f"--- {split_name} Split Statistics ---")
        print(f"Total samples: {total_samples}")
        print(f"Label distribution:\n{label_counts}")
        print(f"Label percentages:\n{label_percentages.round(2)}%\n")

def format_time(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"