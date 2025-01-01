from datasets.arrow_dataset import Dataset
import pandas as pd
import pickle
import ast

def remove_duplicates(pair_df):
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
    set_: set[str] = eval(x)
    assert isinstance(set_, set), f"Conversion failed for {x}"
    return set_


def split_by_model(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    df["num_unique_activities"] = df["unique_activities"].apply(len)

    # only keep rows with at least 2 activities
    df = df[df["num_unique_activities"] > 1]

    with open(f"train_val_test.pkl", "rb") as file:
        train_ids, val_ids, test_ids = pickle.load(file)
    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]

    # TODO or shuffle here?

    return train_df, val_df, test_df


def load_trace_data() -> Dataset:    
    trace_df: pd.DataFrame = pd.read_csv("T_SAD.csv")

    # invert labels because the model will predict whether the trace is correct, and not wrong (=anomalous)
    trace_df["ds_labels"] = (~trace_df["anomalous"]).astype(bool)  # Convert to bool for labels

    # # Parse 'prefix' column safely into lists
    trace_df["trace"] = trace_df["trace"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # drop duplicates - tuple conversion required for standartization
    trace_df.trace = trace_df.trace.apply(lambda x: tuple(x))
    trace_df = remove_duplicates(trace_df)

    # using a set ensures that each activity is listed only once, removing duplicates automatically
    trace_df.unique_activities = trace_df.unique_activities.apply(setify)

    columns = ["model_id", "revision_id", "unique_activities", "trace", "ds_labels"]
    trace_df = trace_df.loc[:, columns]

    train_df, val_df, test_df = split_by_model(trace_df)

    # TODO shuffle each subset?
    
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )


def load_pairs_data(split: str = "train") -> Dataset:
    eval_pairs = pd.read_csv(
        "A_SAD.csv"
    )
    eval_pairs["labels"] = ~(eval_pairs.out_of_order)
    eval_pairs = remove_duplicates(eval_pairs)
    # eval_pairs.trace = eval_pairs.trace.apply(lambda x: tuple(x))
    eval_pairs.unique_activities = eval_pairs.unique_activities.apply(setify)
    columns = [
        "model_id",
        "revision_id",
        "unique_activities",
        "labels",
        "eventually_follows",
    ]
    eval_pairs = eval_pairs.loc[:, columns]
    train, val, test = split_by_model(eval_pairs)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)
    

def convert_next_label(line: dict):
    if not line["next"] == "[END]":
        return list(line["unique_activities"]).index(line["next"]) + 1
    else:
        return 0


def load_next_activity_data(split: str = "train") -> Dataset:
    eval_prefix = pd.read_csv(
        "S_NAP.csv"
    )
    eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: tuple(x))
    eval_prefix = remove_duplicates(eval_prefix)
    eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: list(x))
    # eval_pairs.trace = eval_pairs.trace.apply(lambda x: tuple(x))
    eval_prefix.unique_activities = eval_prefix.unique_activities.apply(setify)
    # eval_prefix is a dataframe and 'unique_activities' and 'next' are columns
    # how do I make the apply work?
    mask = ~(eval_prefix.next == "[END]")
    eval_prefix["labels"] = eval_prefix.apply(convert_next_label, axis=1)
    columns = [
        "model_id",
        "revision_id",
        "trace",
        "prefix",
        "next",
        "unique_activities",
        "labels",
    ]
    eval_prefix = eval_prefix.loc[:, columns]
    eval_prefix = eval_prefix.loc[mask]
    train, val, test = split_by_model(eval_prefix)
    # x = train['unique_activities'].apply(lambda x: len(x)).value_counts()
    # y = val['unique_activities'].apply(lambda x: len(x)).value_counts()
    # z = test['unique_activities'].apply(lambda x: len(x)).value_counts()
    # z_ = pd.concat([x, y, z], axis=1)
    # z_ = pd.concat([x, y, z], axis=0)
    # z_.columns = ["train", "val", "test"]
    # z__ = z_ / z_.sum(0)
    # z__.sort_index().cumsum(0)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)
    

def load_dfg_data(data_dir: str, split: str):
    df = pd.read_csv("S-PMD.csv")
    train, val, test = split_by_model(df, data_dir=data_dir)
    if split == "train":
        df = train
    elif split == "val":
        df = val
    if split == "test":
        df = test
    columns = ["id", "unique_activities", "dfg"]
    df = df.loc[:, columns]
    df["unique_activities"] = df["unique_activities"].apply(lambda ele: setify(ele))
    df["dfg"] = df["dfg"].apply(lambda ele: eval(ele))
    return Dataset.from_pandas(df)

# ? collate_next_activity_pred, preprocess_pt (shifted labels??)

def load_pt_data(data_dir: str, split: str):
    df = pd.read_csv(data_dir + "S-PMD.csv")
    train, val, test = split_by_model(df, data_dir=data_dir)
    if split == "train":
        df = train
    elif split == "val":
        df = val[:16]
    if split == "test":
        df = test[:16]
    columns = ["id", "unique_activities", "pt"]
    df = df.loc[:, columns]
    df["unique_activities"] = df["unique_activities"].apply(lambda ele: setify(ele))
    # df["dfg"] = df["dfg"].apply(lambda ele: eval(ele))
    return Dataset.from_pandas(df)

