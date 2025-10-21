import pandas as pd

def read_ds(ds_name: str):

    df = pd.read_csv(
        ds_name,
        header=None,
        engine='python',
        on_bad_lines='skip',
    )
    if ds_name == "downloads/train.csv":
        df.rename(columns={0: "util", 1: "browser"}, inplace=True)
    elif ds_name == "downloads/test.csv":
        df.rename(columns={0: "browser"}, inplace=True)


    return df

