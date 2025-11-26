import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def transform_price_to_log(df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({'notRepairedDamage': {'-': np.nan}})
    df["price_log"] = np.log1p(df["price"])
    df = df.drop(columns=["price"])
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
    return df
