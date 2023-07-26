# %%
import pandas as pd
from pypdfium2 import PdfDocument

csv_path = "datasets/apple-vqg/gold_standard/Goldstandard.csv"

def gold_standard_to_apple_vqg(csv_path: str, slide_path: str) -> None:
    pass
    # %%
    data_df = pd.read_csv(csv_path)\
        .dropna(subset=["Question"])\
        .drop(columns=["PDF-Name", "Marked for processing", "Comment", "Includes Image Data"])

    data_df["Page Number"] = data_df["Page Number"].astype(int)


