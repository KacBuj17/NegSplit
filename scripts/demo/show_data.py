import pandas as pd

from config import TRAIN_DATA

ROWS = 5
SHORT = ["split", "image_file", "obj_present", "obj_absent", "mcq_template"]
SENTENCES = ["caption_pos", "caption_neg_true", "caption_neg_false", "mcq_correct", "mcq_d1", "mcq_d2", "mcq_d3"]


def main():
    df = pd.read_parquet(TRAIN_DATA)
    pd.set_option("display.width", 200)
    print(f"{TRAIN_DATA.name}: {len(df)} rows, columns: {', '.join(df.columns)}")
    print(df.split.value_counts().to_string(), "\n")
    print(df[SHORT].head(ROWS).to_string(), "\n")
    for i, row in df.head(ROWS).iterrows():
        print(f"--- row {i} ---")
        for col in SENTENCES:
            print(f"{col:18} {row[col]}")
        print()


if __name__ == "__main__":
    main()
