import os
import pandas as pd

import functions as funs

train_set = f"..{os.sep}Data{os.sep}release{os.sep}train.jsonl"
dev_set = f"..{os.sep}Data{os.sep}release{os.sep}dev.jsonl"
test_set = f"..{os.sep}Data{os.sep}release{os.sep}test.jsonl"

# load json files and convert them to dataframes to load faster next time
# train_df = funs.json_to_df(train_set,"train")
# dev_df = funs.json_to_df(dev_set,"dev")
# test_df = funs.json_to_df(test_set,"test")

# load saved dataframes 
df_dir = f"..{os.sep}Data{os.sep}DataFrames"
# train_df = pd.read_csv(os.path.join(df_dir,"train_set.csv"))
dev_df = pd.read_csv(os.path.join(df_dir,"dev_set.csv"))
# test_df = pd.read_csv(os.path.join(df_dir,"test_set.csv"))

# print(dev_df.columns)
# dev_texts = dev_df["text"].values
# dev_sums = dev_df["summary"].values

dev_data = funs.text_processing(dev_df,"dev")

