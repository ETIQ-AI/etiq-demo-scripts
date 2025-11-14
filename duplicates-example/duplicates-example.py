import pandas as pd
from datasets import load_dataset
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load a small tabular dataset from HuggingFace datasets
dataset = load_dataset("mstz/adult", split="train")

# Convert to pandas DataFrame
df = dataset.to_pandas()

# Introduce duplicate rows
# Randomly select rows to duplicate
num_duplicates = 15  # Number of duplicate rows to add
duplicate_indices = np.random.choice(df.index, size=num_duplicates, replace=True)
duplicated_rows = df.loc[duplicate_indices]

# Concatenate original data with duplicates
df_with_duplicates = pd.concat([df, duplicated_rows], ignore_index=True)

# Shuffle the dataset so duplicates aren't all at the end
df_with_duplicates = df_with_duplicates.sample(frac=1, random_state=42).reset_index(drop=True)

# Check for duplicates
num_duplicate_rows = df_with_duplicates.duplicated().sum()