import pandas as pd

# Path to your dataset — update this if your file name is different
file_path = "data/tess_toi.csv"

try:
    # Try to read the CSV file
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
    print("Here are the first 5 rows:\n")
    print(df.head())
except FileNotFoundError:
    print("⚠️ The dataset file wasn't found. Make sure you downloaded it into the 'data' folder.")
