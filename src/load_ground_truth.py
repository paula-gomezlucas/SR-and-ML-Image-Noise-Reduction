import os
from io import StringIO
import pandas as pd

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "..", "data", "benchmark", "test.cat")

# Read file, skipping headers
with open(file_path, 'r') as f:
    lines = [line for line in f if not line.startswith('#')]

df = pd.read_csv(StringIO(''.join(lines)), sep=r'\s+')

print(df.head())

output_csv = os.path.join(script_dir, "..", "data", "benchmark", "sextractor_output.csv")
df.to_csv(output_csv, index=False)