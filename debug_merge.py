import pandas as pd

# Load the data
code_snippets_df = pd.read_csv('code_snippets.csv')
train_queries_df = pd.read_csv('train_queries.csv')

# Write the code columns to a file for inspection
with open('code_comparison.txt', 'w', encoding='utf-8') as f:
    for i in range(10):
        f.write(f"--- Train Query {i} ---\n")
        f.write(repr(train_queries_df.iloc[i]['code']))
        f.write("\n\n")

    for i in range(10):
        f.write(f"--- Code Snippet {i} ---\n")
        f.write(repr(code_snippets_df.iloc[i]['code']))
        f.write("\n\n")