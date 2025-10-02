
import pandas as pd
import difflib
from sklearn.model_selection import train_test_split
from sparse_retrieval import TFIDFRetriever, BM25Retriever
from preprocess import load_code_snippets, preprocess

def find_best_match(query_code, code_snippets_df):
    """Finds the best match for a code snippet in the corpus."""
    best_ratio = 0
    best_match_id = -1
    for _, row in code_snippets_df.iterrows():
        ratio = difflib.SequenceMatcher(None, query_code, row['code']).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_id = row['code_id']
    return best_match_id, best_ratio

def evaluate(retriever, queries_df, code_snippets_df):
    """Evaluates a retriever on a set of queries."""
    recall_at_10 = 0
    for _, row in queries_df.iterrows():
        query = row['query']
        true_code_id = row['matched_code_id']
        if true_code_id != -1:
            top_k_indices = retriever.retrieve(query, k=10)
            top_k_code_ids = retriever.documents.iloc[top_k_indices]['code_id'].tolist()
            if true_code_id in top_k_code_ids:
                recall_at_10 += 1
    return recall_at_10 / len(queries_df)

if __name__ == '__main__':
    # 載入資料
    code_snippets_df = load_code_snippets('code_snippets.csv')
    processed_df = preprocess(code_snippets_df)
    train_queries_df = pd.read_csv('train_queries.csv')

    # Create validation set by matching code snippets
    val_df, _ = train_test_split(train_queries_df, test_size=0.8, random_state=42) # Use 20% for validation
    matched_code_ids = []
    for _, row in val_df.iterrows():
        matched_id, ratio = find_best_match(row['code'], code_snippets_df)
        if ratio > 0.9:
            matched_code_ids.append(matched_id)
        else:
            matched_code_ids.append(-1)
    val_df['matched_code_id'] = matched_code_ids

    # --- Sparse Models ---
    print("--- Evaluating Sparse Models ---")
    tfidf_retriever = TFIDFRetriever(processed_df)
    bm25_retriever = BM25Retriever(processed_df)

    tfidf_recall = evaluate(tfidf_retriever, val_df, code_snippets_df)
    bm25_recall = evaluate(bm25_retriever, val_df, code_snippets_df)

    print(f"TF-IDF Recall@10: {tfidf_recall:.4f}")
    print(f"BM25 Recall@10: {bm25_recall:.4f}")