
import pandas as pd
from sklearn.model_selection import train_test_split
from sparse_retrieval import TFIDFRetriever, BM25Retriever
from preprocess import preprocess

def evaluate_on_self(retriever, queries_df, query_expansion=False):
    """Evaluates a retriever on a self-contained dataset."""
    recall_at_10 = 0
    for _, row in queries_df.iterrows():
        query = row['query']
        true_code_id = row['code_id']
        top_k_indices = retriever.retrieve(query, k=10, query_expansion=query_expansion)
        top_k_code_ids = retriever.documents.iloc[top_k_indices]['code_id'].tolist()
        if true_code_id in top_k_code_ids:
            recall_at_10 += 1
    return recall_at_10 / len(queries_df)


if __name__ == '__main__':
    # Load train_queries.csv
    train_queries_df = pd.read_csv('train_queries.csv')

    # Create a self-contained corpus and test set
    corpus_df = train_queries_df.copy()
    corpus_df['code_id'] = range(len(corpus_df))
    test_df = corpus_df.copy()

    # Preprocess the corpus (unigrams)
    processed_corpus_unigram = preprocess(corpus_df.copy(), n_gram_range=(1, 1))

    # --- Unigrams without Query Expansion ---
    print("--- Evaluating with Unigrams (No Query Expansion) ---")
    tfidf_retriever_no_qe = TFIDFRetriever(processed_corpus_unigram)
    bm25_retriever_no_qe = BM25Retriever(processed_corpus_unigram)
    tfidf_recall_no_qe = evaluate_on_self(tfidf_retriever_no_qe, test_df)
    bm25_recall_no_qe = evaluate_on_self(bm25_retriever_no_qe, test_df)
    print(f"TF-IDF Recall@10: {tfidf_recall_no_qe:.4f}")
    print(f"BM25 Recall@10: {bm25_recall_no_qe:.4f}")

    # --- Unigrams with Query Expansion ---
    print("\n--- Evaluating with Unigrams (With Query Expansion) ---")
    tfidf_retriever_qe = TFIDFRetriever(processed_corpus_unigram)
    bm25_retriever_qe = BM25Retriever(processed_corpus_unigram)
    tfidf_recall_qe = evaluate_on_self(tfidf_retriever_qe, test_df, query_expansion=True)
    bm25_recall_qe = evaluate_on_self(bm25_retriever_qe, test_df, query_expansion=True)
    print(f"TF-IDF Recall@10: {tfidf_recall_qe:.4f}")
    print(f"BM25 Recall@10: {bm25_recall_qe:.4f}")
