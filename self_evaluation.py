import pandas as pd
from sparse_retrieval import TFIDFRetriever, BM25Retriever
from preprocess import preprocess
from tqdm import tqdm

# 由於提供的檔案中沒有用於驗證稀疏檢所器的data，此處利用用於微調密集檢索器的資料train_queries.csv建立一個新的語料庫以及對應的問題-答案集，用於評估當前檢索器的效能


def evaluate(retriever, df, query_expansion=False):
    """
    在完整的資料集上評估檢索器的效能
    資料集同時作為語料庫和查詢集
    """
    recall_at_10 = 0
    # 使用tqdm顯示進度條
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Evaluating {retriever.__class__.__name__}"):
        query = row['query']
        true_code_id = row['code_id']
        
        top_k_indices, _ = retriever.retrieve(query, k=10, query_expansion=query_expansion)
        top_k_code_ids = retriever.documents.iloc[top_k_indices]['code_id'].tolist()
        
        if true_code_id in top_k_code_ids:
            recall_at_10 += 1
            
    return recall_at_10 / len(df)

if __name__ == '__main__':
    # 載入train_queries.csv
    print("Loading train_queries.csv for self-evaluation...")
    queries_df = pd.read_csv('train_queries.csv')

    # 建立一個唯一的code_id作為真實答案
    queries_df['code_id'] = range(len(queries_df))

    # --- 預處理語料庫 ---
    # 在此情境下，完整的queries_df就是我們的語料庫
    print("\nPreprocessing corpus...")
    processed_corpus = preprocess(queries_df.copy(), n_gram_range=(1, 1))

    # --- 實驗一: 基本Unigram(無查詢擴充) ---
    print("\n--- Evaluating with Unigrams (No Query Expansion) ---")
    # 初始化檢索器
    tfidf_retriever = TFIDFRetriever(processed_corpus)
    bm25_retriever = BM25Retriever(processed_corpus)

    # 在完整的資料集上進行評估
    tfidf_recall = evaluate(tfidf_retriever, queries_df, query_expansion=False)
    bm25_recall = evaluate(bm25_retriever, queries_df, query_expansion=False)
    
    print(f"TF-IDF Recall@10: {tfidf_recall:.4f}")
    print(f"BM25 Recall@10: {bm25_recall:.4f}")

    # --- 實驗二:Unigram+查詢擴充 ---
    print("\n--- Evaluating with Unigrams (With Query Expansion) ---")
    # 檢索器已建立，直接調用評估函式並開啟查詢擴充
    tfidf_recall_qe = evaluate(tfidf_retriever, queries_df, query_expansion=True)
    bm25_recall_qe = evaluate(bm25_retriever, queries_df, query_expansion=True)
    
    print(f"TF-IDF with Query Expansion Recall@10: {tfidf_recall_qe:.4f}")
    print(f"BM25 with Query Expansion Recall@10: {bm25_recall_qe:.4f}")