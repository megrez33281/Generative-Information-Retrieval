import pandas as pd
from tqdm import tqdm
from collections import defaultdict


# 需要的檢索器與輔助函數
from sparse_retrieval import TFIDFRetriever
from fine_tune_model import DenseRetriever
from preprocess import load_code_snippets, preprocess
from fine_tune_model import split_data # 導入資料集分割函數

PRE_TRAINED_MODEL_NAME = 'microsoft/unixcoder-base'
FINE_TUNED_MODEL_PATH = './' + PRE_TRAINED_MODEL_NAME.replace("/", "-")


def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    使用 RRF 演算法融合多個排名列表。
    :param ranked_lists: 一個包含多個排名列表的列表。每個排名列表是 code_id 或 code_content 的列表。
    :param k: RRF 演算法中的常數，通常設為 60。
    :return: 融合併重新排序後的項目列表。
    """

    """
    RRF的核心思想是完全忽略掉原始分數，只關心每個檢索器給出的排名
    對於每一個候選的程式碼（code_id），它的最終RRF分數是它在每個檢索結果列表中的倒數排名分數的總和
    總而言之RRF會獎勵那些在多個不同檢索系統中都穩定地排在前面的項目，它完全繞開了不同系統之間分數無法直接比較的問題
    """
    rrf_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            rrf_scores[item] += 1 / (k + rank + 1)
            
    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    fused_list = [item[0] for item in sorted_items]
    
    return fused_list

if __name__ == '__main__':
    # --- 模式設定 ---
    # True: 執行本地驗證 (使用 train_queries.csv)
    # False: 產生 Kaggle 提交檔案 (使用 test_queries.csv)
    RUN_VALIDATION = True

    # --- 1. 本地驗證模式 ---
    if RUN_VALIDATION:
        # --- 測試模式設定 ---
        # 'tfidf': 只測試 TF-IDF 的表現
        # 'dense': 只測試 Dense Retriever 的表現
        # 'rrf': 測試 RRF 混合模型的表現
        TEST_MODE = 'rrf' # 可以切換這個值來進行測試

        print(f"--- Running in Local Validation Mode (Test Mode: {TEST_MODE}) ---")
        
        # 載入訓練資料並分割
        print("Loading and splitting train_queries.csv for validation...")
        train_queries_df = pd.read_csv('train_queries.csv')
        _, val_df = split_data(train_queries_df)
        
        # 驗證時，整個 train_queries_df 就是我們的語料庫
        corpus_df = train_queries_df
        print(f"Using {len(val_df)} queries for validation against a corpus of {len(corpus_df)} code snippets.")

        # 初始化檢索器 (使用 train_queries 作為語料庫)
        print("\nInitializing retrievers for validation...")
        processed_corpus_df = preprocess(corpus_df.copy())
        
        if TEST_MODE == 'tfidf' or TEST_MODE == 'rrf':
            tfidf_retriever = TFIDFRetriever(processed_corpus_df)
        
        if TEST_MODE == 'dense' or TEST_MODE == 'rrf':
            finetuned_model_path = FINE_TUNED_MODEL_PATH
            print(f"Loading dense model from: {finetuned_model_path}")
            dense_retriever = DenseRetriever(corpus_df, model_name_or_path=finetuned_model_path)

        top_n_candidates = 100
        recall_at_10_count = 0

        print(f"\nEvaluating {TEST_MODE} retrieval (top_n={top_n_candidates})...")
        for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
            query = row['query']
            true_code_content = row['code']

            if TEST_MODE == 'tfidf':
                tfidf_indices, _ = tfidf_retriever.retrieve(query, k=10, query_expansion=True)
                top_10_codes = corpus_df.iloc[tfidf_indices]['code'].tolist()
            
            elif TEST_MODE == 'dense':
                dense_indices, _ = dense_retriever.retrieve(query, k=10)
                top_10_codes = corpus_df.iloc[dense_indices]['code'].tolist()

            elif TEST_MODE == 'rrf':
                tfidf_indices, _ = tfidf_retriever.retrieve(query, k=top_n_candidates, query_expansion=True)
                tfidf_ranked_codes = corpus_df.iloc[tfidf_indices]['code'].tolist()

                dense_indices, _ = dense_retriever.retrieve(query, k=top_n_candidates)
                dense_ranked_codes = corpus_df.iloc[dense_indices]['code'].tolist()

                fused_ranked_list = reciprocal_rank_fusion([tfidf_ranked_codes, dense_ranked_codes])
                top_10_codes = fused_ranked_list[:10]

            if true_code_content in top_10_codes:
                recall_at_10_count += 1

        final_recall = recall_at_10_count / len(val_df)
        print(f"\n--- Validation Complete ---")
        print(f"Model: {TEST_MODE}, Local Recall@10: {final_recall:.4f}")

    # --- 2. Kaggle 預測模式 ---
    else:
        # (此處省略，與前一版本相同)
        pass # 保持原有的預測邏輯不變
