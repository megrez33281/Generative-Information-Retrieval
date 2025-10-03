import torch
import pandas as pd
from tqdm import tqdm
# 導入 sentence-transformers 的 CrossEncoder
import os
from sentence_transformers import CrossEncoder

# 導入我們現有的檢索器和預處理函數
from fine_tune_model import DenseRetriever, split_data
from preprocess import load_code_snippets, preprocess
from transformers import AutoModel, AutoTokenizer

# --- 組態設定 ---
# 使用絕對路徑來避免任何歧義
CWD = os.getcwd()
BI_ENCODER_PATH = os.path.join(CWD, 'fine_tuned_unixcoder_hard_neg')
CROSS_ENCODER_NAME = os.path.join(CWD, 'fine_tuned_cross_encoder_v2')

# 召回階段要檢索的候選者數量
TOP_K_RETRIEVE = 100

# 設定設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if __name__ == '__main__':
    # --- 模式設定 ---
    # True: 執行本地驗證 (使用 train_queries.csv)
    # False: 產生 Kaggle 提交檔案 (使用 test_queries.csv)
    RUN_VALIDATION = True

    # --- 1. 本地驗證模式 ---
    if RUN_VALIDATION:
        print("--- Running in Local Validation Mode for Re-ranker ---")
        
        # 1a. 載入訓練資料並分割
        print("Loading and splitting train_queries.csv for validation...")
        train_queries_df = pd.read_csv('train_queries.csv')
        _, val_df = split_data(train_queries_df)
        
        # 驗證時，整個 train_queries_df 就是我們的語料庫
        corpus_df = train_queries_df
        print(f"Using {len(val_df)} queries for validation against a corpus of {len(corpus_df)} code snippets.")

        # 1b. 初始化模型
        print("\nInitializing models for validation...")
        # 召回器使用 train_queries 作為語料庫
        retriever = DenseRetriever(corpus_df, model_name_or_path=BI_ENCODER_PATH)
        # 精排器直接載入
        re_ranker = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE)

        # 1c. 執行並計算 Recall@10
        recall_at_10_count = 0
        print(f"\nEvaluating Re-rank pipeline (Retrieve Top {TOP_K_RETRIEVE})...")
        for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
            query = row['query']
            true_code_content = row['code']

            # 第一階段：召回
            candidate_indices, _ = retriever.retrieve(query, k=TOP_K_RETRIEVE)
            candidate_snippets = corpus_df.iloc[candidate_indices]

            # 第二階段：精排
            cross_encoder_input = [(query, snippet) for snippet in candidate_snippets['code']]
            cross_encoder_scores = re_ranker.predict(cross_encoder_input)
            
            candidate_snippets = candidate_snippets.copy()
            candidate_snippets['rerank_score'] = cross_encoder_scores
            reranked_snippets = candidate_snippets.sort_values(by='rerank_score', ascending=False)
            
            top_10_codes = reranked_snippets.head(10)['code'].tolist()

            if true_code_content in top_10_codes:
                recall_at_10_count += 1

        # 計算並印出最終分數
        final_recall = recall_at_10_count / len(val_df)
        print(f"\n--- Validation Complete ---")
        print(f"Re-rank Model Local Recall@10: {final_recall:.4f}")

    # --- 2. Kaggle 預測模式 ---
    else:
        print("--- Running in Prediction Mode for Re-ranker ---")
        
        # 2a. 載入資料
        print("Loading data for prediction...")
        code_snippets_df = load_code_snippets('code_snippets.csv')
        test_queries_df = pd.read_csv('test_queries.csv')

        # 2b. 初始化模型
        print("\nInitializing models for prediction...")
        # 召回器使用 code_snippets 作為語料庫
        retriever = DenseRetriever(code_snippets_df, model_name_or_path=BI_ENCODER_PATH)
        # 精排器直接載入
        re_ranker = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE)

        # 2c. 執行 Retrieve-and-Re-rank 流程
        final_results = []
        print(f"\nRunning Retrieve-and-Re-rank pipeline (Retrieve Top {TOP_K_RETRIEVE})...")
        for _, row in tqdm(test_queries_df.iterrows(), total=test_queries_df.shape[0]):
            query_id = row['query_id']
            query = row['query']

            # 第一階段：召回
            candidate_indices, _ = retriever.retrieve(query, k=TOP_K_RETRIEVE)
            candidate_snippets = code_snippets_df.iloc[candidate_indices]

            # 第二階段：精排
            cross_encoder_input = [(query, snippet) for snippet in candidate_snippets['code']]
            cross_encoder_scores = re_ranker.predict(cross_encoder_input)
            
            candidate_snippets = candidate_snippets.copy()
            candidate_snippets['rerank_score'] = cross_encoder_scores
            reranked_snippets = candidate_snippets.sort_values(by='rerank_score', ascending=False)
            
            top_10_code_ids = reranked_snippets.head(10)['code_id'].tolist()

            final_results.append({
                'query_id': query_id,
                'code_id': ' '.join(map(str, top_10_code_ids))
            })

        # 2d. 儲存提交檔案
        submission_df = pd.DataFrame(final_results)
        output_path = 'submission_rerank.csv'
        submission_df.to_csv(output_path, index=False)
        print(f"\nRe-rank submission file saved to {output_path}")
