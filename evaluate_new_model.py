import pandas as pd
import os
import sys
from tqdm import tqdm
from dense_retrieval import DenseRetriever
from preprocess import load_code_snippets

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_new_model.py <model_name_or_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    # 從模型名稱產生輸出檔案名，替換斜線以避免路徑問題
    output_filename = f"submission_{model_name.replace('/', '_')}.csv"

    print(f"--- Evaluating model: {model_name} ---")

    # --- 載入資料 ---
    print("Loading data...")
    code_snippets_df = load_code_snippets('code_snippets.csv')
    test_queries_df = pd.read_csv('test_queries.csv')

    # --- 初始化檢索器 ---
    print(f"\nInitializing DenseRetriever with {model_name}...")
    retriever = DenseRetriever(code_snippets_df, model_name_or_path=model_name)

    # --- 產生提交檔案 ---
    print(f"Generating submission file: {output_filename}...")
    results = []
    for _, row in tqdm(test_queries_df.iterrows(), total=test_queries_df.shape[0]):
        query_id = row['query_id']
        query = row['query']
        
        top_k_indices, _ = retriever.retrieve(query, k=10)
        top_k_code_ids = retriever.documents.iloc[top_k_indices]['code_id'].tolist()
        
        results.append({
            'query_id': query_id,
            'code_id': ' '.join(map(str, top_k_code_ids))
        })
        
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_filename, index=False)
    print(f"\nSubmission file saved to {output_filename}")
