import pandas as pd
import numpy as np
from tqdm import tqdm
from sparse_retrieval import TFIDFRetriever, BM25Retriever
from dense_retrieval import DenseRetriever
from preprocess import load_code_snippets, preprocess

def normalize_scores(scores):
    """將分數正規化到 [0, 1] 區間"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

if __name__ == '__main__':
    # --- 載入資料 ---
    print("Loading data...")
    code_snippets_df = load_code_snippets('code_snippets.csv')
    test_queries_df = pd.read_csv('test_queries.csv')

    # --- 初始化所有需要的檢索器 ---
    print("\nInitializing retrievers...")
    processed_snippets_df = preprocess(code_snippets_df.copy())

    # 使用最佳參數初始化稀疏模型
    tfidf_retriever = TFIDFRetriever(processed_snippets_df)
    bm25_retriever = BM25Retriever(processed_snippets_df, k1=2.0, b=0.9)
    
    # 初始化密集模型 (假設最佳模型已儲存)
    finetuned_model_path = './fine_tuned_codebert'
    dense_retriever = DenseRetriever(code_snippets_df, model_name_or_path=finetuned_model_path)

    # --- 設定混合參數 ---
    # alpha 決定了密集檢索器分數的權重
    alpha = 0.5 
    # 從每個檢索器中取回的候選數量，以便進行重新排序
    top_n_candidates = 100

    final_results = []

    print(f"\nGenerating hybrid retrieval submission with alpha={alpha}...")
    for _, row in tqdm(test_queries_df.iterrows(), total=test_queries_df.shape[0]):
        query_id = row['query_id']
        query = row['query']

        # 1. 從每個檢索器獲取 top_n 候選者及其分數
        # 此處我們結合 TF-IDF 和微調後的 Dense 模型
        tfidf_indices, tfidf_scores = tfidf_retriever.retrieve(query, k=top_n_candidates, query_expansion=True)
        dense_indices, dense_scores = dense_retriever.retrieve(query, k=top_n_candidates)

        # 2. 結合候選者並建立一個分數映射
        all_indices = np.union1d(tfidf_indices, dense_indices)
        score_map = {idx: {'tfidf': 0.0, 'dense': 0.0} for idx in all_indices}

        for idx, score in zip(tfidf_indices, tfidf_scores):
            score_map[idx]['tfidf'] = score
        for idx, score in zip(dense_indices, dense_scores):
            score_map[idx]['dense'] = score

        # 3.正規化並計算混合分數
        tfidf_scores_all = np.array([s['tfidf'] for s in score_map.values()])
        dense_scores_all = np.array([s['dense'] for s in score_map.values()])

        norm_tfidf_scores = normalize_scores(tfidf_scores_all)
        norm_dense_scores = normalize_scores(dense_scores_all)

        hybrid_scores = (1 - alpha) * norm_tfidf_scores + alpha * norm_dense_scores

        # 4. 重新排序並選出前 10 名
        all_indices_list = list(score_map.keys())
        final_ranked_indices = np.array(all_indices_list)[np.argsort(hybrid_scores)[::-1]][:10]

        # 獲取 code_id
        top_10_code_ids = code_snippets_df.iloc[final_ranked_indices]['code_id'].tolist()

        final_results.append({
            'query_id': query_id,
            'code_id': ' '.join(map(str, top_10_code_ids))
        })

    # --- 儲存提交檔案 ---
    submission_df = pd.DataFrame(final_results)
    output_path = 'submission_hybrid.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"\nHybrid submission file saved to {output_path}")
