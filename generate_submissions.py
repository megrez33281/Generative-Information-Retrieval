
import pandas as pd
import os
from tqdm import tqdm
from sparse_retrieval import TFIDFRetriever, BM25Retriever
from dense_retrieval import DenseRetriever
from preprocess import load_code_snippets, preprocess

def generate_submission(retriever, test_df, documents_df, output_path):
    """
    Generates a submission file for a given retriever.
    """
    print(f"Generating submission for {output_path}...")
    results = []
    # 使用tqdm顯示進度條
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc=output_path):
        query_id = row['query_id']
        query = row['query']
        
        # retrieve 方法返回的是索引，我們需要將它們映射回code_id
        top_k_indices = retriever.retrieve(query, k=10)
        top_k_code_ids = documents_df.iloc[top_k_indices]['code_id'].tolist()
        
        results.append({
            'query_id': query_id,
            'code_id': ' '.join(map(str, top_k_code_ids))
        })
        
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

if __name__ == '__main__':
    # --- 載入資料 ---
    print("Loading data...")
    code_snippets_df = load_code_snippets('code_snippets.csv')
    test_queries_df = pd.read_csv('test_queries.csv')

    # --- 稀疏模型 ---
    print("\nInitializing sparse models...")
    processed_snippets_df = preprocess(code_snippets_df.copy())
    
    # TF-IDF Retriever
    tfidf_retriever = TFIDFRetriever(processed_snippets_df)
    generate_submission(tfidf_retriever, test_queries_df, code_snippets_df, 'submission_tfidf.csv')

    # BM25 Retriever
    bm25_retriever = BM25Retriever(processed_snippets_df)
    generate_submission(bm25_retriever, test_queries_df, code_snippets_df, 'submission_bm25.csv')

    # --- 密集模型 ---
    # 檢查微調後的模型是否存在
    finetuned_model_path = './fine_tuned_codebert'
    
    # 預訓練的密集檢索器
    print("\nInitializing pre-trained dense model...")
    pretrained_retriever = DenseRetriever(code_snippets_df, model_name_or_path='microsoft/codebert-base')
    generate_submission(pretrained_retriever, test_queries_df, code_snippets_df, 'submission_pretrained.csv')

    if not os.path.exists(finetuned_model_path):
        print(f"\n在指定位置找不到模型：'{finetuned_model_path}'.")
        print("跳過微調模型的答案生產")
    else:
        # 微調後的密集檢索器
        print("\nInitializing fine-tuned dense model...")
        finetuned_retriever = DenseRetriever(code_snippets_df, model_name_or_path=finetuned_model_path)
        generate_submission(finetuned_retriever, test_queries_df, code_snippets_df, 'submission_finetuned.csv')

    print("\nAll submission files have been generated.")
