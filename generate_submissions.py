import pandas as pd
import os
from tqdm import tqdm
from sparse_retrieval import TFIDFRetriever, BM25Retriever
from fine_tune_model import DenseRetriever
from preprocess import load_code_snippets, preprocess

def generate_submission(retriever, test_df, output_path, query_expansion=False):
    """
    Generates a submission file for a given retriever.
    """
    print(f"Generating submission for {output_path}...")
    results = []
    # 使用tqdm顯示進度條
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc=output_path):
        query_id = row['query_id']
        query = row['query']
        
        # 根據檢索器類型決定是否使用查詢擴充
        if isinstance(retriever, (TFIDFRetriever, BM25Retriever)):
            top_k_indices, _ = retriever.retrieve(query, k=10, query_expansion=query_expansion)
        else:
            top_k_indices, _ = retriever.retrieve(query, k=10)
        
        # 直接使用檢索器內部儲存的 documents DataFrame 來獲取 code_id
        top_k_code_ids = retriever.documents.iloc[top_k_indices]['code_id'].tolist()
        
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
    print("\nInitializing sparse models with best parameters...")
    processed_snippets_df = preprocess(code_snippets_df.copy())
    
    # TF-IDF Retriever with Query Expansion
    tfidf_retriever = TFIDFRetriever(processed_snippets_df)
    generate_submission(tfidf_retriever, test_queries_df, 'submission_tfidf.csv', query_expansion=True)

    # BM25 Retriever with optimized parameters and Query Expansion
    bm25_retriever = BM25Retriever(processed_snippets_df, k1=2.0, b=0.9)
    generate_submission(bm25_retriever, test_queries_df, 'submission_bm25.csv', query_expansion=True)

    # --- 密集模型 ---
    # 檢查微調後的模型是否存在
    finetuned_model_path = './fine_tuned_codebert'
    
    # 預訓練的密集檢索器
    print("\nInitializing pre-trained dense model...")
    pretrained_retriever = DenseRetriever(code_snippets_df, model_name_or_path='microsoft/codebert-base')
    generate_submission(pretrained_retriever, test_queries_df, 'submission_pretrained.csv')

    if not os.path.exists(finetuned_model_path):
        print(f"\nFine-tuned model not found at '{finetuned_model_path}'.")
        print("Skipping submission generation for the fine-tuned model.")
    else:
        # 微調後的密集檢索器
        print("\nInitializing fine-tuned dense model...")
        finetuned_retriever = DenseRetriever(code_snippets_df, model_name_or_path=finetuned_model_path)
        generate_submission(finetuned_retriever, test_queries_df, 'submission_finetuned.csv')

    print("\nAll submission files have been generated.")