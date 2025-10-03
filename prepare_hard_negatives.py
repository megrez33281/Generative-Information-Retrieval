import pandas as pd
import json
from tqdm import tqdm
from sparse_retrieval import TFIDFRetriever
from preprocess import load_code_snippets, preprocess

def prepare_hard_negatives(top_k=50):
    """
    為訓練查詢生成困難負樣本。
    對於train_queries.csv（訓練資料）中的每個查詢，使用TF-IDF在code_snippets.csv中（test_queries的語料庫，也是微調模型時的負樣本）
    檢索top-k的相似程式碼，並將其作為困難負樣本儲存。
    """
    print("--- 開始生成困難負樣本 ---")

    # 1. 載入所有需要的資料
    print("步驟 1/4: 載入資料...")
    train_queries_df = pd.read_csv('train_queries.csv')
    code_snippets_df = load_code_snippets('code_snippets.csv')

    # 2. 預處理程式碼片段並初始化TF-IDF檢索器
    print("步驟 2/4: 初始化 TF-IDF 檢索器...")
    processed_snippets_df = preprocess(code_snippets_df)
    # 使用TF-IDF檢索器
    tfidf_retriever = TFIDFRetriever(processed_snippets_df)
    print("檢索器初始化完成。")

    # 3. 為每個訓練查詢尋找困難負樣本
    print("步驟 3/4: 挖掘困難負樣本...")
    training_data_with_negatives = []
    for _, row in tqdm(train_queries_df.iterrows(), total=len(train_queries_df), desc="處理查詢"):
        query = row['query']
        positive_code = row['code']

        # 使用 TF-IDF 檢索 Top-K個候選
        # 由於 train_queries.csv 中的 code 不存在於 code_snippets.csv 中，不需要擔心檢索到正樣本
        top_indices, _ = tfidf_retriever.retrieve(query, k=top_k, query_expansion=True)

        # 將檢索到的索引 (indices) 轉換為實際的 code_id（code_snippets中有code_id）
        hard_negative_ids = [int(code_snippets_df.iloc[i]['code_id']) for i in top_indices]

        training_data_with_negatives.append({
            'query': query,
            'positive_code': positive_code,
            'hard_negative_ids': hard_negative_ids
        })

    # 4. 儲存結果到JSON檔案
    output_path = 'train_data_with_negatives.json'
    print(f"步驟 4/4: 儲存結果到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data_with_negatives, f, indent=4, ensure_ascii=False)

    print(f"--- 成功生成並儲存 {len(training_data_with_negatives)} 筆含困難負樣本的訓練資料 ---")

if __name__ == '__main__':
    prepare_hard_negatives()
