import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from fine_tune_model import evaluate_recall, split_data

# 組態設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = 'microsoft/unixcoder-base'
FINE_TUNED_MODEL_PATH = './' + PRE_TRAINED_MODEL_NAME.replace("/", "-")

# 選擇要測試的model
try_prtrained_model = False
try_fine_tuned_model = True


if __name__ == '__main__':
    # 1. 載入資料並準備驗證集
    print("--- 1. 載入資料並準備驗證集 ---")
    # 注意，此處用於測試的語料庫也來自於train_queries.csv，訓練時保留了10%的query沒有用於訓練
    train_queries_df = pd.read_csv('train_queries.csv')


    # 使用與微調腳本完全相同的分割方式
    _, val_df = split_data(train_queries_df)
    print(f"已載入 {len(val_df)} 筆樣本用於驗證。")

    # 2. 評估預訓練模型
    if try_prtrained_model:
        print("\n--- 2. 評估預訓練模型 ---")
        print(f"模型: {PRE_TRAINED_MODEL_NAME}")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        pretrained_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(DEVICE)
        
        pretrained_recall, corpus_embeddings_pretrained = evaluate_recall(pretrained_model, pretrained_tokenizer, val_df, train_queries_df)
        print(f"\n預訓練模型 Recall@10: {pretrained_recall:.4f}")

    # 3. 評估微調後的模型
    if try_fine_tuned_model:
        print("\n--- 3. 評估微調後的模型 ---")
        print(f"模型: {FINE_TUNED_MODEL_PATH}")
        try:
            finetuned_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
            finetuned_model = AutoModel.from_pretrained(FINE_TUNED_MODEL_PATH).to(DEVICE)
            
            # 微調後的模型需要重新計算語料庫的嵌入向量
            finetuned_recall, _ = evaluate_recall(finetuned_model, finetuned_tokenizer, val_df, train_queries_df)
            print(f"\n微調後模型 Recall@10: {finetuned_recall:.4f}")

        except OSError:
            print(f"錯誤: 在 '{FINE_TUNED_MODEL_PATH}' 找不到微調後的模型。")
            print("請先執行 'fine_tune_model.py' 來訓練並儲存模型。")

    print("\n--- 評估完成 ---")