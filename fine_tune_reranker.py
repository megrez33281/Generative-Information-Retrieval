import torch
import pandas as pd
import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
# 導入 sentence-transformers 的 CrossEncoder 和 InputExample
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from torch.nn import BCEWithLogitsLoss
import os

# --- 組態設定 ---
# 要進行微調的預訓練 Cross-Encoder
CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# 微調後模型的儲存路徑
OUTPUT_PATH = './fine_tuned_cross_encoder_v2' # 使用新路徑 v2

# 訓練超參數
EPOCHS = 3 # 增加訓練輪數
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
NUM_HARD_NEGATIVES_PER_POSITIVE = 2 # 為每個正樣本配備的困難負樣本數量

# 設定設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


if __name__ == '__main__':
    # --- 1. 準備資料 ---
    print("--- Preparing Data for Cross-Encoder Fine-tuning ---")
    
    train_queries_df = pd.read_csv('train_queries.csv')
    with open('train_data_with_negatives.json', 'r', encoding='utf-8') as f:
        train_data_with_negatives = json.load(f)
    
    code_snippets_df = pd.read_csv('code_snippets.csv')
    code_id_to_code_map = pd.Series(code_snippets_df.code.values, index=code_snippets_df.code_id).to_dict()

    # --- 2. 準備訓練樣本 (使用 InputExample) ---
    print("\n--- Preparing training examples ---")
    train_samples = []
    
    print("Creating positive and hard negative examples...")
    for item in tqdm(train_data_with_negatives):
        query = item['query']
        positive_code = item['positive_code']
        
        # 1. 添加正樣本
        train_samples.append(InputExample(texts=[query, positive_code], label=1.0))
        
        # 2. 添加多個困難負樣本
        # 從50個困難樣本中隨機抽取，避免重複
        if len(item['hard_negative_ids']) >= NUM_HARD_NEGATIVES_PER_POSITIVE:
            sampled_negative_ids = random.sample(item['hard_negative_ids'], NUM_HARD_NEGATIVES_PER_POSITIVE)
        else:
            sampled_negative_ids = item['hard_negative_ids'] # 如果不夠，全選
            
        for neg_id in sampled_negative_ids:
            negative_code = code_id_to_code_map[neg_id]
            train_samples.append(InputExample(texts=[query, negative_code], label=0.0))

    print(f"Total training samples: {len(train_samples)}")

    # --- 3. 初始化模型和 DataLoader ---
    print("\n--- Initializing Model and DataLoader ---")
    model = CrossEncoder(CROSS_ENCODER_NAME, num_labels=1, device=DEVICE)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fct = BCEWithLogitsLoss()

    # --- 4. 微調模型 ---
    print(f"\n--- Fine-tuning Cross-Encoder for {EPOCHS} epoch(s) ---")
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_fct,
        epochs=EPOCHS,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True,
        output_path=OUTPUT_PATH,
        save_best_model=False
    )

    print(f"\n--- Model fine-tuning complete. Model saved to {OUTPUT_PATH} ---")
