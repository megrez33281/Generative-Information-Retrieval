import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

num_layers = 4 # 選擇要用最後幾層的Output平均作為特徵
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletDataset(Dataset):
    """三元組數據集，用於微調密集檢索模型"""
    # 用於把資料變成模型的輸入格式
    def __init__(self, queries_df, negative_corpus_df, tokenizer, max_length=512):
        """初始化數據集"""
        self.queries_df = queries_df
        # 注意：這裡改成用 code_snippets.csv 作為負樣本來源，讓模型盡可能學習到真實情境
        self.corpus_df = negative_corpus_df  
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回數據集的大小"""
        return len(self.queries_df)

    def __getitem__(self, idx):
        """獲取一個數據樣本 (只回傳 raw text，tokenizer 在 collate_fn 中做batch化)"""
        anchor_text = self.queries_df.iloc[idx]['query']  # 錨點(anchor)是查詢
        positive_code = self.queries_df.iloc[idx]['code']  # 正樣本(positive)是與查詢對應的正確程式碼
        negative_candidates = self.corpus_df[self.corpus_df['code'] != positive_code]  # 從最終語料庫code_snippets中選負樣本
        negative_code = negative_candidates.sample(1)['code'].values[0]

        return anchor_text, positive_code, negative_code


def collate_fn(batch, tokenizer, max_length=512):
    """將 batch 的文字一次性 tokenizer，提高效率"""
    anchors, positives, negatives = zip(*batch)

    anchor_inputs = tokenizer(list(anchors), return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    positive_inputs = tokenizer(list(positives), return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    negative_inputs = tokenizer(list(negatives), return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)

    return {
        'anchor': {key: val.to(DEVICE) for key, val in anchor_inputs.items()},
        'positive': {key: val.to(DEVICE) for key, val in positive_inputs.items()},
        'negative': {key: val.to(DEVICE) for key, val in negative_inputs.items()}
    }


def get_embedding(model, tokenizer, text, max_length=512):
    """輔助函式，用於獲取單個文本的嵌入向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length).to(DEVICE)
    with torch.no_grad():
        # 判斷模型是否為 Encoder-Decoder 架構
        if hasattr(model, 'get_encoder'):
            outputs = model.get_encoder()(**inputs, output_hidden_states=True)
        else:
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        stacked_layers = torch.stack(hidden_states[-num_layers:])
        mean_last_layers = torch.mean(stacked_layers, dim=0)
        embedding = mean_last_layers.mean(dim=1)
    return embedding


# 將anchor、positive、negative的token輸入模型，取多層hidden_state做平均
def get_layerwise_embeddings(model, batch_inputs, num_layers=num_layers):
    """
    batch_inputs: batch['anchor'] / batch['positive'] / batch['negative']
    num_layers: 取最後幾層做平均（作為文字的特徵）
    """
    outputs = model(**batch_inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of all layers

    # 取最後 num_layers 層平均
    stacked_layers = torch.stack(hidden_states[-num_layers:])  # shape: (num_layers, batch_size, seq_len, hidden_size)
    mean_last_layers = torch.mean(stacked_layers, dim=0)      # shape: (batch_size, seq_len, hidden_size)

    # 對 token 平均 pooling，得到每個樣本的句子向量
    embeddings = mean_last_layers.mean(dim=1)  # shape: (batch_size, hidden_size)
    return embeddings


def evaluate_recall(model, tokenizer, val_df, corpus_df, cached_corpus_embeddings=None):
    """在驗證集上評估 Recall@10，使用 code_snippets.csv 作為語料庫"""
    model.eval() # 切換到評估模式
    
    # 先計算全部的語料庫特徵
    if cached_corpus_embeddings is None:
        print("\nCreating cached embeddings for the corpus (code_snippets.csv)...")
        all_codes = list(corpus_df['code'])
        corpus_embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(all_codes), batch_size), desc="Corpus Embeddings"):
            batch_codes = all_codes[i:i+batch_size]
            inputs = tokenizer(batch_codes, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                stacked_layers = torch.stack(hidden_states[-num_layers:])
                mean_last_layers = torch.mean(stacked_layers, dim=0)
                embeddings = mean_last_layers.mean(dim=1)
            corpus_embeddings.append(embeddings.cpu())
        corpus_embeddings = torch.cat(corpus_embeddings, dim=0).to(DEVICE)
    else:
        corpus_embeddings = cached_corpus_embeddings

    recall_at_10 = 0
    for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0], desc="Evaluating Recall@10"):
        query = row['query']
        true_code_string = row['code']
        
        query_embedding = get_embedding(model, tokenizer, query)
        
        # 計算餘弦相似度
        scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
        top_k_indices = torch.argsort(scores, descending=True)[:10]
        top_k_codes = corpus_df.iloc[top_k_indices]['code'].values
        
        if true_code_string in top_k_codes:
            recall_at_10 += 1
            
    return recall_at_10 / len(val_df), corpus_embeddings


def fine_tune_model(model, tokenizer, train_df, val_df, negative_corpus_df, epochs=3, lr=2e-5, batch_size=8):
    """微調預訓練模型"""
    model.to(DEVICE)

    # 這是對訓練資料的準備，會產生每個樣本的anchor/positive/negative張量
    dataset = TripletDataset(train_df, negative_corpus_df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda x: collate_fn(x, tokenizer))  # 使用 collate_fn 做 batch tokenizer
    
    # 設定優化器和損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # 標準Transformer訓練用優化器
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0) # 三元組損失，目標是讓anchor（查詢）與positive（正確答案之間的距離小於anchor與negative（錯誤答案）之間的距離，至少相差一個margin

    cached_corpus_embeddings = None

    # 訓練模型
    for epoch in range(epochs):
        model.train() # 切換到訓練模式
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad() # 清除上一步梯度
            
            anchor_embeddings = get_layerwise_embeddings(model, batch['anchor'])
            positive_embeddings = get_layerwise_embeddings(model, batch['positive'])
            negative_embeddings = get_layerwise_embeddings(model, batch['negative'])

            # 計算損失
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()  # 計算梯度
            optimizer.step()  # 更新參數
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")



    return model


class DenseRetriever:
    """密集檢索器"""
    def __init__(self, documents, model_name_or_path='microsoft/codebert-base', batch_size=32):
        """初始化密集檢索器"""
        self.documents = documents
        # 載入預訓練模型和斷詞器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(DEVICE)
        self.model.eval() # 預設為評估模式
        self.batch_size = batch_size
        # 建立文件的嵌入向量（使用 batch 化）
        self.doc_embeddings = self._create_doc_embeddings()

    def _create_doc_embeddings(self):
        """建立所有文件的嵌入向量 (batch 化加速)"""
        all_codes = list(self.documents['code'])
        embeddings = []
        for i in tqdm(range(0, len(all_codes), self.batch_size), desc="Creating document embeddings"):
            batch_codes = all_codes[i:i+self.batch_size]
            inputs = self.tokenizer(batch_codes, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                stacked_layers = torch.stack(hidden_states[-num_layers:])
                mean_last_layers = torch.mean(stacked_layers, dim=0)
                batch_embeddings = mean_last_layers.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)

    def retrieve(self, query, k=10):
        """根據查詢檢索文件"""
        # 對單一 query 編碼
        query_embedding = get_embedding(self.model, self.tokenizer, query).cpu().numpy()
        # 計算餘弦相似度
        scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        return top_k_indices, top_k_scores
    
def split_data(train_queries_df):
    # 90% 的code-query配對用於訓練，剩餘10%的query用於評估並對答案
    # 每個 code 是一個 group
    groups = train_queries_df['code']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(train_queries_df, groups=groups))
    train_df = train_queries_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_queries_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df

if __name__ == '__main__':
    # --- 1. 準備資料 ---
    print("--- Preparing Data ---")
    # 載入所有用於訓練的查詢對
    train_queries_df = pd.read_csv('train_queries.csv')
    # 載入最終檢索語料庫（code_snippets.csv），作為負樣本來源
    code_snippets_df = pd.read_csv('code_snippets.csv')

    train_df, val_df = split_data(train_queries_df)

    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

    # --- 2. 初始化模型 ---
    print("\n--- Initializing Model ---")
    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # --- 3. 微調模型 ---
    print("\n--- Fine-tuning model with robust validation ---")
    fine_tuned_model = fine_tune_model(model, tokenizer, train_df, val_df, code_snippets_df, epochs=3)

    # --- 4. 儲存模型 ---
    output_dir = './fine_tuned_codebert_robust'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nSaving the final model to {output_dir}...")
    fine_tuned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n--- Model training complete. ---")
