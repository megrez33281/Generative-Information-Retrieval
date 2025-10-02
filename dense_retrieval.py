import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from preprocess import load_code_snippets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy

num_layers = 4 # 選擇要用最後幾層的Output平均作為特徵
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletDataset(Dataset):
    """三元組數據集，用於微調密集檢索模型"""
    # 用於把資料變成模型的輸入格式
    def __init__(self, queries_df, snippets_df, tokenizer, max_length=512):
        """初始化數據集"""
        self.queries_df = queries_df
        self.snippets_df = snippets_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回數據集的大小"""
        return len(self.queries_df)

    def __getitem__(self, idx):
        """獲取一個數據樣本"""
        # 錨點(anchor)是查詢
        anchor_text = self.queries_df.iloc[idx]['query']
        # 正樣本(positive)是與查詢對應的正確程式碼
        positive_code = self.queries_df.iloc[idx]['code']

        # 負樣本(negative)是隨機選擇的錯誤程式碼
        # 這裡使用了最簡單的隨機負採樣，更進階的方法是使用困難負樣本
        negative_candidates = self.snippets_df[self.snippets_df['code'] != positive_code] # 避免拿到正樣本
        negative_code = negative_candidates.sample(1)['code'].values[0]

        # 對錨點、正樣本和負樣本進行斷詞和編碼
        anchor_inputs = self.tokenizer(anchor_text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        positive_inputs = self.tokenizer(positive_code, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        negative_inputs = self.tokenizer(negative_code, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)

        return {
            'anchor': {key: val.squeeze().to(DEVICE) for key, val in anchor_inputs.items()},
            'positive': {key: val.squeeze().to(DEVICE) for key, val in positive_inputs.items()},
            'negative': {key: val.squeeze().to(DEVICE) for key, val in negative_inputs.items()}
        }

def get_embedding(model, tokenizer, text, max_length=512):
    """輔助函式，用於獲取單個文本的嵌入向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length).to(DEVICE)
    with torch.no_grad():
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
    # 在輸出層前，會經過很多層的Transformer（特徵抽取），而輸入的文字會隨著一次次的特徵抽取變得越來越抽象（變成高階特徵），所以如果結合前幾層的Output，就可以保留一些細節特徵
    outputs = model(**batch_inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of all layers

    # 取最後 num_layers 層平均
    # hidden_states[-num_layers:] → tuple of last N layers
    stacked_layers = torch.stack(hidden_states[-num_layers:])  # shape: (num_layers, batch_size, seq_len, hidden_size)
    mean_last_layers = torch.mean(stacked_layers, dim=0)      # shape: (batch_size, seq_len, hidden_size)

    # 對 token 平均 pooling，得到每個樣本的句子向量
    embeddings = mean_last_layers.mean(dim=1)  # shape: (batch_size, hidden_size)
    return embeddings

def evaluate_recall(model, tokenizer, val_df):
    """在驗證集上評估 Recall@10"""
    model.eval() # 切換到評估模式
    
    # 將驗證集中的 code 當作一個小型的語料庫 (corpus)
    val_corpus = val_df['code'].tolist()
    corpus_embeddings = torch.cat([get_embedding(model, tokenizer, code) for code in val_corpus]).to(DEVICE)
    
    recall_at_10 = 0
    for idx, row in val_df.iterrows():
        query = row['query']
        true_code_index = val_df.index.get_loc(idx)
        
        query_embedding = get_embedding(model, tokenizer, query)
        
        # 計算餘弦相似度
        scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
        top_k_indices = torch.argsort(scores, descending=True)[:10]
        
        if true_code_index in top_k_indices:
            recall_at_10 += 1
            
    return recall_at_10 / len(val_df)

def fine_tune_model(model, tokenizer, train_df, val_df, code_snippets_df, epochs=3, lr=2e-5, batch_size=8):
    """微調預訓練模型，並在每個 epoch 後進行驗證"""
    model.to(DEVICE)

    # 這是對訓練資料的準備，會產生每個樣本的anchor/positive/negative張量
    dataset = TripletDataset(train_df, code_snippets_df, tokenizer)
    # dataloader會自行調用dataset中的method產生訓練資料，自動batch、打亂資料
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 設定優化器和損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # 標準Transformer訓練用優化器
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0) # 三元組損失，目標是讓anchor（查詢）與positive（正確答案之間的距離小於anchor與negative（錯誤答案）之間的距離，至少相差一個margin

    best_recall = -1
    best_model_state = None

    # 訓練模型
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad() # 清除上一步梯度
            
            # 獲取錨點、正樣本和負樣本的嵌入向量（在這裡，只會用到模型的特徵抽取能力，不會用到輸出）
            anchor_embeddings = get_layerwise_embeddings(model, batch['anchor'])
            positive_embeddings = get_layerwise_embeddings(model, batch['positive'])
            negative_embeddings = get_layerwise_embeddings(model, batch['negative'])

            # 計算損失
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            # 計算梯度
            loss.backward()
            # 更新參數
            optimizer.step()
            """
            Loss Function可以理解為模型的目標，目標就是要讓這個函數的值越來越小。
            對應此處的情境，就是讓anchor的特徵向量與正確答案的特徵向量越來越近，與錯誤答案的特徵向量越來越遠
            """
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")

        # 在驗證集上評估
        recall = evaluate_recall(model, tokenizer, val_df)
        print(f"Epoch: {epoch+1}, Validation Recall@10: {recall:.4f}")

        # 如果是目前最好的模型，就儲存起來
        if recall > best_recall:
            best_recall = recall
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model found with Recall@10: {best_recall:.4f}!")

    # 載入表現最好的模型狀態
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

class DenseRetriever:
    """密集檢索器"""
    def __init__(self, documents, model_name_or_path='microsoft/codebert-base'):
        """初始化密集檢索器"""
        self.documents = documents
        # 載入預訓練模型和斷詞器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(DEVICE)
        self.model.eval() # 預設為評估模式
        # 建立文件的嵌入向量
        self.doc_embeddings = self._create_doc_embeddings()

    def _create_doc_embeddings(self):
        """建立所有文件的嵌入向量"""
        embeddings = []
        for doc in tqdm(self.documents['code'], desc="Creating document embeddings"):
            embedding = get_embedding(self.model, self.tokenizer, doc)
            embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)

    def retrieve(self, query, k=10):
        """根據查詢檢索文件"""
        query_embedding = get_embedding(self.model, self.tokenizer, query).cpu().numpy()
        # 計算餘弦相似度
        scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        return top_k_indices, top_k_scores

if __name__ == '__main__':
    # ---載入資料---
    code_snippets_df = load_code_snippets('code_snippets.csv')
    train_queries_df = pd.read_csv('train_queries.csv')

    # --- 切分訓練/驗證集 ---
    train_df, val_df = train_test_split(train_queries_df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)

    # --- 初始化模型 ---
    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)

    # --- 第一部分: 評估預訓練模型 ---
    print("--- Evaluating Pre-trained Model ---")
    pretrained_recall = evaluate_recall(model, tokenizer, val_df)
    print(f"Pre-trained Model Validation Recall@10: {pretrained_recall:.4f}")

    # --- 第二部分: 微調模型並評估 ---
    print("\n--- Fine-tuning model ---")
    fine_tuned_model = fine_tune_model(model, tokenizer, train_df, val_df, code_snippets_df, epochs=3)

    # 儲存微調後的最佳模型
    print("\nSaving the best model...")
    fine_tuned_model.save_pretrained('./fine_tuned_codebert')
    tokenizer.save_pretrained('./fine_tuned_codebert') # 保持tokenization一致性與可復現性

    print("\n--- Testing the fine-tuned model ---")
    finetuned_retriever = DenseRetriever(code_snippets_df, model_name_or_path='./fine_tuned_codebert')
    query = "add two numbers"
    top_k_finetuned = finetuned_retriever.retrieve(query)
    print(f"Query: '{query}'")
    print(f"Fine-tuned retriever top 10 results: {top_k_finetuned}")
