import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from preprocess import load_code_snippets
import numpy as np
import pandas as pd

class TripletDataset(Dataset):
    """三元組數據集，用於微調密集檢索模型"""
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
        # 錨點 (anchor) 是查詢
        anchor_text = self.queries_df.iloc[idx]['query']
        # 正樣本 (positive) 是與查詢對應的正確程式碼
        positive_code = self.queries_df.iloc[idx]['code']

        # 負樣本 (negative) 是隨機選擇的錯誤程式碼
        # 這裡使用了簡化的隨機負採樣，更進階的方法是使用「困難負樣本」
        negative_code = self.snippets_df.sample(1)['code'].values[0]

        # 對錨點、正樣本和負樣本進行斷詞和編碼
        anchor_inputs = self.tokenizer(anchor_text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        positive_inputs = self.tokenizer(positive_code, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        negative_inputs = self.tokenizer(negative_code, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)

        return {
            'anchor': {key: val.squeeze() for key, val in anchor_inputs.items()},
            'positive': {key: val.squeeze() for key, val in positive_inputs.items()},
            'negative': {key: val.squeeze() for key, val in negative_inputs.items()}
        }

def fine_tune_model(model, tokenizer, train_queries_df, code_snippets_df, epochs=1, lr=2e-5, batch_size=8):
    """微調預訓練模型"""
    dataset = TripletDataset(train_queries_df, code_snippets_df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 設定優化器和損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    # 訓練模型
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # 獲取錨點、正樣本和負樣本的嵌入向量
            anchor_embeddings = model(**batch['anchor']).last_hidden_state.mean(dim=1)
            positive_embeddings = model(**batch['positive']).last_hidden_state.mean(dim=1)
            negative_embeddings = model(**batch['negative']).last_hidden_state.mean(dim=1)

            # 計算損失
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    return model

class DenseRetriever:
    """密集檢索器"""
    def __init__(self, documents, model_name_or_path='microsoft/codebert-base'):
        """初始化密集檢索器"""
        self.documents = documents
        # 載入預訓練模型和斷詞器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        # 建立文件的嵌入向量
        self.doc_embeddings = self._create_doc_embeddings()

    def _create_doc_embeddings(self):
        """建立所有文件的嵌入向量"""
        embeddings = []
        for doc in self.documents['code']:
            inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 使用最後一層隱藏層的平均值作為嵌入向量
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)

    def retrieve(self, query, k=10):
        """根據查詢檢索文件"""
        inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 獲取查詢的嵌入向量
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # 計算餘弦相似度
        scores = np.dot(self.doc_embeddings, query_embedding) / (np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding))

        # 取得前 k 個結果
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices

if __name__ == '__main__':
    # 載入資料
    code_snippets_df = load_code_snippets('code_snippets.csv')
    train_queries_df = pd.read_csv('train_queries.csv')

    # --- 第一部分: 使用預訓練模型 ---
    print("--- 使用預訓練模型 ---")
    pretrained_retriever = DenseRetriever(code_snippets_df)
    query = "add two numbers"
    top_k = pretrained_retriever.retrieve(query)
    print(f"查詢: '{query}'")
    print(f"密集檢索器 (預訓練) 前 10 個檢索到的文件 ID: {top_k}")

    # --- 第二部分: 微調模型 ---
    print("\n--- 微調模型 ---")
    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 微調模型
    fine_tuned_model = fine_tune_model(model, tokenizer, train_queries_df, code_snippets_df, epochs=3)

    # 儲存微調後的模型
    fine_tuned_model.save_pretrained('./fine_tuned_codebert')
    tokenizer.save_pretrained('./fine_tuned_codebert')

    print("\n--- 使用微調後的模型 ---")
    finetuned_retriever = DenseRetriever(code_snippets_df, model_name_or_path='./fine_tuned_codebert')
    top_k_finetuned = finetuned_retriever.retrieve(query)
    print(f"查詢: '{query}'")
    print(f"密集檢索器 (微調後) 前 10 個檢索到的文件 ID: {top_k_finetuned}")