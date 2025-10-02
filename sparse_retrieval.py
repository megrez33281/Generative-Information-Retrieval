import numpy as np
from collections import Counter
from preprocess import load_code_snippets, preprocess, build_statistics, create_semantic_mapping, tokenize, lemmatize_tokens

class TFIDFRetriever:
    """TF-IDF檢索器"""
    def __init__(self, documents):
        """初始化TF-IDF檢索器"""
        self.documents = documents
        # 建立詞彙庫、文件頻率等統計數據
        self.vocab, self.doc_freq, self.tokenized_docs, self.avg_doc_len = build_statistics(documents)
        self.num_docs = len(self.tokenized_docs)

        # 把詞彙表（vocab）裡的每個單詞對應到一個整數索引，之後可以用來查詢詞彙對應的索引
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        # 計算IDF
        self.idf = self._calculate_idf()
        # 建立TF-IDF向量
        self.doc_vectors = self._create_doc_vectors()

    def _calculate_idf(self):
        """用公式計算IDF分數"""
        idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            idf[i] = np.log(self.num_docs / (self.doc_freq[word] + 1)) # +1避免分母為0
        return idf

    def _create_doc_vectors(self):
        """建立每篇文件的TF-IDF向量"""

        # 建立一個全零矩陣，大小是(文件數量 × 詞彙表大小)
        doc_vectors = np.zeros((self.num_docs, len(self.vocab)))
        for i, doc in enumerate(self.tokenized_docs):
            # doc代表每個文件斷詞後的詞彙列表，利用Counter計算每個詞彙在該文件中出現的次數
            tf = Counter(doc)
            for word, count in tf.items():
                if word in self.vocab_map:
                    # 將詞彙頻率放在對應位置（i代表第i個文件，self.vocab_map[word]是該詞彙在詞彙表中的索引）
                    doc_vectors[i, self.vocab_map[word]] = count
        return doc_vectors * self.idf # 矩陣或向量中對應位置的元素進行乘法運算

    def retrieve(self, query, k=10, query_expansion=False):
        """進行query，返回文件在資料庫中的索引"""
        query_tokens = tokenize(query)
        if query_expansion:
            # 擴展功能，可選擇是否對token進行詞型還原
            lemmatized_tokens = lemmatize_tokens(query_tokens)
            query_tokens.extend(lemmatized_tokens)

        # 計算query的TF
        query_vector = np.zeros(len(self.vocab))
        tf = Counter(query_tokens)
        for word, count in tf.items():
            if word in self.vocab_map:
                query_vector[self.vocab_map[word]] = count
        # 計算query的TF-IDF
        query_vector = query_vector * self.idf
        
        # 計算餘弦相似度（cosine公式）
        # np.linalg.norm：計算範數（預設為L2）
        scores = np.dot(self.doc_vectors, query_vector) / (np.linalg.norm(self.doc_vectors, axis=1) * np.linalg.norm(query_vector))
        
        # 取得前k個結果
        top_k_indices = np.argsort(scores)[::-1][:k]

        """
        TF衡量詞在文件內的重要性
        IDF衡量詞在整個語料庫中的稀有程度（越稀有越重要）
        TF×IDF = 詞在文件中的加權重要性 → 形成TF-IDF向量，高分的詞通常是該文件獨有且重要的詞，在計算相似度時的貢獻也更大
        """
        return top_k_indices


class BM25Retriever:
    """BM25 檢索器"""
    def __init__(self, documents, k1=1.5, b=0.75):
        """初始化 BM25 檢索器"""
        self.documents = documents

        # k1較大時，高詞頻的詞對分數貢獻增加，飽和程度降低 => 讓高頻詞影響更大
        # k1較小時，高詞頻對分數的邊際影響降低，快速飽和 => 避免長文本裡同一個詞過度加分
        self.k1 = k1

        # 當 b=1 → 完全使用長度正規化，長文件的詞頻被縮小
        # 當 b=0 → 不考慮文件長度，所有文件同樣計算
        self.b = b

        # 建立詞彙庫、文件頻率等統計數據
        self.vocab, self.doc_freq, self.tokenized_docs, self.avg_doc_len = build_statistics(documents)
        self.num_docs = len(self.tokenized_docs)
        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        # 計算 IDF
        self.idf = self._calculate_idf()

    def _calculate_idf(self):
        """計算BM25的IDF分數"""
        idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            idf[i] = np.log(((self.num_docs - self.doc_freq[word] + 0.5) / (self.doc_freq[word] + 0.5)) + 1)
        return idf

    def retrieve(self, query, k=10, query_expansion=False):
        """根據查詢檢索文件"""
        query_tokens = tokenize(query)
        if query_expansion:
            lemmatized_tokens = lemmatize_tokens(query_tokens)
            query_tokens.extend(lemmatized_tokens)
        scores = np.zeros(self.num_docs)
        
        for i in range(self.num_docs):
            tf = Counter(self.tokenized_docs[i])
            score = 0
            for word in query_tokens:
                if word in self.vocab_map:
                    tf_word = tf[word]
                    idf_word = self.idf[self.vocab_map[word]]
                    # 計算BM25分數
                    score += idf_word * (tf_word * (self.k1 + 1)) / (tf_word + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_doc_len))
            scores[i] = score
            
        # 取得前k個結果
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices

if __name__ == '__main__':
    # 載入並預處理資料
    code_snippets_df = load_code_snippets('code_snippets.csv')
    processed_df = preprocess(code_snippets_df)

    # 初始化檢索器
    tfidf_retriever = TFIDFRetriever(processed_df)
    bm25_retriever = BM25Retriever(processed_df)

    # 範例查詢
    query = "add two numbers"
    
    # TF-IDF
    tfidf_top_k = tfidf_retriever.retrieve(query)
    print(f"查詢: '{query}'")
    print(f"TF-IDF 前 10 個檢索到的文件 ID: {tfidf_top_k}")

    # BM25
    bm25_top_k = bm25_retriever.retrieve(query)
    print(f"BM25 前 10 個檢索到的文件 ID: {bm25_top_k}")