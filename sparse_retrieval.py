import numpy as np
from collections import Counter
from preprocess import load_code_snippets, preprocess, build_statistics, create_semantic_mapping, tokenize, lemmatize_tokens

class TFIDFRetriever:
    """TF-IDF 檢索器"""
    def __init__(self, documents):
        """初始化 TF-IDF 檢索器"""
        self.documents = documents
        # 建立詞彙庫、文件頻率等統計數據
        self.vocab, self.doc_freq, self.tokenized_docs, self.avg_doc_len = build_statistics(documents)
        self.num_docs = len(self.tokenized_docs)
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        # 計算 IDF
        self.idf = self._calculate_idf()
        # 建立文件向量
        self.doc_vectors = self._create_doc_vectors()

    def _calculate_idf(self):
        """計算 IDF 分數"""
        idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            idf[i] = np.log(self.num_docs / (self.doc_freq[word] + 1))
        return idf

    def _create_doc_vectors(self):
        """建立文件的 TF-IDF 向量"""
        doc_vectors = np.zeros((self.num_docs, len(self.vocab)))
        for i, doc in enumerate(self.tokenized_docs):
            tf = Counter(doc)
            for word, count in tf.items():
                if word in self.vocab_map:
                    doc_vectors[i, self.vocab_map[word]] = count
        return doc_vectors * self.idf

    def retrieve(self, query, k=10, query_expansion=False):
        query_tokens = tokenize(query)
        if query_expansion:
            lemmatized_tokens = lemmatize_tokens(query_tokens)
            query_tokens.extend(lemmatized_tokens)
        query_vector = np.zeros(len(self.vocab))
        tf = Counter(query_tokens)
        for word, count in tf.items():
            if word in self.vocab_map:
                query_vector[self.vocab_map[word]] = count
        
        query_vector = query_vector * self.idf
        
        # 計算餘弦相似度
        scores = np.dot(self.doc_vectors, query_vector) / (np.linalg.norm(self.doc_vectors, axis=1) * np.linalg.norm(query_vector))
        
        # 取得前 k 個結果
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices


class BM25Retriever:
    """BM25 檢索器"""
    def __init__(self, documents, k1=1.5, b=0.75):
        """初始化 BM25 檢索器"""
        self.documents = documents
        self.k1 = k1
        self.b = b
        # 建立詞彙庫、文件頻率等統計數據
        self.vocab, self.doc_freq, self.tokenized_docs, self.avg_doc_len = build_statistics(documents)
        self.num_docs = len(self.tokenized_docs)
        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        # 計算 IDF
        self.idf = self._calculate_idf()

    def _calculate_idf(self):
        """計算 BM25 的 IDF 分數"""
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
                    # 計算 BM25 分數
                    score += idf_word * (tf_word * (self.k1 + 1)) / (tf_word + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_doc_len))
            scores[i] = score
            
        # 取得前 k 個結果
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