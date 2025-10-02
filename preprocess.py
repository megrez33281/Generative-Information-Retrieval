import pandas as pd
import re
from collections import Counter

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """對 token 列表進行詞形還原"""
    return [lemmatizer.lemmatize(token) for token in tokens]

def load_code_snippets(file_path):
    """從 CSV 檔案載入程式碼片段"""
    return pd.read_csv(file_path)

def tokenize(text, n_gram_range=(1, 1), keep_both=True):
    """
    將程式碼文本進行斷詞，並可選擇性地生成 n-grams
    - n_gram_range: (min_n, max_n)，例如 (1,2) 會同時生成unigram與bigram
    - keep_both: 是否同時保留"空格形式"與"原樣形式"
    """
    # 保留運算符作為獨立的 token
    tokens = re.findall(r'\w+|==|!=|<=|>=|[\+\-\*/=<>!&|%\^~]', text)
    tokens = [token.lower() for token in tokens]

    # 如果只要 unigram
    if n_gram_range == (1, 1):
        return tokens

    ngrams = []
    for n in range(n_gram_range[0], n_gram_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            slice_tokens = tokens[i:i+n]
            if keep_both:
                # 空格形式（以空格進行連接，但可能造成詞語的改變）
                ngrams.append(" ".join(slice_tokens))
                # 原樣形式直接連起來）
                ngrams.append("".join(slice_tokens))
            else:
                ngrams.append(" ".join(slice_tokens))
    return ngrams

def preprocess(df, n_gram_range=(1, 1)):
    """對程式碼片段進行預處理"""
    df['tokens'] = df['code'].apply(lambda x: tokenize(x, n_gram_range=n_gram_range))
    return df

def create_semantic_mapping():
    """建立從自然語言到程式碼運算符的語意映射"""

    # 考慮到此處的程式碼為python，運算符號本身存在意義，需要另外保留運算符（+-*/等）作為獨立的Token，並為自然語言的查詢建立對應的映射，使其能與運算符匹配
    return {
        "add": "+", "sum": "+", "plus": "+", "addition": "+",
        "concatenate": "+", "join": "+",
        "assign": "=", "set": "=",
        "subtract": "-", "minus": "-", "subtraction": "-",
        "multiply": "*", "times": "*", "multiplication": "*",
        "divide": "/", "division": "/",
        "equals": "==", "is": "==",
        "less": "<", "smaller": "<",
        "greater": ">", "larger": ">",
    }

def build_statistics(processed_df):
    """建立詞彙庫和其他統計數據"""
    tokenized_docs = processed_df['tokens'].tolist()
    
    # 文件頻率，計算每個Token出現在多少篇文件中
    doc_freq = Counter()
    for doc in tokenized_docs:
        # 遍歷每個doc
        doc_freq.update(set(doc))

    # 詞彙庫
    vocab = list(doc_freq.keys())
    
    # token總數
    total_tokens = sum(len(doc) for doc in tokenized_docs)
    
    # 文件總數
    num_docs = len(tokenized_docs)
    
    # 平均文件長度
    avg_doc_len = total_tokens / num_docs if num_docs > 0 else 0
    
    return vocab, doc_freq, tokenized_docs, avg_doc_len

if __name__ == '__main__':
    # 載入資料
    code_snippets_df = load_code_snippets('code_snippets.csv')

    # 預處理資料
    processed_df = preprocess(code_snippets_df)

    # 建立統計數據
    vocab, doc_freq, tokenized_docs, avg_doc_len = build_statistics(processed_df)

    # 建立語意映射
    semantic_mapping = create_semantic_mapping()

    # 印出一些統計數據
    print(f"詞彙庫大小: {len(vocab)}")
    print(f"文件總數: {len(tokenized_docs)}")
    print(f"平均文件長度: {avg_doc_len:.2f}")
    print("\n前 10 個最常見的 token:")
    print(doc_freq.most_common(10))
    print("\n語意映射:")
    print(semantic_mapping)