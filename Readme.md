
# 文字到程式碼檢索實作

本專案旨在實作一個文字到程式碼的檢索系統並比較不同的方法，其中包含了稀疏檢索（TF-IDF、BM25）和密集檢索（CodeBERT）兩種方法  

## 專案結構

```
.
├── code_snippets.csv
├── train_queries.csv
├── test_queries.csv
├── preprocess.py
├── sparse_retrieval.py
├── dense_retrieval.py
├── evaluation.py
├── requirements.txt
└── Readme.md
```

*   `code_snippets.csv`: 包含所有程式碼片段的語料庫
*   `train_queries.csv`: 用於微調密集檢索模型的訓練資料
*   `test_queries.csv`: 用於生成Kaggle提交檔案的測試查詢
*   `preprocess.py`: 包含資料前處理的相關函式
*   `sparse_retrieval.py`: 實現了TF-IDF和BM25檢索模型
*   `dense_retrieval.py`: 實現了使用預訓練和微調後CodeBERT的密集檢索模型
*   `evaluation.py`: 用於評估模型和生成提交檔案
*   `requirements.txt`: 專案所需的Python套件
*   `Readme.md`: 本文件

## 安裝相依套件

```bash
pip install -r requirements.txt
```

## TF-IDF vs. BM25 性能分析
在一個使用`train_queries.csv`自建的驗證集上，得到了以下結果：  
*   **TF-IDF Recall@10: 0.7380**
*   **BM25 Recall@10: 0.6680**
這個結果顯示，在這個特定的數據集上，TF-IDF的表現優於BM25。   
這與一般的預期可能有所不同，可能的原因如下：  
1.  **文件長度相對一致**
    BM25的文件長度正規化對於文件長度差異較大的語料庫特別有效  
    如果`train_queries.csv`中的程式碼片段長度都差不多，那麼BM25的這個優勢就無法發揮，甚至可能因為不當的懲罰而導致性能下降  
2.  **查詢較短或關鍵字單一**
    BM25的詞頻飽和度機制對於較長的查詢更有幫助  
    如果查詢本身很短，或者只包含少數幾個關鍵字，那麼詞頻的影響可能不是主要因素，TF-IDF的簡單加權方式可能反而更有效  
3.  **預設參數非最佳化**
    BM25有兩個可調參數`k1`和`b`，此處使用的是通用的預設值（k1=1.5, b=0.75）    
    這些參數可能不是這個特定數據集的最佳選擇  
    TF-IDF沒有需要調整的參數，因此不存在這個問題  


## BM25 參數調整
為了嘗試提升 BM25 的性能，有嘗試對`k1`和`b`參數進行了調整。以下是實驗結果：
| k1  | b    | Recall@10 |
| --- | ---- | --------- |
| 1.2 | 0.6  | 0.6580    |
| 1.2 | 0.75 | 0.6660    |
| 1.2 | 0.9  | 0.6620    |
| 1.5 | 0.6  | 0.6640    |
| 1.5 | 0.75 | 0.6680    |
| 1.5 | 0.9  | 0.6680    |
| 2.0 | 0.6  | 0.6700    |
| 2.0 | 0.75 | 0.6740    |
| 2.0 | 0.9  | 0.6780    | 

最佳結果是在`k1=2.0`和`b=0.9`時，Recall@10達到了 **0.6780**  
雖然比預設參數的0.6680 有所提升，但仍然低於TF-IDF的 0.7380  
這進一步證實了，在這個特定的自建驗證集上，TF-IDF 是表現最好的稀疏檢索模型  

## 執行專案
1.  **資料前處理**：
    ```bash
    python preprocess.py
    ```
2.  **稀疏檢索**：
    ```bash
    python sparse_retrieval.py
    ```
3.  **密集檢索**：
    ```bash
    python dense_retrieval.py
    ```
4.  **評估與提交**：
    ```bash
    python evaluation.py
    ```

## N-gram 實驗
嘗試在斷詞時加入N-gram(bigrams 和 trigrams)來捕捉更多的上下文資訊。以下是自建驗證集上的結果：
*   **Unigrams (1-gram)**:
    *   TF-IDF Recall@10: 0.7380
    *   BM25 Recall@10: 0.6680
*   **Bigrams (1-gram + 2-gram)**:
    *   TF-IDF Recall@10: 0.7300
    *   BM25 Recall@10: 0.6700
*   **Trigrams (1-gram + 2-gram + 3-gram)**:
    *   TF-IDF Recall@10: 0.7320
    *   BM25 Recall@10: 0.6700
從結果來看，加入N-gram並沒有提升稀疏模型的性能，反而略有下降
這可能表示對於這個數據集，額外的上下文資訊並沒有帶來好處，甚至可能引入了噪音


## 查詢擴展 (Query Expansion) 實驗
嘗試了使用詞形還原（Lemmatization）來進行查詢擴展   
當查詢進入時，將**原始查詢詞和其詞形還原後的詞**都納入考量，以期捕捉更多相關的程式碼片段，以下是自建驗證集上的結果：  
*   **Unigrams (無查詢擴展)**:
    *   TF-IDF Recall@10: 0.7380
    *   BM25 Recall@10: 0.6680
*   **Unigrams (有查詢擴展)**:
    *   TF-IDF Recall@10: 0.7460
    *   BM25 Recall@10: 0.6720
從結果來看，查詢擴展對TF-IDF和BM25的性能都有輕微的提升  
TF-IDF從0.7380提升到0.7460，BM25從0.6680提升到0.6720  
這表明詞形還原在一定程度上幫助模型捕捉了更多的語義相關性  