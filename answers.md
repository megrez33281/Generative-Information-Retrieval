
# Answers to the Three Questions

## Question 1: Sparse Retrieval (TF-IDF vs. BM25)

**In Sparse Retrieval methods, compare the retrieval performance of TF-IDF and BM25. Which method performs better in this assignment? Analyze the possible reasons behind the difference (e.g., term frequency handling, document length normalization).**

In most text retrieval tasks, **BM25 is expected to outperform TF-IDF**. While I was unable to get a definitive Recall@10 score on a validation set for this specific task, the reasons for BM25's typical superiority are:

*   **Term Frequency (TF) Saturation**: TF-IDF's scoring increases linearly with the term frequency. This means a word appearing 100 times has 10 times the impact of a word appearing 10 times. BM25, on the other hand, incorporates a term frequency saturation component (controlled by the `k1` parameter). This means that after a certain point, the increase in term frequency has a diminishing return on the score. This is more aligned with how humans judge relevance; the difference between a word appearing 50 times and 100 times is less significant than the difference between 1 and 5 appearances.

*   **Document Length Normalization**: TF-IDF's normalization can be overly simplistic, sometimes over-penalizing longer documents. BM25 uses a more sophisticated normalization method (controlled by the `b` parameter) that takes into account the average document length of the entire corpus. This allows for a more nuanced normalization that is less biased against longer documents that are naturally more verbose.

Given these two major advantages, it is highly probable that the `submission_bm25.csv` will achieve a higher score on the Kaggle leaderboard than `submission_tfidf.csv`.

## Question 2: Dense Retrieval (Pre-trained vs. Fine-tuned)

**In Dense Retrieval methods, compare the performance of using a pre-trained model directly versus fine-tuning with training data. Which approach performs better? Explain the possible reasons for the difference.**

**The fine-tuned model is expected to perform significantly better** than the pre-trained model. The reasons are:

*   **Task-Specific Adaptation**: The pre-trained CodeBERT model is trained on a massive corpus of code and natural language, giving it a general understanding of code and its semantics. However, it is not specifically trained for the task of matching natural language queries to code snippets. Fine-tuning on the `train_queries.csv` dataset adapts the model to this specific task. The model learns to produce embeddings where the query and its corresponding correct code snippet are closer together in the vector space.

*   **Triplet Loss**: The use of Triplet Loss during fine-tuning is crucial. This loss function explicitly trains the model to distinguish between positive (correct) and negative (incorrect) code snippets for a given query. It pushes the embeddings of the query and the positive snippet closer together, while pushing the embeddings of the query and the negative snippets further apart. This results in a more discriminative embedding space that is optimized for retrieval.

Therefore, the `submission_finetuned.csv` is expected to have a much higher Recall@10 score than `submission_pretrained.csv`.

## Question 3: Sparse vs. Dense Retrieval and Further Improvements

**In the Text-to-Code Retrieval task, compare the differences and performance between Sparse Retrieval and Dense Retrieval. Beyond these approaches, what other methods (e.g., Retrieve-and-Re-rank) could further improve retrieval performance?**

**Differences and Performance**:

*   **Sparse Retrieval (TF-IDF, BM25)**:
    *   **How it works**: Based on keyword matching. It represents documents and queries as high-dimensional, sparse vectors where most elements are zero.
    *   **Strengths**: Efficient, easy to implement, and works well when the query contains the exact keywords present in the document.
    *   **Weaknesses**: Fails to capture semantic meaning. For example, it would not understand that "add" and "sum" are related unless explicitly told to. It also struggles with synonyms and different ways of expressing the same concept.

*   **Dense Retrieval (CodeBERT)**:
    *   **How it works**: Based on semantic similarity. It uses deep learning models to create low-dimensional, dense vectors (embeddings) that capture the meaning of the text.
    *   **Strengths**: Can understand the semantic meaning of queries and code, even if they don't share the same keywords. It can handle synonyms and more abstract queries.
    *   **Weaknesses**: More computationally expensive to set up and run. Requires a large amount of training data for fine-tuning to achieve good performance.

**Performance Comparison**: In general, for text-to-code retrieval, **dense retrieval models, especially when fine-tuned, are expected to significantly outperform sparse retrieval models**. This is because understanding the user's intent and the semantics of the code is crucial, and this is where dense models excel.

**Further Improvements**:

*   **Retrieve-and-Re-rank**: This is a powerful and common technique to improve retrieval performance. It involves a two-stage process:
    1.  **Retrieve**: Use a fast but less accurate retrieval model (like BM25) to quickly retrieve a large number of candidate documents (e.g., top 100).
    2.  **Re-rank**: Use a more powerful but slower model (like a fine-tuned CodeBERT) to re-rank the candidate documents and produce the final top-10 list. This approach combines the speed of sparse models with the accuracy of dense models.

*   **Hybrid Retrieval**: This approach combines the scores from both sparse and dense retrieval models. A simple way is to take a weighted average of the BM25 score and the cosine similarity score from the dense model. This can be very effective as it leverages the strengths of both keyword matching and semantic understanding.

*   **Better Negative Mining**: In the fine-tuning process, instead of using random negative samples, one could use "hard negatives". These are negative samples that are semantically similar to the query but are incorrect. For example, for a query about adding two numbers, a hard negative might be a function that subtracts two numbers. Using hard negatives forces the model to learn finer-grained distinctions.




baseline 部分（TF-IDF / BM25）→ 完全用 numpy 計算
額外實驗部分 → 加上 query expansion (lemmatization) 觀察效果