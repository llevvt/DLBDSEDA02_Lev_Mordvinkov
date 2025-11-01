from __future__ import annotations

import os
import pandas as pd
import logging
from dotenv import load_dotenv
from normalize import normalize_dataset
from vectorize import vectorize_with_bow, vectorize_with_tfidf, matrix_stats
from extract_topics import topic_extraction_with_lda, topic_extraction_with_lsa
from helpers import _ensure_dir, save_vectorization, save_topics, save_comparison_overview
import numpy as np

if __name__ == "__main__":
    #Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    #Loading .env values
    load_dotenv()
    ARTIFACTS_DIR = _ensure_dir(os.getenv("OUTPUT_DIR", "artifacts"))
    min_token_size = int(os.getenv('MIN_TOKEN_LEN'))
    vocabulary_size = int(os.getenv('VOCABULARY_SIZE'))
    min_n_gram = int(os.getenv('MIN_N_GRAM'))
    max_n_gram = int(os.getenv('MAX_N_GRAM'))
    ngram_range = (min_n_gram, max_n_gram)
    min_freq = int(os.getenv('MIN_FREQUENCY'))
    n_topics = int(os.getenv('NUMBER_OF_TOPICS'))
    n_top_words = int(os.getenv('NUMBER_OF_TOP_WORDS'))
    n_iterations = int(os.getenv('NUMBER_OF_ITERATIONS'))
    random_state = int(os.getenv('RANDOM_STATE'))

    #Creating or loading file for normalized texts
    if not os.path.exists(os.getenv('NORMALIZED_TEXTS')):
        normalize_dataset(
            csv_path=os.getenv('RAW_DATASET'),
            output_csv=os.getenv('NORMALIZED_TEXTS'),
            logger=logger,
            min_token_len=min_token_size)
    df = pd.read_csv(os.getenv('NORMALIZED_TEXTS'))
    df = df['narrative'].astype(str)

    #Vectorize with BoW
    X_bow, vectorizer_bow, features_bow = vectorize_with_bow(
        texts=df,
        logger=logger,
        ngram_range=ngram_range,
        max_features=vocabulary_size,
        min_df=min_freq
    )

    #Saving results of vectorization with BoW
    meta_bow = save_vectorization(
        method="bow",
        X=X_bow,
        feature_names=features_bow,
        vectorizer=vectorizer_bow,
        outdir=ARTIFACTS_DIR,
    )
    logger.info("Saved BoW artifacts: %s", meta_bow)

    #Topic Extraction with LDA
    lda, doc_topic, topics, topic_word_weights = topic_extraction_with_lda(
        X=X_bow,
        vectorizer=vectorizer_bow,
        logger=logger,
        n_topics=n_topics,
        n_top_terms=n_top_words,
        n_iter=n_iterations,
        learning_method="batch",
        random_state=random_state
    )

    #Saving results of Topic Extraction with LDA based on BoW
    meta_lda = save_topics(
        method="lda_bow",
        topics=topics,
        doc_topic=doc_topic,
        outdir=ARTIFACTS_DIR,
    )
    logger.info("Saved LDA artifacts: %s", meta_lda)

    #vectorize with TF-IDF
    X_tfidf, vectorizer_tfidf, features_tfidf = vectorize_with_tfidf(
        logger=logger,
        texts=df,
        max_features=vocabulary_size,
        ngram_range=ngram_range,
        min_df=min_freq
    )

    #Saving results of TF-IDF vectorization
    meta_tfidf = save_vectorization(
        method="tfidf",
        X=X_tfidf,
        feature_names=features_tfidf,
        vectorizer=vectorizer_tfidf,
        outdir=ARTIFACTS_DIR,
    )
    logger.info("Saved TF-IDF artifacts: %s", meta_tfidf)

    #Topic Extraction with LSA
    svd, doc_topic, topics = topic_extraction_with_lsa(
        logger=logger,
        X=X_tfidf,
        feature_names=features_tfidf,
        n_topics=n_topics,
        n_top_terms=n_top_words,
        normalize_doc_topics=True,
        random_state=random_state,
        n_iter=n_iterations,
    )

    #Saving results of LSA topic extraction based on TF-IDF vectorization
    meta_lsa = save_topics(
        method="lsa_tfidf",
        topics=topics,
        doc_topic=doc_topic,
        outdir=ARTIFACTS_DIR,
    )
    logger.info("Saved LSA artifacts: %s", meta_lsa)

    #Comparing Vectorization Results
    comparison_lines = []
    comparison_lines.append("=== VECTORISATION COMPARISON REPORT ===\n")

    comparison_lines.append(matrix_stats(X_bow, "BoW"))
    comparison_lines.append(matrix_stats(X_tfidf, "TF-IDF"))
    comparison_lines.append("\nTop 5 terms by mean weight:\n")

    bow_means = np.asarray(X_bow.mean(axis=0)).ravel()
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()

    top5_bow_idx = bow_means.argsort()[-5:][::-1]
    top5_tfidf_idx = tfidf_means.argsort()[-5:][::-1]

    top5_bow_terms = [features_bow[i] for i in top5_bow_idx]
    top5_tfidf_terms = [features_tfidf[i] for i in top5_tfidf_idx]

    comparison_lines.append("BoW top-5 terms: " + ", ".join(top5_bow_terms))
    comparison_lines.append("TF-IDF top-5 terms: " + ", ".join(top5_tfidf_terms))

    comparison_path = os.path.join(ARTIFACTS_DIR, "vectorization_comparison.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("\n".join(comparison_lines))

    logger.info(f"Saved vectorization comparison report to {comparison_path}")

    #Saving comparison results
    save_comparison_overview(methods=["lda_bow", "lsa_tfidf"], outdir=ARTIFACTS_DIR)
    logger.info("Created comparison overview CSV")
