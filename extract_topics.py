from __future__ import annotations
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from numpy import ndarray
from typing import List, Dict, Any
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from logging import Logger


def topic_extraction_with_lda(
    X: csr_matrix,
    vectorizer: CountVectorizer,
    logger: Logger,
    *,
    n_topics: int = 10,
    n_top_terms: int = 15,
    n_iter: int = 20,
    learning_method: str = "batch",      # since in project I'm using a relatively small data set, batch method is better
    learning_decay: float = 0.7,         # used for "online"
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[LatentDirichletAllocation, ndarray, list[list[str]], list[ndarray]]:

    """
    Extract topics using Latent Dirichlet Allocation (LDA) from a BoW matrix.

    :param X: Document-term matrix from CountVectorizer (BoW COUNTS)
    :param vectorizer: The fitted CountVectorizer used to create X
    :param n_topics: Number of topics (components) to learn
    :param n_top_terms:Number of top words to return per topic for readability
    :param n_iter: Maximum number of EM iterations
    :param learning_method: LDA learning method (sklearn). "online" is faster on very large corpora
    :param learning_decay: Learning rate decay for "online" method
    :param random_state: Random seed for reproducibility
    :param n_jobs: Threads for parallelization
    :param logger: logger object, used for logging

    :returns lda: The fitted LDA model
    :returns doc_topic: Per-document topic distribution
    :returns topics: Human-readable list of topics[k] is the top `n_top_words` terms for topic k
    :returns topic_word_weights: Raw word weights per topic
    """

    #Checks the input data
    if not hasattr(vectorizer, "get_feature_names_out"):
        raise ValueError("`vectorizer` must be a fitted CountVectorizer with get_feature_names_out().")
    if not isinstance(X, csr_matrix):
        X = X.tocsr()

    #Logs input matrix shape and the number of topics
    n_docs, n_terms = X.shape
    logger.info(f"Fitting LDA on BoW matrix with shape={X.shape} "
                f"(docs={n_docs:,}, terms={n_terms:,}), topics={n_topics}...")

    #Starts the learning process for LDA with all parameters
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=n_iter,
        learning_method=learning_method,
        learning_decay=learning_decay,
        random_state=random_state,
        n_jobs=n_jobs,
        evaluate_every=-1,
        verbose=0,
    )
    doc_topic = lda.fit_transform(X)   #learns the model and returns distribution of topics among documents matrix
    logger.info("LDA fit complete.")

    #Prepares the list of all features and forms a list of to store topic-feature list
    feature_names = vectorizer.get_feature_names_out()
    topics: List[List[str]] = []
    topic_word_weights: List[np.ndarray] = []

    #Forms the list of the top-k features for each topic
    for k, comp in enumerate(lda.components_):   # shape (n_topics, n_terms)
        # larger value -> word is more representative of topic
        top_idx = np.argsort(comp)[-n_top_terms:][::-1]
        top_terms = [feature_names[i] for i in top_idx]
        topics.append(top_terms)
        topic_word_weights.append(comp[top_idx])
        logger.info(f"Topic #{k:02d}: " + ", ".join(top_terms))

    #Logs the shape of doc-topic matrix
    logger.info("Document-topic matrix shape: %s", doc_topic.shape)

    return lda, doc_topic, topics, topic_word_weights


def topic_extraction_with_lsa(
    logger: Logger,
    X: csr_matrix,
    feature_names: List[str],
    *,
    n_topics: int = 20,
    n_top_terms: int = 15,
    normalize_doc_topics: bool = True,
    random_state: int = 42,
    n_iter: int = 7,
) -> tuple[TruncatedSVD, Any, list[dict]]:
    """
    Extracts topics from the matrixm using LSA
    :param logger: logger object, used for logging
    :param X: document-term matrix from CountVectorizer (BoW COUNTS)
    :param feature_names: names of terms from matrix
    :param n_topics: Number of topics (components) to learn
    :param n_top_terms: Number of top words to return per topic for readability
    :param normalize_doc_topics: If true, all vectors in resulting matrix are converted to the length 1 (normalized)
    :param random_state: Random seed for reproducibility
    :param n_iter: Maximum number of EM iterations

    :returns svd: fitted SVD-model (LSA method)
    :returns doc_topic: coordinates of the documents within the space of themes
    :returns topics: list of topics with ids, top-terms, and weights of terms
    """

    #Logging the start time
    t0 = time.time()
    logger.info(
        "LSA started: n_topics=%s, n_top_terms=%s, normalize_doc_topics=%s",
        n_topics, n_top_terms, normalize_doc_topics
    )

    #Starting the learning process of LSA model
    svd = TruncatedSVD(
        n_components=n_topics,
        n_iter=n_iter,
        random_state=random_state,
    )
    doc_topic = svd.fit_transform(X)  # (n_docs × n_topics)

    #Optionally normalizing the doc-topic matrix
    if normalize_doc_topics:
        doc_topic = Normalizer(copy=False).fit_transform(doc_topic)

    #Extracting top-terms for each topic based on the weight and the number of top-terms
    components = svd.components_  # (n_topics × n_features)
    topics: List[Dict] = []
    for t_idx in range(components.shape[0]):
        comp = components[t_idx]
        # Сортировка по убыванию вклада терма в компоненту
        top_indices = np.argsort(comp)[::-1][:n_top_terms]
        terms_with_weights = [(feature_names[i], float(comp[i])) for i in top_indices]
        topics.append({
            "topic_id": int(t_idx),
            "terms": terms_with_weights
        })

    #Logging the form of doc-topic
    elapsed = time.time() - t0
    logger.info(
        "LSA finished: n_topics=%s, doc_topic_shape=%s, explained_var=%.4f, time=%.2fs",
        n_topics, doc_topic.shape, svd.explained_variance_ratio_.sum(), elapsed
    )

    return svd, doc_topic, topics
