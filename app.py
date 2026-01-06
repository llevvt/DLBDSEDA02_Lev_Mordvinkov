from __future__ import annotations

import os
import pandas as pd
import logging
from dotenv import load_dotenv
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import ndarray
from typing import List, Dict, Any
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import re
from typing import Optional, Iterable, Tuple
import contractions
import spacy
import json
from datetime import datetime
from pathlib import Path


#Support functionality
def _ensure_dir(path: str) -> Path:
    """
    Checking the existence of the directory, which is used to save the results and outputs

    :param path: string, which contains the resulting directory path
    :return p: ready-to-use directory, created with path
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_vectorization(method: str,
                       X,
                       feature_names: list[str],
                       vectorizer,
                       outdir: str) -> dict:
    """
    Save artifacts of the vectorization:
      - matrix X (NPZ)
      - the list of features (CSV)
      - short description (JSON)
    Returns resume dictionary for further processing.

    :param method: method of vectorization
    :param X: results matrix
    :param feature_names: features from the matrix (rows)
    :param vectorizer: the object, which was used for vectorization
    :param outdir: the directory for results saving
    :returns meta: a dictionary with brief description of the saved vectorization data
    """

    #Ensures, that the path for the results directory exists
    outdir = _ensure_dir(outdir)

    stamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    #Saves matrix
    save_npz(outdir / f"X_{method}.npz", X)

    #Saves features
    pd.DataFrame({"feature": feature_names}).to_csv(outdir / f"features_{method}.csv", index=False)

    # Serialize vectorization parameters
    params = {}
    try:
        for k, v in vectorizer.get_params().items():
            if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
                params[k] = v
            else:
                params[k] = str(v)
    except Exception:
        params = {}

    meta = {
        "method": method,
        "shape": list(X.shape),
        "nnz": int(X.nnz),
        "n_features": len(feature_names),
        "vectorizer_type": type(vectorizer).__name__,
        "vectorizer_params": params,
        "created_utc": stamp,
    }

    #Saves meta data of the vectorization
    with open(outdir / f"vectorization_{method}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _to_term_weight_pairs(items) -> List[Tuple[str, float|None]]:
    """
    Normalizes the list of term-weight pairs to unified form.

    :param items: a list of terms, which can contain different data types
    :returns pairs: a tuple of pairs of term-weight values
    """

    pairs = []

    #Goes through each row in terms, forms pairs and adds them to the output variable
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            term, weight = it[0], it[1]
        else:
            term, weight = it, None
        try:
            w = None if weight is None else float(weight)
        except Exception:
            w = None
        pairs.append((str(term), w))
    return pairs


def normalize_topics_input(topics: List[Dict] | List[Tuple[int, List[Tuple[str, float]]]] | List[List[Tuple[str, float]]]) -> List[Dict]:
    """
    Normalizes the diverse topics representation formats to provide unified format for further processing

    :param topics: this is a list of different objects, that contains topics descriptions
    :returns normalized: a normalized list of topics
    """
    normalized = []

    #Checks that topics is a list
    if not isinstance(topics, list):
        return normalized

    # Processes topics in case if topics is a List[Dict]
    if topics and isinstance(topics[0], dict):
        for i, t in enumerate(topics):
            terms = t.get("terms", t.get("top_terms", t.get("words", [])))
            topic_id = t.get("topic_id", t.get("id", i))
            normalized.append({
                "topic_id": int(topic_id),
                "terms": _to_term_weight_pairs(terms),
            })
        return normalized

    # Processes topics in case if topics is a List[Tuple[int, List[...]]
    if topics and isinstance(topics[0], (list, tuple)) and len(topics[0]) == 2 \
       and isinstance(topics[0][1], (list, tuple)):
        for pair in topics:
            topic_id, terms = pair
            normalized.append({
                "topic_id": int(topic_id),
                "terms": _to_term_weight_pairs(terms),
            })
        return normalized

    # Processes topics in case if each element does not have an ID (topics is a List[List[...]])
    if topics and isinstance(topics[0], (list, tuple)):
        for i, terms in enumerate(topics):
            normalized.append({
                "topic_id": i,
                "terms": _to_term_weight_pairs(terms),
            })
        return normalized

    #Returns empty list in case if topics does not suit for any of listed cases
    return normalized


def save_topics(method: str,
                topics,
                doc_topic,
                outdir: str,
                top_k_overview: int = 10) -> dict:
    """
    Processes, normalizes, and saves artefacts

    :param method: the name of topic extraction method
    :param doc_topic: raw topics list in one of the supported formats
    :param doc_topic: distribution of topics among documents matrix
    :param outdir: path to the output directory
    :param top_k_overview: terms per topic rate
    :returns meta: short resume (also saved in JSON file)
    """

    #Preparation for saving
    outdir = _ensure_dir(outdir)
    stamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    #Normalizes topics
    norm_topics = normalize_topics_input(topics)
    if not norm_topics:
        raise ValueError(
            f"Unsupported topics structure for method '{method}'. "
            f"Expected list of dicts/tuples/lists; got: {type(topics).__name__}"
        )

    #Long-list of topics with weights and ranks
    rows = []
    for t in norm_topics:
        tid = int(t["topic_id"])
        for rank, (term, weight) in enumerate(t["terms"], start=1):
            rows.append({
                "method": method,
                "topic_id": tid,
                "rank": rank,
                "term": term,
                "weight": (None if weight is None else float(weight)),
            })
    df_terms = pd.DataFrame(rows)
    df_terms.to_csv(outdir / f"topics_terms_{method}.csv", index=False)

    #Forms overview topic representation with topic id and best k terms for each topic
    overview = []
    for t in norm_topics:
        top_terms = [w[0] for w in t["terms"][:top_k_overview]]
        overview.append({
            "method": method,
            "topic_id": int(t["topic_id"]),
            f"top_{top_k_overview}_terms": ", ".join(top_terms)
        })
    df_overview = pd.DataFrame(overview).sort_values(["method", "topic_id"])
    df_overview.to_csv(outdir / f"topics_overview_{method}.csv", index=False)

    #Saves doc-topic matrix
    dt = doc_topic.toarray() if hasattr(doc_topic, "toarray") else np.asarray(doc_topic)
    dt_df = pd.DataFrame(dt, columns=[f"topic_{i:02d}" for i in range(dt.shape[1])])
    dt_df.index.name = "doc_id"
    try:
        dt_df.to_parquet(outdir / f"doc_topic_{method}.parquet", index=True)
        doc_topic_path = str(outdir / f"doc_topic_{method}.parquet")
    except Exception:
        dt_df.to_csv(outdir / f"doc_topic_{method}.csv", index=True)
        doc_topic_path = str(outdir / f"doc_topic_{method}.csv")

    #Forms and saves meta-data to JSON file
    meta = {
        "method": method,
        "n_topics": int(dt.shape[1]),
        "n_docs": int(dt.shape[0]),
        "doc_topic_path": doc_topic_path,
        "created_utc": stamp,
    }
    with open(outdir / f"topics_{method}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def save_comparison_overview(methods: list[str],
                             outdir: str,
                             filename: str = "topics_overview__comparison.csv") -> None:
    """
    Function joins all individual topics overview tables and saves them into one CSV file

    :param methods: list of topics extraction methods, used for analysis
    :param outdir: path to the output directory
    :param filename: output file for comparison CSV saving
    :returns None
    """
    #Ensures that output directory exists
    outdir = _ensure_dir(outdir)

    frames = []

    #Checks, if for ech method in list an overview file exists
    for m in methods:
        path = outdir / f"topics_overview_{m}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))

    #Joins all detected overview files into one CSV file
    if frames:
        pd.concat(frames, axis=0, ignore_index=True).to_csv(outdir / filename, index=False)


#Vectorization functionality
def vectorize_with_bow(
    logger: logging.Logger,
    texts: Iterable[str],
    *,
    max_features: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    binary: bool = False,
    lowercase: bool = False,
    strip_accents: str = None,
    vocabulary: dict[str, int] = None,
) -> tuple[csr_matrix, object, list[str]]:
    """
    Vectorizes the normalized collection of texts into Bag-of-Words matrix.

    :param logger: Logger object for logging
    :param texts: Normalized texts for vectorization
    :param max_features: Maximum vocabulary size
    :param ngram_range: Range of n-grams to extract
    :param min_df: Ignore terms with document frequency below this threshold
    :param binary: If True, use 0/1 presence instead of term counts. For the project I do not need binary matrix,
    therefore it is False by default
    :param lowercase: Parameter to normalize the text and make it lowercase. Texts are already normalized,
    therefore False by default
    :param strip_accents: Option to normalize texts. Since texts are already normalized, None by default
    :param vocabulary: Optional vocabulary. Of provided, vectorizer does not learn, just processes the provided
    vocabulary. Project requires full circle, therefore the default value is None

    :returns X: Sparse document-term matrix
    :returns vectorizer: The fitted vectorizer
    :returns feature_names: Vocabulary terms in the same order as columns of X
    """

    #Ensures that all elements of texts are strings and logs parameters
    texts = [t if isinstance(t, str) else "" for t in texts]
    n_docs = len(texts)
    logger.info(f"Starting BoW vectorization for {n_docs:,} documents...")
    logger.info(f"Parameters: max_features={max_features}, ngram_range={ngram_range}, "
                f"min_df={min_df}, binary={binary}")

    #Initializes the vectorizer object
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        binary=binary,
        lowercase=lowercase,
        strip_accents=strip_accents,
        vocabulary=vocabulary,
        token_pattern=r"(?u)\b[a-z]{2,}\b",  # restricts tokens to min 2 letters
    )

    t0 = time.time()

    #Starts the learning process
    if vocabulary is None:
        logger.info("Fitting vocabulary and transforming texts...")
        X = vectorizer.fit_transform(texts)
        logger.info("Vocabulary successfully fitted.")
    else:
        logger.info("Using existing vocabulary for transformation only...")
        X = vectorizer.transform(texts)
        logger.info("Transformation complete.")

    #Extracts the name of features
    feature_names = list(vectorizer.get_feature_names_out())
    elapsed = time.time() - t0
    logger.info(f"Vectorization finished: shape={X.shape}, features={len(feature_names):,}, "
                f"time={elapsed:.2f}s")

    return X.tocsr(), vectorizer, feature_names


def vectorize_with_tfidf(
        logger: logging.Logger,
        texts: Iterable[str],
        *,
        max_features: int,
        ngram_range: Tuple[int, int],
        min_df,
        sublinear_tf: bool = True,
        norm: str = "l2",
        lowercase: bool = False,
        strip_accents: str = None,
        vocabulary: dict[str, int] = None,
        use_idf: bool = True,
        smooth_idf: bool = True,
        dtype="float32",
) -> tuple[csr_matrix, TfidfVectorizer, list[str]]:
    """
    Vectorizes the normalized collection of texts into TF-IDF matrix.

    :param logger: Logger object for logging
    :param texts: Normalized texts for vectorization
    :param max_features: Maximum vocabulary size
    :param ngram_range: Range of n-grams to extract
    :param min_df: Ignore terms with document frequency below this threshold
    :param sublinear_tf: Log-transfromation of the TF value to make frequent words less 'heavy'
    :param norm: sets the normalization mode for TF-IDF vectors normalization. If normalization is applied, documents
    with different lengths canbe compared more accurate.
    :param lowercase: Parameter to normalize the text and make it lowercase. Texts are already normalized,
    therefore False by default
    :param strip_accents: Option to normalize texts. Since texts are already normalized, None by default
    :param vocabulary: Optional vocabulary. Of provided, vectorizer does not learn, just processes the provided
    vocabulary. Project requires full circle, therefore the default value is None
    :param use_idf: Parameter, that sets should IDF be used in calculations or not. For more complete analysis,
    is set True by default
    :param smooth_idf: If True, adds 1 into TF-IDF ratio to avoid high values for very rare words and
    avoids zero division errors. Set True by default
    :param dtype: Data type for matrix. Set float32 to decrease memory utilization

    :returns X: Sparse document-term matrix
    :returns vectorizer: The fitted vectorizer
    :returns feature_names: Vocabulary terms in the same order as columns of X
    """

    #Logging key parameters
    t0 = time.time()
    logger.info(
        "TF-IDF vectorization started: max_features=%s, ngram_range=%s, min_df=%s, sublinear_tf=%s, norm=%s",
        max_features, ngram_range, min_df, sublinear_tf, norm
    )

    # Initializes the vectorizer object
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=lowercase,
        strip_accents=strip_accents,
        vocabulary=vocabulary,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        norm=norm,
        dtype=dtype,
        )

    # Starts the learning process
    if vocabulary is None:
        logger.info("Fitting TF-IDF vectorizer and transforming texts...")
        X = vectorizer.fit_transform(texts)
        logger.info("Fit+transform complete.")
    else:
        logger.info("Using existing vocabulary for TF-IDF transform only...")
        X = vectorizer.transform(texts)
        logger.info("Transform complete.")

    # Extracts the name of features
    feature_names = list(vectorizer.get_feature_names_out())
    elapsed = time.time() - t0
    logger.info(
        "TF-IDF finished: shape=%s, features=%s, time=%.2fs",
        X.shape, f"{len(feature_names):,}", elapsed
    )

    return X.tocsr(), vectorizer, feature_names


def matrix_stats(X:csr_matrix, name:str) -> str:
    '''
    Prepares statistical values for matrix obtained in vectorization

    :param X: Matrix from vectorization
    :param name: Name of the method
    :return: string with the information about matrix
    '''

    nnz = X.nnz
    total = X.shape[0] * X.shape[1]
    density = nnz / total
    mean_val = X.data.mean() if X.nnz > 0 else 0
    return f"{name}: shape={X.shape}, nnz={nnz}, density={density:.6f}, mean_weight={mean_val:.6f}"


#Normalization functionality
def normalize_dataset(
        logger: logging.Logger,
        csv_path: str,
        output_csv: str,
        narrative_col: str = "narrative",
        spacy_model: str = "en_core_web_sm",
        extra_stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
) -> pd.DataFrame:
    """
    Load a CSV and normalize reviews contained in `narrative_col`. Normalization is performed with:
    - Setting to lowercase all texts
    - Expand English contractions
    - Removing URLs/Email/Hashtags/mentions
    - Lemmatize
    - Removing stop-words
    - Removing small tokens

    :param logger: Logger object for logging
    :param csv_path: Path to the input CSV file
    :param narrative_col: Name of the column containing the free-text reviews. Default "narrative",
    since in used dataset it is called so
    :param output_csv: Saves cleaned dataframe to this path
    :param spacy_model: spaCy English model to use for tokenization/lemmatization. Default "en_core_web_sm", since in
    dataset English language is the primary one
    :param extra_stopwords: Extra stopwords to remove (case-insensitive, will be lowercased)
    :param min_token_len: Discard tokens shorter than this length after lemmatization.

    :returns df: DataFrame with original columns plus a `narrative_clean` column.
    """

    logger.info('Starting normalization function')

    #Checks if the path with raw text exists
    if not os.path.exists(csv_path):
        logger.error('Raw data file is not set')
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    #Loads the raw text file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError or Exception as e:
        logger.error('Raw data os not loaded!')
        raise FileNotFoundError()

    #Checks if the column with the used name is present
    if narrative_col not in df.columns:
        logger.error(f"Column '{narrative_col}' not found. Available: {list(df.columns)}")
        raise KeyError()
    else:
        logger.info(f'Column {narrative_col} is found')

    #First normalization: sets all values to str, replaces IRRELEVANT with NaN, removes empty strings and duplicates
    IRRELEVANT = {
        "", " ", "na", "n/a", "none", "null", "nan", "no comment", "-", "—", "*", ".", "— —", "n.a."
    }
    df[narrative_col] = (
        df[narrative_col]
        .astype(str)
        .str.strip()
        .replace({v: np.nan for v in IRRELEVANT}, regex=False)
    )
    logger.info('All values in narrative column are set to str values')
    logger.info('All leading and trailing spaces are removed')
    logger.info('All irrelevant values are replaced with NaN')
    df = df.dropna(subset=[narrative_col])
    logger.info('All empty strings are removed')
    df = df.drop_duplicates(subset=[narrative_col]).reset_index(drop=True)
    logger.info('All duplicates are removed')

    #Prepares patterns for irrelevant values removal
    url_re = re.compile(r"""https?://\S+|www\.\S+""", flags=re.IGNORECASE)
    email_re = re.compile(r"""\b[\w\.-]+@[\w\.-]+\.\w+\b""", flags=re.IGNORECASE)
    handle_re = re.compile(r"""(?:^|\s)@[\w_]+""")
    hashtag_re = re.compile(r"""(?:^|\s)#[\w_]+""")

    # Keep letters and spaces only (after we expand contractions etc.)
    non_letter_re = re.compile(r"[^a-z\s]+")

    #Loads spaCy model
    try:
        nlp = spacy.load(spacy_model, disable=["ner", "textcat"])
        logger.info("spaCy model is loaded successfully")
    except OSError as e:
        logger.error(f"spaCy model '{spacy_model}' not found. Install it via: python -m spacy download {spacy_model}")
        raise OSError() from e

    #Forms the set of stop-words
    stopwords = {w.lower() for w in nlp.Defaults.stop_words}
    logger.info('Stopwords set is built')
    if extra_stopwords:
        stopwords |= {w.lower() for w in extra_stopwords}

    #Subfunction for text clearing before lemmatize
    def _normalize_text(text: str) -> str:
        # lowercase first (helps contractions & matching)
        t = text.lower()

        # expand contractions (e.g., isn't -> is not)
        t = contractions.fix(t)

        # remove urls/emails/handles/hashtags
        t = url_re.sub(" ", t)
        t = email_re.sub(" ", t)
        t = handle_re.sub(" ", t)
        t = hashtag_re.sub(" ", t)

        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()

        # remove digits and symbols, keep letters and space only
        t = non_letter_re.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    #Results of cleared texts
    df["_normalized"] = df[narrative_col].map(_normalize_text)

    # Tokenize + Lemmatize with spaCy (streaming with pipe for speed)
    cleaned_texts = []
    for doc in nlp.pipe(df["_normalized"].tolist(), batch_size=512, n_process=1):
        lemmas = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.lower().strip()
            # Filter stopwords and very short tokens
            if lemma and lemma not in stopwords and len(lemma) >= min_token_len:
                # spaCy may return "-PRON-" in some models; skip weird lemmas
                if lemma.isalpha():
                    lemmas.append(lemma)
        cleaned_texts.append(" ".join(lemmas))
    df["narrative_clean"] = cleaned_texts
    logger.info('Texts are tokenized')

    #Removes empty strings after normalization
    df = df[df["narrative_clean"].str.len() > 0].reset_index(drop=True)
    logger.info('All empty lines are removed')

    #Saves results to CSV
    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            logger.info('Output file is created')
        df.to_csv(output_csv, index=False)
        logger.info(f'Data is exported to {output_csv}')

    # Cleanup temp column
    if "_normalized" in df.columns:
        df = df.drop(columns=["_normalized"])

    return df

#Topic extraction functionality
def topic_extraction_with_lda(
    X: csr_matrix,
    vectorizer: CountVectorizer,
    logger: logging.Logger,
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
    logger: logging.Logger,
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





#Script
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
