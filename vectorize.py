from typing import Iterable, Tuple
import time
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from logging import Logger


def vectorize_with_bow(
    logger: Logger,
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
        logger: Logger,
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