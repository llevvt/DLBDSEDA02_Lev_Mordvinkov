import re
from typing import Optional, Iterable
import contractions
import spacy
import pandas as pd
import os
import numpy as np
from logging import Logger


def normalize_dataset(
        logger: Logger,
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
