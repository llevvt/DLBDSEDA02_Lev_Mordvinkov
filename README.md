# NLP Topic Modeling Pipeline

## 🧭 Overview

This project implements a complete **text processing and topic extraction pipeline** using Natural Language Processing (NLP) techniques.  
It was developed as part of the *DLBDSEDA02 – Data Literacy and Big Data Security Engineering* assignment.

The pipeline performs:
- **Text normalization** (lowercasing, tokenization, lemmatization, stop-word removal)
- **Vectorization** using Bag-of-Words (BoW) and TF–IDF models
- **Topic extraction** via **Latent Semantic Analysis (LSA)** and **Latent Dirichlet Allocation (LDA)**
- **Saving and reporting** results for further analysis

All major parameters (e.g., vocabulary size, n-gram range, topic count, iterations) are configured through the `.env` file.

---

## 🧩 Project Structure

```
.
├── app.py                # Main orchestrator script
├── normalize.py             # Text normalization logic
├── vectorize.py             # TF–IDF and BoW vectorization
├── extract_topics.py        # LSA and LDA topic extraction
├── helpers.py               # Utility functions (saving, logging, etc.)
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
├── Dockerfile               # Docker build instructions
├── docker-compose.yml       # Container orchestration file
└── README.md                # Documentation
```

Additionally, you can find in repository a source file, which was used for analysis, 
as well as files with results. However, these files are not necessary for the code execution,
you can try to use your own dataset. Make sure, that you have adjustet .env and code to work with custom dataset.

---

## ⚙️ Configuration

All pipeline parameters are stored in `.env`.  
Example:

```env
RAW_DATASET='./complaints_processed.csv'
NORMALIZED_TEXTS='./normalized_texts.csv'
OUTPUT_DIR=results

# Normalization
MIN_TOKEN_LEN='2'

# Vectorization
VOCABULARY_SIZE='20000'
MIN_N_GRAM='1'
MAX_N_GRAM='2'
MIN_FREQUENCY='2'

# Topic Extraction
NUMBER_OF_TOPICS='10'
NUMBER_OF_TOP_WORDS='15'
NUMBER_OF_ITERATIONS='20'
RANDOM_STATE='42'
```

---

## 🚀 How to Run

### 🐳 Option 1 — Run in Docker Container (Recommended)

This option guarantees reproducibility and isolates dependencies.

1. **Build the Docker image**
   ```bash
   docker compose build
   ```

2. **Run the pipeline**
   ```bash
   docker compose up --abort-on-container-exit
   ```

3. **Results**
   - The pipeline automatically creates a directory defined by `OUTPUT_DIR` (default: `results`).
   - All generated CSV, JSON, and topic reports are stored there and visible on the host.

Example output structure:
```
./results/
├── normalized_texts.csv
├── bow_vectorization.json
├── tfidf_vectorization.json
├── lda_topics.json
└── lsa_topics.json
```

---

### 💻 Option 2 — Run Directly on Host System

If you prefer to execute the pipeline locally (outside Docker):

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   # OR
   venv\Scripts\activate      # On Windows
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the script**
   ```bash
   python app.py
   ```

4. **Results**
   - Output files will be saved to the folder defined in `.env` (default: `./results`).

---

## 🧠 Notes

- The project supports **customizable NLP preprocessing** using the spaCy English model (`en_core_web_sm` by default).  
  You can change it by editing `.env` or providing a different model name at runtime.
- Ensure that `RAW_DATASET` points to a valid CSV file containing the text column (default: `narrative`).
- Logging information is printed to console and optionally stored in the results directory.

---

## 📚 Technologies Used

- **Python 3.11**
- **pandas**, **numpy**, **scikit-learn**
- **spaCy** for lemmatization
- **Docker** + **docker-compose** for reproducible environments

---

## 🏁 Purpose

The primary goals of this project are to:
- Demonstrate a **complete NLP workflow** for text analytics;
- Compare **topic modeling methods** (LSA vs LDA);
- Provide **clean, modular, production-ready Python code**, which can be further used to 
- Train custom models and reuse them in different contexts.

---

**Author:** Lev Mordvinkov

**Institution:** IU International University of Applied Sciences
