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
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
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

### 💻 Directly on Host System

1. **Install repository with git**
    ```bash
   git clone https://github.com/llevvt/DLBDSEDA02_Lev_Mordvinkov.git  # On Linux/Mac/Windows
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   # OR
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the script**
   ```bash
   python app.py
   ```

5. **Results**
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
