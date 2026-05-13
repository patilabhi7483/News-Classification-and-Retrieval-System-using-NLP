# Module 1 — Language Model

## Overview

This module implements the Language Modeling component of the Intelligent News Analysis Pipeline using Natural Language Processing (NLP) techniques. The module processes raw news article text, performs preprocessing, builds Unigram and Bigram Language Models, applies Laplace Smoothing, and evaluates model performance using Perplexity.

The implementation uses a BBC News dataset containing news articles related to multiple domains such as Sports, Politics, and Technology.

---

## Features

* Text preprocessing
* Tokenization
* Stopword removal
* Unigram Language Model
* Bigram Language Model
* Laplace (Add-One) Smoothing
* Vocabulary generation
* Perplexity calculation
* Train-test split

---

## Files

| File Name           | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `language_model.py` | Main implementation of preprocessing, language modelling, smoothing, and perplexity |
| `outputs/`          | Contains screenshots and output images generated during execution                   |

---

## Dataset Used

* Dataset: `bbc_news.csv`
* Format: CSV
* Content:

  * News titles
  * News descriptions
  * Multiple categories

---

## Libraries Used

* pandas
* nltk
* math
* random
* collections

---

## Installation

Install required libraries using:

```bash
pip install pandas nltk
```

---

## NLTK Downloads

Run the following before execution:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Run

Open terminal inside the project directory and run:

```bash
python language_model.py
```

---

## Outputs Generated

The program generates:

* Dataset information
* Preprocessed text
* Vocabulary size
* Unigram frequency samples
* Bigram frequency samples
* Perplexity without smoothing
* Perplexity with Laplace smoothing

---

## Sample Workflow

1. Load BBC News dataset
2. Combine title and description
3. Preprocess text
4. Tokenize news articles
5. Build Unigram and Bigram models
6. Apply Laplace smoothing
7. Compute perplexity scores

---

## Concepts Implemented

* Statistical Language Modelling
* Maximum Likelihood Estimation
* Add-One Smoothing
* Probability Estimation
* Text Preprocessing
* Tokenization

---

## Future Improvements

* Trigram Language Models
* Neural Language Models
* Transformer-based Language Models
* Real-time news processing
* Better smoothing techniques

---

## Author

NLP News Analysis Project
2025–2026
