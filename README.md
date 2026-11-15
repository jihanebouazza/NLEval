# NLP Evaluation Metrics — Classification, NER & Machine Translation

## Project Overview

This project explores the evaluation of three fundamental Natural Language Processing (NLP) tasks using real datasets and pretrained transformer models.

The main objective is to understand how different evaluation metrics assess model performance and why specific NLP tasks require different evaluation approaches.

The project covers three experiments:

1. **Text Classification:** Sentiment analysis using DistilBERT on IMDb movie reviews.
2. **Sequence Labeling:** Named Entity Recognition (NER) using BERT on CoNLL-2003.
3. **Machine Translation:** English-to-French translation using MarianMT on OPUS Books.

All experiments use real model predictions rather than simulated outputs.

The project also includes visualizations, error analysis, comparative evaluation, and a final discussion of the strengths and limitations of the metrics.

---

## 1. Repository Structure

```text
NLP-Evaluation-Metrics/
│
├── README.md
└── NLEval.ipynb
```

**README.md:** Contains the project documentation, experimental methodology, dependencies, evaluation results, and execution instructions.

**NLEval.ipynb:** Contains the complete implementation of all experiments, including dataset loading, model inference, metric computation, visualizations, interpretations, and comparative analysis.

All project components are included in a single Jupyter Notebook.

---

## 2. Technologies and Libraries

The project was developed using Python and Jupyter Notebook / Google Colab.

### Main Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- seqeval
- SacreBLEU
- NLTK
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- SentencePiece
- Accelerate

### Installation

Install the required libraries using:

```bash
pip install torch transformers datasets accelerate sentencepiece scikit-learn seqeval sacrebleu nltk numpy pandas matplotlib seaborn tqdm
```

When using Google Colab, the installation command can be executed directly in a notebook cell by adding `!` before `pip`.

The pretrained models and datasets are downloaded automatically during execution.

A GPU is recommended for faster inference, although the experiments can also run on CPU.

---

## 3. Experiment 1 — Sentiment Classification

### Objective

Evaluate a pretrained DistilBERT model on a binary sentiment classification task.

The model predicts whether a movie review expresses positive or negative sentiment.

### Dataset

**IMDb Movie Reviews Dataset**

| Property | Value |
|---|---|
| Dataset | IMDb |
| Original test dataset | 25,000 reviews |
| Evaluation sample | 500 reviews |
| Negative reviews | 254 |
| Positive reviews | 246 |
| Number of classes | 2 |

### Methodology

1. Load the IMDb test dataset.
2. Select 500 reviews using a fixed random seed.
3. Analyze class distribution and review lengths.
4. Load a pretrained DistilBERT sentiment classifier.
5. Generate predictions and class probabilities.
6. Compare predictions with ground-truth labels.
7. Calculate classification metrics.
8. Visualize the confusion matrix and ROC curve.
9. Analyze incorrectly classified reviews.

### Evaluation Metrics

**Accuracy:** Proportion of correctly classified reviews.

**Precision:** Proportion of predicted positive reviews that are actually positive.

**Recall:** Proportion of actual positive reviews correctly identified.

**F1-score:** Harmonic mean of Precision and Recall.

**ROC-AUC:** Measures the model's ability to distinguish between positive and negative reviews across classification thresholds.

### Experimental Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8900 |
| Precision | 0.9099 |
| Recall | 0.8618 |
| F1-score | 0.8852 |
| ROC-AUC | 0.9537 |

The model correctly classified 445 out of 500 reviews.

The confusion matrix identified 21 false positives and 34 false negatives.

### Key Findings

The model achieved an accuracy of 89%, demonstrating strong classification performance on the selected IMDb subset.

Precision was higher than Recall, indicating that positive predictions were generally reliable, although some positive reviews were incorrectly classified as negative.

ROC-AUC reached 95.37%, indicating strong discrimination between the two sentiment classes.

---

## 4. Experiment 2 — Named Entity Recognition

### Objective

Evaluate the ability of a pretrained BERT model to identify and classify named entities within English sentences.

### Dataset

**CoNLL-2003**

| Property | Value |
|---|---:|
| Training sentences | 14,041 |
| Validation sentences | 3,250 |
| Test sentences | 3,453 |
| Evaluation sample | 500 sentences |

The dataset contains four entity categories:

- **PER:** Person
- **ORG:** Organization
- **LOC:** Location
- **MISC:** Miscellaneous

The annotations follow the BIO tagging scheme.

### Model

`dslim/bert-base-NER`

### Methodology

1. Load the CoNLL-2003 dataset.
2. Select 500 test sentences.
3. Explore BIO tag distributions.
4. Load the pretrained BERT NER model.
5. Generate predictions for each sentence.
6. Align subword predictions with original words.
7. Evaluate individual BIO tags.
8. Evaluate complete named entities using strict IOB2 matching.
9. Analyze the confusion matrix and entity-specific errors.

### Evaluation Metrics

**Token-level evaluation:**

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

**Entity-level evaluation:**

- Precision
- Recall
- F1-score
- Entity-specific classification report

### Experimental Results

| Metric | Score |
|---|---:|
| Token Accuracy | 0.9791 |
| Token Macro Precision | 0.8641 |
| Token Macro Recall | 0.9049 |
| Token Macro F1-score | 0.8830 |
| Entity Precision | 0.9010 |
| Entity Recall | 0.9236 |
| Entity F1-score | 0.9122 |

### Performance by Entity Category

| Entity Type | F1-score |
|---|---:|
| PER | 0.9648 |
| LOC | 0.9336 |
| ORG | 0.8906 |
| MISC | 0.7864 |

### Key Findings

The BERT model achieved a token-level accuracy of 97.91% and an entity-level F1-score of 91.22%.

The highest performance was observed for person entities, while miscellaneous entities were more challenging.

The experiment also demonstrated that token-level accuracy can be misleading when non-entity tokens dominate the dataset.

Strict entity-level evaluation provides a more meaningful assessment of complete entity extraction.

---

## 5. Experiment 3 — Machine Translation

### Objective

Evaluate the quality of English-to-French translations generated by a pretrained MarianMT model.

### Dataset

**OPUS Books — English/French**

| Property | Value |
|---|---|
| Dataset | OPUS Books |
| Original dataset size | 127,085 sentence pairs |
| Evaluation sample | 200 sentence pairs |
| Source language | English |
| Target language | French |

### Model

`Helsinki-NLP/opus-mt-en-fr`

### Methodology

1. Load the English–French parallel dataset.
2. Select 200 reproducible sentence pairs.
3. Analyze source and reference sentence lengths.
4. Load the pretrained MarianMT model.
5. Generate French translations using beam search.
6. Compare translations against human references.
7. Compute corpus-level evaluation metrics.
8. Calculate sentence-level metrics.
9. Visualize score distributions.
10. Analyze sentence length, metric correlations, and translation errors.

### Evaluation Metrics

**BLEU:** Measures word n-gram overlap between generated and reference translations.

**METEOR:** Evaluates word alignment using Precision, Recall, and fragmentation penalties.

**chrF:** Measures character n-gram similarity.

**TER:** Estimates the number of edits needed to transform a generated translation into a reference translation.

### Experimental Results

| Metric | Score |
|---|---:|
| BLEU | 20.3004 |
| METEOR | 0.3788 |
| chrF | 48.2011 |
| TER | 73.2036 |

BLEU, chrF, and TER use their conventional SacreBLEU scales, while METEOR is reported on a 0–1 scale.

### Key Findings

The model generated French translations with varying degrees of similarity to the human references.

The corpus-level BLEU score was 20.30, while chrF reached 48.20.

Sentence-level analysis showed considerable variation in translation scores.

The experiment also demonstrated that valid translations may receive relatively low automatic scores when their wording differs from the reference.

Some examples suggested possible language-direction or alignment inconsistencies in the OPUS Books dataset.

These observations highlight the importance of evaluating both model outputs and reference data quality.

---

## 6. Comparative Analysis

The three experiments demonstrate that different NLP tasks require different evaluation strategies.

| NLP Task | Model | Dataset | Main Evaluation Metric |
|---|---|---|---|
| Sentiment Classification | DistilBERT | IMDb | Accuracy / F1-score |
| Named Entity Recognition | BERT | CoNLL-2003 | Entity-level F1-score |
| Machine Translation | MarianMT | OPUS Books | BLEU / chrF |

### Main Observations

**Classification:** Evaluation focuses on predicting the correct category for each input.

**Named Entity Recognition:** Evaluation must consider both individual token labels and complete entity boundaries.

**Machine Translation:** Evaluation measures similarity between generated translations and human references.

The numerical scores cannot be directly compared across tasks because they represent different evaluation objectives and scales.

### Importance of Error Analysis

Aggregate metrics summarize model performance but do not fully explain prediction errors.

The experiments therefore include:

- Confusion matrix analysis for sentiment classification.
- Token-level and entity-level error analysis for NER.
- Sentence-level evaluation and qualitative translation analysis for machine translation.

These approaches provide additional insight into model strengths and limitations.

---

## 7. How to Run the Project

### Option 1 — Google Colab

1. Download `NLEval.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload the notebook.
4. Select a GPU runtime if available.
5. Execute the cells sequentially.

### Option 2 — Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/NLP-Evaluation-Metrics.git
cd NLP-Evaluation-Metrics
```

Install the required libraries using the command provided in Section 2.

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `NLEval.ipynb` and execute the cells sequentially.

### Reproducibility

The experiments use a fixed random seed of 42 for dataset sampling.

All three models are pretrained and are evaluated without additional fine-tuning during the experiments.

Execution time depends on the available hardware.

---

## 8. Experimental Limitations

Several limitations should be considered:

1. The experiments use subsets of the original datasets.
2. Only one pretrained model is evaluated for each task.
3. Token-level NER accuracy is influenced by class imbalance.
4. Machine translation metrics depend on reference translation quality.
5. Some OPUS Books sentence pairs appear to contain language-direction or alignment inconsistencies.
6. Potential overlap between OPUS Books and the translation model's training data cannot be ruled out.
7. Automatic evaluation metrics do not fully capture semantic correctness or linguistic fluency.
8. Results may differ across domains, languages, and evaluation datasets.

---

## 9. Conclusion

This project provides a practical comparison of evaluation methodologies for three major Natural Language Processing tasks.

Using real pretrained transformer models and real datasets, the experiments demonstrate how evaluation metrics reveal different aspects of model performance.

The sentiment classification experiment highlights the importance of Precision, Recall, F1-score, and ROC-AUC.

The Named Entity Recognition experiment demonstrates why entity-level evaluation is essential for sequence labeling.

The machine translation experiment illustrates the strengths and limitations of reference-based evaluation metrics such as BLEU, METEOR, chrF, and TER.

The overall conclusion is that **no single metric is universally suitable for evaluating every NLP task**.

Reliable NLP evaluation requires task-appropriate metrics, representative datasets, quantitative analysis, and qualitative examination of model errors.
