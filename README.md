# When Does Data Augmentation Help? A Comparative Study on Short-Text Sentiment Classification

A controlled experiment testing whether classic text data augmentation techniques (EDA) actually improve performance in a low-resource setting — with an honest negative result.

## Overview

Data augmentation is often assumed to help when labeled data is scarce. This project tests that assumption directly: using only 300 labeled tweets per class (1,200 total, approximately 1.7% of the available training data), I compared five augmentation strategies against a no-augmentation baseline and a full-data reference, using a fixed classifier so the comparison isolates the effect of the training data itself.

**Key finding:** none of the five augmentation techniques outperformed the no-augmentation baseline. On this short-text dataset, augmentation slightly *hurt* performance rather than helping — a result that is analyzed and explained in the notebook.

## Research Question

When training data is limited, how much does each text augmentation technique actually help — and are some techniques better than others, or even harmful?

## Dataset

**Twitter Entity Sentiment Analysis** — approximately 74K tweets labeled across four classes:

- Positive
- Negative
- Neutral
- Irrelevant

A separate 1,000-tweet validation set was used as the held-out evaluation set throughout the experiment.

## Techniques Compared

The augmentation techniques were implemented from scratch following **EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks** by Wei & Zou (EMNLP-IJCNLP 2019).

| Technique | Description |
|---|---|
| **Synonym Replacement (SR)** | Replace random non-stopwords with a WordNet synonym |
| **Random Insertion (RI)** | Insert a synonym of a random word at a random position |
| **Random Swap (RS)** | Swap the positions of two random words |
| **Random Deletion (RD)** | Randomly delete words with probability *p* |
| **Combined** | Apply all four augmentation operations in sequence |

## Methodology

- Simulated a low-resource scenario by sampling **300 tweets per class** from the training data.
- This produced **1,200 sampled tweets**, with **1,191 remaining after cleaning**.
- Generated **2 augmented variants per original tweet** for each technique, producing 3× the baseline training size.
- Used a **fixed TF-IDF + Linear SVM classifier** across every dataset so that differences in performance primarily reflect changes in the training data.
- Evaluated every version on the same untouched **1,000-tweet validation set**.
- Used the full cleaned training set as a **reference point** for comparing the performance gap between low-resource and full-data training.
- Performed an augmentation-strength (`alpha`) sweep on **Combined**, the best-performing augmentation method in the main comparison.

## Results

| Dataset | Train Size | Macro F1 |
|---|---:|---:|
| Baseline (no augmentation) | 1,191 | **0.5069** |
| Combined (all 4) | 3,573 | 0.5055 |
| Random Insertion | 3,573 | 0.5005 |
| Random Deletion | 3,573 | 0.4923 |
| Synonym Replacement | 3,573 | 0.4848 |
| Random Swap | 3,573 | 0.4790 |
| Full data (reference / upper bound) | 71,315 | **0.8140** |

The full-data reference achieved a Macro F1 of **0.8140**, compared with **0.5069** for the low-resource baseline, giving a performance gap of approximately **0.3071 Macro F1**.

None of the five augmentation techniques closed this gap. Instead, every augmented dataset performed slightly worse than the baseline. Relative to the full-data gap, the changes ranged from **-0.5% for Combined** to **-9.1% for Random Swap**.

### Augmentation Strength Analysis

A follow-up sweep tested the **Combined** augmentation technique at:

`alpha = [0.05, 0.10, 0.20, 0.30]`

The resulting Macro F1 scores were approximately:

`[0.488, 0.501, 0.492, 0.491]`

The results showed a **non-monotonic pattern** with no consistent improvement as augmentation strength increased. The highest score occurred at `alpha = 0.10` (0.501), but it still remained below the no-augmentation baseline of 0.5069.

This suggests that simply tuning the augmentation strength was not enough to overcome the limitations of word-level augmentation for this dataset.

### Why Did Augmentation Hurt?

Tweets are short and information-dense, so modifying even a small number of words can meaningfully affect the sentiment information available to the classifier. For example, deleting or changing an important sentiment-bearing word can alter the meaning of a tweet.

The effect may be particularly noticeable for a **TF-IDF + Linear SVM** classifier because its representation relies directly on word and phrase features. Synthetic variants that introduce noisy or distorted wording can therefore change the features used for classification.

With `num_aug = 2`, each original tweet produces two augmented variants, increasing the training set from **1,191 to 3,573 examples**. This means that two-thirds of the training examples are synthetic variants rather than original tweets, potentially introducing enough noise to outweigh the benefit of having more training examples.

## Project Structure

```text
├── comparing-text-data-augmentation-techniques.ipynb
├── twitter_training.csv
├── twitter_validation.csv
└── README.md

## How to Run

1. Clone the repository and open the notebook in Jupyter Notebook, JupyterLab, or Kaggle.
2. Make sure `twitter_training.csv` and `twitter_validation.csv` are in the same directory as the notebook, or update the file paths accordingly.
3. Install the required dependencies:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn

## Limitations

- Word-level augmentation does not guarantee semantic preservation. A synonym replacement, insertion, swap, or deletion can change the meaning or sentiment of a short tweet.
- The results are specific to this dataset, the **TF-IDF + Linear SVM** classifier, the tested augmentation configurations, and the low-resource budget of **300 tweets per class**.
- The experiment only evaluated simple, CPU-based word-level augmentation techniques. More advanced approaches, such as **back-translation** and **contextual or transformer-based substitution**, may produce different results.

## Future Work

- Add **back-translation** and compare it with the word-level augmentation techniques.
- Try **contextual augmentation** using **BERT-based or other transformer-based word substitution**.
- Repeat the experiment with a **fine-tuned DistilBERT classifier** instead of TF-IDF + Linear SVM.
- Vary the low-resource budget, for example **50, 100, 300, and 1,000 tweets per class**, to investigate whether augmentation becomes useful at different training sizes.
- Test the augmentation techniques on **longer-form text**, such as product or movie reviews, to investigate whether the short-text nature of tweets contributes to the observed negative result.

## Reference

Wei, J., & Zou, K. (2019). *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks*. EMNLP-IJCNLP 2019.
