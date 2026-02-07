# 🎯 Amazon Reviews Sentiment Analysis  
## A Complete Machine Learning Project from Data to Deployment

> A **production-ready sentiment classification system** analyzing 300K+ Amazon Electronics reviews, achieving **88.03% accuracy** with advanced NLP and machine learning techniques. Built from scratch demonstrating full ML lifecycle expertise.

> [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amazon-review-sentiment-ml-karangautam870.streamlit.app/])

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)](https://pandas.pydata.org)

---

## 🚀 Project Highlights

This project demonstrates **end-to-end machine learning engineering** - from raw data to deployed web application.

| Achievement | Metric | Impact |
|-------------|--------|--------|
| **Test Accuracy** | 88.03% ⭐ | Nearly 9 out of 10 reviews classified correctly |
| **Weighted AUC** | 0.9592 ✨ | Excellent discrimination across all sentiments |
| **Macro AUC** | 0.9458 | Strong performance treating all classes equally |
| **Train-Test Gap** | 0.90% | Excellent generalization, minimal overfitting |
| **Data Extraction** | Reservoir Sampling | Extracted 300K from 10GB JSON file efficiently |
| **Dataset Size** | 300,000 reviews | Real-world scale processing |
| **Models Compared** | 3 algorithms | LR, MultinomialNB, BernoulliNB evaluated |
| **Features** | 4,500 TF-IDF + bigrams | Advanced NLP preprocessing |
| **Deployment** | Production-ready | Serialized + Streamlit web app |

---

## ⚡ Quick Start (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web app
streamlit run sentiment_detection.py
```

**Try it live:**
```
Review: "This product is amazing! Best purchase ever"
→ 😊 Positive (High confidence)

Review: "Terrible quality, broke after 1 day"
→ 😞 Negative (High confidence)
```

---

## 📊 Model Performance Results

### Test Set Metrics (Best: Logistic Regression)
```
Accuracy:  88.03% 🏆
Precision: 86.79% (weighted)
Recall:    88.03% (weighted)
F1-Score:  87.05% (weighted)

Discrimination Performance:
├─ Macro AUC:    0.9458 (one-vs-all)
└─ Weighted AUC: 0.9592 (proportional to class size)
```

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| **Logistic Regression** ⭐ | **88.03%** | **86.79%** | **88.03%** | **87.05%** | **✅ WINNER** |
| Multinomial NB | 85.10% | 83.94% | 85.10% | 82.47% | Good baseline |
| Bernoulli NB | 71.98% | 77.60% | 71.98% | 74.30% | Lower performance |

**Selection Justification:**
✅ Highest test accuracy (88.03%)  
✅ Excellent generalization (0.90% train-test gap)  
✅ Interpretable coefficients  
✅ Faster inference  

---

## 🏗️ Project Structure

```
amazon/ (Git-Tracked)
│
├── 📓 Analysis Notebooks (3 files)
│   ├── data_prepare.ipynb                      # Preprocessing pipeline
│   ├── EDA.ipynb                               # Initial exploration
│   └── EDA_amazon_dataset_electronics.ipynb    # 82-cell deep analysis
│       ├─ Distribution visualizations
│       ├─ Statistical tests (Chi-square, ANOVA, t-tests)
│       ├─ N-gram analysis
│       ├─ POS tagging
│       ├─ Outlier detection
│       └─ Feature engineering
│
├── 🤖 ML Pipeline (3 files)
│   ├── model.ipynb                             # 45 executed cells
│   │   ├─ Data preprocessing
│   │   ├─ Stratified train-test split (80-20)
│   │   ├─ TF-IDF vectorization (4500 features)
│   │   ├─ 3-model training & tuning
│   │   ├─ Comprehensive evaluation
│   │   ├─ Confusion matrices
│   │   ├─ ROC-AUC analysis
│   │   └─ 6 preprocessing validation tests
│   │
│   ├── pipeline_class.py                       # (113 lines)
│   │   ├─ Negation handling (15 types)
│   │   ├─ Automatic preprocessing
│   │   ├─ Error validation
│   │   └─ Batch processing
│   │
│   └── sentiment_pipeline.pkl                  # Serialized model (272KB)
│
├── 🌐 Deployment (2 files)
│   ├── sentiment_detection.py                  # Streamlit UI
│   └── requirements.txt                        # Dependencies
│
└── ⚙️ Config
    └── .gitignore                              # Excludes large files
```

---

## 🔬 Technical Implementation

### Data Extraction & Sampling (10GB → 300K Reviews)

**Challenge**: Original dataset is 10GB+ JSON file - too large for direct processing in memory

**Solution**: **Reservoir Sampling** Algorithm
```python
# Efficient streaming algorithm
- Read JSON file sequentially (no full load needed)
- Use random replacement for memory-efficient sampling
- Probability: k/n (k=sample size, n=total items)
- Maintains uniform distribution across entire dataset
- Memory usage: O(k) instead of O(n)
```

**Implementation:**
✅ Stream process 10GB JSON file line-by-line  
✅ Extract 300,000 Electronics reviews (~3% of data)  
✅ Preserve class distribution (balanced sampling)  
✅ No data loss, representative subset  
✅ Reduced memory footprint (from 10GB → ~200MB)  

**Result**: 300,000 reviews with uniform distribution maintained across all date ranges

### Data Preprocessing
- **Dataset**: 300,000 Amazon Electronics reviews (sampled via reservoir sampling)
- **Split**: 80-20 train-test (stratified)
- **Normalization**:
  - Contraction expansion (don't → do not)
  - Negation marking (not_good)
  - Lowercase, tokenization, stopword removal

### Feature Engineering
```
TF-IDF Vectorizer (4,500 features):
├─ max_features: 4500
├─ ngram_range: (1, 2)    # Words + pairs
├─ max_df: 0.8            # Remove common
├─ min_df: 2              # Remove rare
└─ sublinear_tf: True     # Log-scaling
```

### Advanced NLP
- ✅ Custom negation handling (15 contractions)
- ✅ Statistical validation (Chi-square, ANOVA, t-tests)
- ✅ N-gram analysis
- ✅ POS tagging
- ✅ Outlier detection (IQR)

---

## 💻 Code Quality

### OOP Design
```python
class SentimentPipeline:
    """Production-grade sentiment pipeline"""
    - Automatic preprocessing
    - Component validation
    - Error handling
    - Batch support
```

### Error Handling
- Validates fitted components
- Graceful None/empty handling
- Type checking
- Clear error messages

### Model Serialization
- Trained model: sentiment_pipeline.pkl (272KB)
- Includes: vectorizer + model + encoder
- Ready to deploy immediately
- Reproducible predictions

---

## 🎯 Skills Demonstrated

### Machine Learning
✅ Classification algorithms  
✅ Model selection & comparison  
✅ Hyperparameter tuning  
✅ Evaluation metrics (accuracy, precision, recall, F1, AUC)  
✅ Overfitting detection  
✅ Train-test splits  

### Natural Language Processing
✅ Text preprocessing  
✅ TF-IDF vectorization  
✅ Negation handling  
✅ N-gram analysis  
✅ Feature engineering  

### Data Science
✅ Exploratory analysis (EDA)  
✅ Statistical testing  
✅ Data visualization  
✅ Feature importance  
✅ Outlier detection  

### Software Engineering
✅ Object-oriented programming  
✅ Code modularity  
✅ Error handling  
✅ Model persistence  
✅ Web deployment (Streamlit)  

### Big Data & Algorithms
✅ Reservoir Sampling (10GB → 300K efficient extraction)  
✅ Memory-efficient streaming  
✅ Statistical sampling techniques  
✅ Data size reduction without bias  

### Technologies
✅ Python (Pandas, NumPy, Scikit-learn, NLTK)  
✅ Jupyter Notebooks  
✅ Git version control  
✅ Streamlit  

---

## 📈 Key Insights

1. **Class Imbalance**: 74% positive, 13.2% negative, 12.8% neutral
2. **Text Length**: Negative reviews more detailed (higher word count)
3. **Negation Importance**: Critical for understanding context
4. **Feature Discrimination**: ~300 features drive most predictions

---

## 🌟 Why This Stands Out

### For Recruiters
✅ Complete ML lifecycle (end-to-end)  
✅ Production code (not just notebooks)  
✅ Rigorous evaluation (3 models, multiple metrics)  
✅ Advanced NLP techniques  
✅ Clean, documented code  
✅ Real-world scale (300K reviews)  
✅ Transparent metrics  

### For Interviews
- Why Logistic Regression over Naive Bayes?
- How does negation handling work?
- How to handle class imbalance?
- What evaluation metrics matter?
- How to deploy in production?

### For Learning
- See complete ML workflow
- Study advanced NLP
- Review best practices
- Learn production patterns

---

## 📝 How to Use

### To Learn:
1. data_prepare.ipynb → preprocessing
2. EDA files → analysis techniques
3. model.ipynb → ML pipeline
4. pipeline_class.py → production code

### To Deploy:
```python
from pipeline_class import SentimentPipeline
import joblib

pipeline = joblib.load('sentiment_pipeline.pkl')
prediction = pipeline.predict("Great product  !")
```

### To Extend:
- Try BERT/DistilBERT
- Implement cross-validation
- Add REST API
- Deploy to cloud

---

## ✨ Project Status

✅ **Complete & Production-Ready**

- ✅ 300,000+ reviews analyzed
- ✅ 88.03% test accuracy achieved
- ✅ 45 model cells executed
- ✅ 6 preprocessing tests pass
- ✅ Pipeline serialized
- ✅ Web app ready

---

<div align="center">

### 📊 Summary

```
Test Accuracy:     88.03%
Weighted AUC:      0.9592
Models Compared:   3
Dataset Size:      300,000
Features:          4,500
Notebooks:         3 files (82+ cells)
Code Files:        2 Python + 1 Pickle
Status:            ✅ Production-Ready
```

**Built as a comprehensive ML portfolio project**  
**Demonstrating real-world engineering skills**

---

**Author**: Karan Gautam  
**Date**: February 2026  
**Dataset**: Amazon Electronics Reviews  
**Status**: ✅ Production-Ready

</div>
