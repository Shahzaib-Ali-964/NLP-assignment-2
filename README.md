# NLP Assignment 2: Neural NLP Pipeline

**Student ID:** i23-XXXX  
**Section:** DS-X  
**Course:** CS-4063 Natural Language Processing  
**Institution:** FAST NUCES

---

## Project Overview

This project implements a complete neural NLP pipeline with three main components:

1. **Part 1: Word Embeddings** ✅ COMPLETE - TF-IDF, PPMI, and Skip-gram Word2Vec from scratch
2. **Part 2: Sequence Labeling** ✅ ARCHITECTURE COMPLETE - BiLSTM for POS tagging and NER with CRF
3. **Part 3: Topic Classification** ✅ ARCHITECTURE COMPLETE - Transformer encoder from scratch

**Current Status:** All architectures implemented. Training and evaluation code ready to run.

---

## Project Structure

```
i23-XXXX_Assignment2_DS-X/
├── i23-XXXX_Assignment2_DS-X.ipynb    # Main notebook with all implementations
├── README.md                           # This file
├── cleaned.txt                         # Preprocessed Urdu corpus
├── report.pdf                          # 2-3 page report (Times New Roman 12pt)
│
├── embeddings/                         # Word embedding outputs
│   ├── tfidf_matrix.npy               # TF-IDF weighted matrix
│   ├── ppmi_matrix.npy                # PPMI weighted co-occurrence matrix
│   ├── embeddings_w2v.npy             # Word2Vec embeddings (d=100)
│   ├── embeddings_w2v_d200.npy        # Word2Vec embeddings (d=200)
│   └── word2idx.json                  # Vocabulary mapping
│
├── models/                             # Trained model checkpoints
│   ├── skipgram_w2v.pt                # Skip-gram Word2Vec model
│   ├── bilstm_pos.pt                  # BiLSTM POS tagger
│   ├── bilstm_ner.pt                  # BiLSTM NER model
│   └── transformer_cls.pt             # Transformer classifier
│
└── data/                               # Annotated datasets
    ├── pos_train.conll                # POS training data
    ├── pos_test.conll                 # POS test data
    ├── ner_train.conll                # NER training data
    └── ner_test.conll                 # NER test data
```

---

## Requirements

### Python Version
- Python 3.8+

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision torchaudio
pip install tqdm jupyter notebook
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/i23-XXXX-NLP-Assignment2.git
cd i23-XXXX-NLP-Assignment2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Data Files

Ensure `cleaned.txt` is in the root directory.

### 4. Run Notebook

```bash
jupyter notebook i23-XXXX_Assignment2_DS-X.ipynb
```

Or use VS Code with Jupyter extension.

---

## Reproduction Instructions

### Part 1: Word Embeddings

1. **Open the notebook** and run cells sequentially from top to bottom
2. **TF-IDF** (Section 4):
   - Builds term-document matrix from cleaned.txt
   - Computes TF-IDF weights
   - Outputs: `embeddings/tfidf_matrix.npy`
   
3. **PPMI** (Section 5):
   - Builds co-occurrence matrix (window size k=5)
   - Applies Positive PMI weighting
   - Generates t-SNE visualization
   - Outputs: `embeddings/ppmi_matrix.npy`
   
4. **Skip-gram Word2Vec** (Section 6):
   - Trains Word2Vec with negative sampling
   - Hyperparameters: d=100, k=5, K=10, η=0.001
   - Training: 5 epochs, batch size 512
   - Outputs: `embeddings/embeddings_w2v.npy`
   
5. **Evaluation** (Section 7):
   - Nearest neighbors for 8 query words
   - Analogy tests (10 tests)
   - Four-condition comparison with MRR

**Expected Runtime:** ~20-30 minutes for Part 1 (CPU)

### Part 2: Sequence Labeling ✅ 

1. **Dataset Preparation** - 500 sentences with POS & NER annotation
   - Rule-based POS tagger with 12 tags
   - Gazetteer-based NER tagger with BIO scheme
   - 70/15/15 train/val/test split
   
2. **BiLSTM Architecture**
   - 2-layer bidirectional LSTM
   - Frozen and fine-tuned embedding modes
   - Dropout p=0.5
   - CRF layer for NER with Viterbi decoding
   
3. **Ready for Training**
   - Models initialized with Word2Vec embeddings
   - Early stopping with patience=5
   - Ablation study framework ready

**To Complete:** Run training loops, evaluate, generate reports

### Part 3: Transformer Encoder ✅

1. **All Components Implemented**
   - Scaled dot-product attention with masking
   - Multi-head self-attention (h=4, d_k=32)
   - Position-wise FFN (d_model=128, d_ff=512)
   - Sinusoidal positional encoding (non-learned)
   - 4 stacked encoder blocks with Pre-LN
   - CLS token classification head (128→64→5)
   
2. **Training Configuration Ready**
   - AdamW optimizer (lr=5e-4, weight decay=0.01)
   - Cosine learning rate schedule with 50 warmup steps
   - 20 epochs planned
   
3. **Evaluation Framework Ready**
   - Attention heatmap generation
   - Confusion matrix
   - BiLSTM comparison

**To Complete:** Prepare topic dataset, train model, evaluate

---

## Key Hyperparameters

### Part 1: Word2Vec
- Vocabulary size: 10,000 (+ `<UNK>`)
- Embedding dimension: 100 (C3), 200 (C4)
- Context window: 5
- Negative samples: 10
- Learning rate: 0.001 (Adam)
- Batch size: 512
- Epochs: 5

### Part 2: BiLSTM
*(To be added)*

### Part 3: Transformer
*(To be added)*

---

## Outputs

### Part 1 Deliverables

✓ **Matrices:**
- `tfidf_matrix.npy` - TF-IDF weighted term-document matrix
- `ppmi_matrix.npy` - PPMI weighted co-occurrence matrix
- `embeddings_w2v.npy` - Word2Vec embeddings (d=100)
- `embeddings_w2v_d200.npy` - Word2Vec embeddings (d=200)

✓ **Visualizations:**
- t-SNE plot of top 200 words (color-coded by category)
- Training loss curves

✓ **Evaluation:**
- Top-10 nearest neighbors for 8 query words
- 10 analogy tests with top-3 candidates
- Four-condition comparison with MRR scores

---

## Notes

- All notebook cells must be executed before submission
- All figures include axis labels and titles
- Training is done from scratch (no pretrained models)
- No PyTorch built-in Transformer classes used in Part 3

---

## GitHub Requirements

- ✓ Public repository
- ✓ Correct naming: `i23-XXXX-NLP-Assignment2`
- ✓ Meaningful commit history (≥5 commits)
- ✓ This README with reproduction instructions
- ✓ Folder structure matches submission layout

---

## Contact

**Student:** [Your Name]  
**Email:** i23-XXXX@nu.edu.pk  
**Repository:** https://github.com/yourusername/i23-XXXX-NLP-Assignment2

---

## License

This project is for academic purposes only (FAST NUCES CS-4063).
