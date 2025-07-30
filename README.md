# 🌐 English–Tamil Neural Machine Translation (Encoder–Decoder Transformer)

A custom-built encoder–decoder Transformer model for English–Tamil sentence-level machine translation.  
Trained from scratch on 11,000 English–Tamil parallel sentence pairs using a custom word-level tokenizer.

---

## 📁 Project Structure

This implementation is currently fully contained within a Jupyter Notebook for simplicity and transparency.  
The notebook walks through the entire pipeline — from tokenization and preprocessing to training and evaluation.

Python script files (`model.py`, `train.py`, etc.) have been scaffolded and will be populated to modularize the codebase for production-readiness.

> 🔧 For now, please refer to the notebook for the complete working implementation.

---

## Features

- ✅ Custom-built encoder–decoder Transformer architecture (from scratch)
- 🌍 Trained on 11,000 English–Tamil parallel sentence pairs
- 🔡 Word-level tokenizer tailored for both English and Tamil scripts
- 🧠 ~4 million trainable parameters
- 🗂️ Modular `.py` scripts planned (`model.py`, `train.py`, etc.)
- 🖥️ Gradio-based demo interface planned for real-time translation
