# 📖 Next-Word Prediction using Bidirectional LSTMs  

## 🚀 Project Overview  

This repository contains a **Next-Word Prediction Model** built using **Bidirectional Long Short-Term Memory (BiLSTM)** networks. The model is trained to predict the next word in a sentence given a sequence of preceding words. It leverages **PyTorch** for deep learning implementation and **NLTK** for text preprocessing.  

## 🛠 Features
 
✅ **Deep Learning Model**: Uses **Bidirectional LSTMs** to improve context awareness.  
✅ **Trained on a Large Text Corpus**: Enhances generalization and accuracy.  
✅ **PyTorch Implementation**: Efficient training and inference pipeline.  
✅ **Tokenization with NLTK**: Ensures structured preprocessing of text data.  
✅ **Pre-Trained Model Included**: Easily load the trained `.pt` file for inference.  

## 📂 Project Structure  

The repository consists of:  
- `next_word_prediction_using_bidirectional_lstms.ipynb` → Jupyter Notebook with full model implementation, training, and evaluation.  
- `next_word_predictor.pt` → Pre-trained PyTorch model for inference.  

## 🔧 Installation & Setup  
To set up the project, follow these steps:  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/SayamAlt/Next-Word-Prediction-using-Bidirectional-LSTMs.git
cd Next-Word-Prediction-using-Bidirectional-LSTMs
```

### 2️⃣ Install Dependencies

Ensure you have Python 3.8+ installed, then install the required libraries:

```bash
pip install nltk torch torchinfo numpy pandas matplotlib
```
### 🏗 Bidirectional LSTM (BiLSTM) Model

The **Bidirectional LSTM (BiLSTM)** model consists of:

- **Embedding Layer**: Converts words into dense vector representations.
- **Bidirectional LSTM**: Captures both past and future context.
- **Fully Connected Layer**: Predicts the next word.

### LSTM Updates

Mathematically, the LSTM updates are given by:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

$$
C~t = \tanh(W_C [h_{t-1}, x_t] + b_C)
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

where:

- \( f_t, i_t, o_t \) are the **forget, input, and output gates**.
- \( C_t \) is the **cell state**.
- \( h_t \) is the **hidden state**.