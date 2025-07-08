# 🛍️ Ecommerce Product Descriptions with Attention Mechanism

This repository contains the implementation of an image captioning system that generates product descriptions from catalog images using an encoder-decoder architecture with an attention mechanism. The model is built with PyTorch and trained on a structured ecommerce dataset.

---

## Features

- **CNN Encoder**: Uses a pre-trained ResNet50 to extract spatial image features.
- **RNN Decoder**: LSTM-based decoder generates captions token-by-token.
- **Visual Attention**: Focuses the model on relevant image regions during caption generation.
- **Evaluation Metrics**: Tracks validation loss and BLEU scores to monitor learning progress.
- **Preprocessing Pipelines**: Includes image transformations, vocabulary construction, and caption encoding.

---

## Dataset

**Fashion Product Images (Small)**

- **Description**: Product images paired with structured product titles/descriptions.
- **Examples**:
  - Input Image: Fashion product photo  
  - Target Caption: `"Peter England Men Striped Green Shirt"`
- **Splits**:
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

To download the dataset, this project leverages the KaggleHub library:

```bash
kagglehub.dataset_download('paramaggarwal/fashion-product-images-small')
```

---

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Pandas
- tqdm
- matplotlib
- nltk
- spacy

---

## Model Architecture

### Encoder
- Pre-trained **ResNet50**, last layers removed, weights frozen (optional).
- Outputs spatial feature maps used for attention.

### Attention
- MLP-based soft attention (Bahdanau-style) between encoder features and decoder hidden state.

### Decoder
- Embedding layer (size: 300)
- LSTM with hidden size 512
- Linear layers for attention projection and vocabulary prediction

---

## Preprocessing

### Images
- Resize to `224x224`
- Normalize with ImageNet mean and std

### Captions
- Tokenized using spaCy
- Vocabulary built using frequency thresholding (e.g., ignore words appearing < 5 times)
- Special tokens: `<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`

---

## Training Details

- **Loss**: CrossEntropyLoss (ignores `<PAD>`)
- **Optimizer**: Adam (lr = `3e-4`)
- **Batch Size**: 32
- **Epochs**: 15
- **Evaluation**: BLEU score (with smoothing)

Checkpoints saved after each epoch with:
- Model weights
- Optimizer state
- Training/Validation loss logs

---

## Evaluation

- **Metrics**:
  - Validation Loss
  - BLEU-1 to BLEU-4 scores

- **Inference**:
  - Generates captions for test images
  - Captions auto-stop at `<EOS>`
  - `<PAD>`, `<SOS>`, `<UNK>`, `<EOS>` are excluded from evaluation

---

## Limitations

- May overfit frequent tokens (like “men”, “shirt”) if dataset is imbalanced.
- `<UNK>` predictions still occur if vocab coverage is low.
- BLEU score may be low despite useful outputs due to phrasing mismatch.

---

## Future Work

- Fine-tune the encoder on fashion/product datasets
- Add beam search for better caption diversity
- Train on larger datasets (e.g. Flipkart, DeepFashion)
- Replace LSTM with Transformer decoder
- Add attribute-conditioning (e.g., color, category, brand embeddings)

---

## Acknowledgments

- [Fashion Product Images (Small) Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- PyTorch and its amazing community
- KaggleHub library for seamless dataset integration

---

## License

MIT License – see `LICENSE` file for details.

---

Feel free to open an issue for questions or feedback!
