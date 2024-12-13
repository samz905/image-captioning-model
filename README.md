# Image Captioning with Attention Mechanism

This repository contains the implementation of an image captioning system that generates descriptive captions for images using an encoder-decoder architecture with an attention mechanism. The model is built with PyTorch and trained on the Flickr8k dataset.

---

## Features

- **CNN Encoder**: Utilizes a pre-trained ResNet50 model to extract spatial features from images.
- **RNN Decoder**: LSTM-based decoder generates captions word-by-word.
- **Attention Mechanism**: Ensures the model focuses on relevant image regions during caption generation.
- **Training Metrics**: Tracks validation loss and BLEU scores to monitor performance.
- **Preprocessing Pipelines**: Includes image transformations and caption tokenization.

---

## Dataset

**Flickr8k Dataset**

- **Description**: Contains 8,000 images, each paired with five textual captions.
- **Splits**:
  - Training: 30,000 samples
  - Validation: 5,000 samples
  - Testing: 5,455 samples

To download the dataset, this project leverages the KaggleHub library:

```bash
kagglehub.dataset_download('adityajn105/flickr8k')
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
- Pre-trained ResNet50 with frozen weights.
- Extracts spatial features from input images.

### Decoder
- Embedding layer (size: 300)
- LSTM with hidden size 512
- Attention mechanism for focusing on salient image regions.

### Training Details
- Loss Function: CrossEntropyLoss (ignores `<PAD>` tokens)
- Optimizer: Adam (learning rate = 3e-4)
- Batch Size: 32
- Epochs: 15

---

## Preprocessing

### Image Transformations
- Resizing to 224x224
- Normalization with ImageNet mean and standard deviation

### Caption Processing
- Tokenization using `spacy`
- Vocabulary construction with frequency filtering
- Padding and special tokens (`<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`)

---

## Training the Model

Progress is tracked using a tqdm progress bar, and checkpoints are saved after each epoch:

- Model weights
- Optimizer states
- Training/validation losses

---

## Evaluation

- **Metrics**:
  - Validation Loss
  - BLEU Score

- **Testing**:
  - Generates captions for test images
  - Computes BLEU scores to evaluate performance

---

## Limitations

- The model's performance is limited by the size and diversity of the Flickr8k dataset.
- Generated captions may not always be fluent or accurate for complex images.

---

## Future Work

- Use larger datasets like MSCOCO for improved training.
- Fine-tune encoder weights for better feature extraction.
- Explore transformer-based decoders for enhanced caption generation.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- [Flickr8k Dataset](https://www.kaggle.com/adityajn105/flickr8k)
- PyTorch and its amazing community
- KaggleHub library for seamless dataset integration

---

Feel free to open an issue for questions or feedback!
