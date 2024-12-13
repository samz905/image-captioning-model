# Image Captioning with Attention

This project implements an image captioning model using a Convolutional Neural Network (CNN) as an encoder and a Recurrent Neural Network (RNN) with attention mechanism as a decoder. The model is built using PyTorch and leverages a pre-trained ResNet-50 model for feature extraction.

## Model Architecture

### Encoder
The encoder is a CNN based on the ResNet-50 architecture. The pre-trained ResNet-50 model is used to extract features from input images. The final fully connected layer of ResNet-50 is removed, and the remaining layers are used to generate feature maps.

### Attention Mechanism
The attention mechanism allows the decoder to focus on different parts of the image at each time step. It computes attention weights for the feature maps and generates a context vector that is fed into the decoder.

### Decoder
The decoder is an RNN that generates captions for the input images. It uses the context vector from the attention mechanism along with the previous hidden state to generate the next word in the caption.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-captioning.git
    cd image-captioning
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python train.py
    ```

4. Generate captions for new images:
    ```bash
    python generate_captions.py image_path
    ```

## Acknowledgements

This project uses the following libraries and models:
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [ResNet-50](https://arxiv.org/abs/1512.03385)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

