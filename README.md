# AI vs Human Art Classifier

This project implements a deep learning model to distinguish between AI-generated and human-made artwork. Using transfer learning with state-of-the-art convolutional neural networks, the classifier achieves high accuracy in identifying subtle differences between AI and human artistic styles.

## Project Structure

```
ai-image-classifier/
├── data/                  # Dataset directory
│   ├── train_data/        # Training images
│   ├── test_data/         # Test images
│   ├── train.csv          # Training labels
│   └── test.csv           # Test image IDs
├── models/                # Model architectures
│   ├── __init__.py
│   └── classifier.py
├── datasets/              # Data loading and preprocessing
│   ├── __init__.py
│   └── art_dataset.py
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── helper.py
├── saved_models/          # Saved model checkpoints
├── config.py              # Configuration settings
├── train.py               # Training script
├── predict.py             # Inference script
├── evaluate.py            # Evaluation script
└── requirements.txt       # Package dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.12
- CUDA-compatible GPU with at least 8GB VRAM (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-art-classifier.git
   cd ai-art-classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your dataset:
   - Place training images in `data/train_data/`
   - Place test images in `data/test_data/`
   - Ensure `train.csv` and `test.csv` are in the `data/` directory

## Dataset

This project uses the [Shutterstock Dataset for AI vs Human Generated Images](https://www.kaggle.com/datasets/shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image/data) from Kaggle.

### Dataset Format

- **train.csv**: Contains image paths and binary labels
  ```
  ,file_name,label
  0,train_data/image1.jpg,1
  1,train_data/image2.jpg,0
  ```
  where `1` indicates AI-generated art and `0` indicates human-made art

- **test.csv**: Contains image paths for prediction
  ```
  id
  test_data/test_image1.jpg
  test_data/test_image2.jpg
  ```

## Model Architecture

The default model uses a ResNet50 backbone pretrained on ImageNet, with a custom classifier head:

- Backbone: ResNet50 (or optionally EfficientNet-B2 or ViT-B/16)
- Custom classifier head with dropout for regularization
- Binary classification using BCE with logits loss
- Mixed precision training for improved performance

## Usage

### Training

```bash
python train.py
```

The training script:
- Splits the data into training and validation sets
- Applies data augmentation (random flips, rotations, color jitter)
- Trains the model with early stopping
- Saves checkpoints and training history

Training parameters can be modified in `config.py`.

### Evaluation

```bash
python evaluate.py
```

The evaluation script:
- Loads the best model checkpoint
- Computes metrics (accuracy, precision, recall, F1, AUC)
- Generates visualization plots (ROC curve, precision-recall curve, confusion matrix)
- Finds the optimal classification threshold

### Prediction

```bash
python predict.py
```

The prediction script:
- Loads the best model checkpoint
- Generates predictions for test images
- Creates two output files:
  - `predictions.csv` with detailed probabilities and predictions
  - `predictions_submission.csv` with just the IDs and binary labels

## Results

The model achieves:
- Validation accuracy: ~XX%
- Validation AUC: ~X.XX
- F1 score: ~X.XX

## Configuration

Model and training parameters can be configured in `config.py`. Key options include:

- `model_name`: Choose between 'resnet50', 'efficientnet_b2', or 'vit_b_16'
- `batch_size`: Adjust based on available GPU memory
- `learning_rate`: Default is 1e-4
- `epochs`: Default is 20
- `mixed_precision`: Enable for faster training on compatible GPUs

## License

Do what you want, dawg.

## Acknowledgements

- This project uses PyTorch and torchvision for deep learning models
- Pretrained models are based on the ImageNet dataset