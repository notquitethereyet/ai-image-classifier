# AI vs Human Art Classifier

This project implements a deep learning model to distinguish between AI-generated and human-made artwork. Using transfer learning with state-of-the-art convolutional neural networks, the classifier achieves high accuracy in identifying subtle differences between AI and human artistic styles.

## Project Structure

```
ai_art_classifier/
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

- Python 3.8+
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

The project implements a deep neural network for binary classification of images as either AI-generated (1) or human-made (0). At its core, the model uses transfer learning with state-of-the-art convolutional neural networks.

### Base Architecture

The default implementation uses ResNet50 as the backbone, but the code supports easy switching between:

- **ResNet50**: Deep residual network with 50 layers that uses skip connections to solve the vanishing gradient problem
- **EfficientNet-B2**: Compact and efficient network that uses compound scaling to balance depth, width, and resolution
- **Vision Transformer (ViT-B/16)**: Transformer-based architecture that treats image patches as tokens for sequence modeling

### Model Components

1. **Backbone Network**:
   - Pretrained on ImageNet for robust feature extraction
   - Final classification layer removed to use as a feature extractor
   - All layers fine-tuned during training (not frozen)

2. **Custom Classification Head**:
   ```
   nn.Sequential(
       nn.Dropout(0.3),           # Reduces overfitting
       nn.Linear(features, 256),   # Dense layer with 256 neurons
       nn.ReLU(),                 # Non-linear activation
       nn.Dropout(0.2),           # Additional regularization
       nn.Linear(256, 1)          # Output layer for binary classification
   )
   ```

3. **Loss Function**: 
   - BCE with Logits Loss, which combines a sigmoid activation and binary cross-entropy loss
   - Provides numerical stability during training, especially with mixed precision

4. **Regularization**:
   - Dropout layers (0.3 and 0.2 rates)
   - Weight decay in the optimizer (1e-5)
   - Data augmentation in the training pipeline

### Input Processing

- Images resized to 224×224 pixels
- Normalization using ImageNet mean and standard deviation
- Data augmentation: random horizontal flips, rotations, and color jitter

### Training Optimizations

- **Mixed Precision Training**: Uses FP16 computation where appropriate to speed up training
- **Gradient Scaling**: Prevents underflow in FP16 gradients
- **Learning Rate Scheduling**: Cosine annealing or step decay
- **Early Stopping**: Prevents overfitting by monitoring validation loss

### Performance Considerations

The model is optimized for CUDA execution on NVIDIA GPUs with at least 8GB VRAM. The architecture strikes a balance between:

- **Accuracy**: By using sophisticated pretrained feature extractors
- **Inference Speed**: Reasonable prediction time suitable for production
- **Memory Usage**: Configurable batch size for different hardware constraints

This architecture is particularly effective at picking up on subtle patterns that differentiate AI-generated from human-made artwork, including texture consistency, object boundaries, and compositional elements.

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

The model achieves excellent performance on the validation set:

- **Accuracy**: 99.70%
- **Precision**: 99.77%
- **Recall**: 99.62%
- **F1 Score**: 99.70%
- **AUC**: 99.99%

These metrics were achieved with the default threshold of 0.5. Further analysis found that an optimal threshold of 0.25 yields slightly better results:

- **Accuracy**: 99.71%
- **Precision**: 99.68%
- **Recall**: 99.75%
- **F1 Score**: 99.71%
- **AUC**: 99.99%

The extremely high performance metrics suggest the model is very effective at distinguishing between AI-generated and human-made artwork in this dataset. The nearly perfect AUC score indicates excellent separability between the two classes.

The model was trained for 11 epochs before reaching optimal performance as determined by early stopping.

### Visualization

The evaluation process generates several visualizations that help understand model performance:

- ROC curve (showing true positive rate vs false positive rate)
- Precision-recall curve
- Confusion matrix
- Threshold analysis plot

These plots are automatically saved in the evaluation directory when running the evaluation script.

## Configuration

Model and training parameters can be configured in `config.py`. Key options include:

- `model_name`: Choose between 'resnet50', 'efficientnet_b2', or 'vit_b_16'
- `batch_size`: Adjust based on available GPU memory
- `learning_rate`: Default is 1e-4
- `epochs`: Default is 20
- `mixed_precision`: Enable for faster training on compatible GPUs

## License

Do what you want with this, dawg

## Acknowledgements

- This project uses PyTorch and torchvision for deep learning models
- Pretrained models are based on the ImageNet dataset