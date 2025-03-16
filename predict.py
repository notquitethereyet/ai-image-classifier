import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import project modules
from datasets.art_dataset import ArtDataset
from models.classifier import create_model
from utils.helper import load_checkpoint
from config import TRAIN_CONFIG, INFERENCE_CONFIG


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate predictions for AI Art Classifier")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=os.path.join(TRAIN_CONFIG['save_dir'], f"{TRAIN_CONFIG['model_name_prefix']}_{TRAIN_CONFIG['model_name']}_best.pt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=TRAIN_CONFIG['data_dir'],
        help="Directory containing dataset"
    )
    parser.add_argument(
        "--test_csv", 
        type=str, 
        default="test.csv",
        help="CSV file with test image IDs"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=INFERENCE_CONFIG['output_csv'],
        help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=INFERENCE_CONFIG['threshold'],
        help="Threshold for binary classification"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for inference"
    )
    return parser.parse_args()


def create_test_loader(data_dir, test_csv, batch_size):
    """Create a data loader for test data"""
    test_transform = transforms.Compose([
        transforms.Resize((TRAIN_CONFIG['img_size'], TRAIN_CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check test CSV structure and format
    test_csv_path = os.path.join(data_dir, test_csv)
    test_df = pd.read_csv(test_csv_path)
    
    # If the CSV has only one column with no header, add a header
    if len(test_df.columns) == 1 and test_df.columns[0] == 'id':
        print(f"Test CSV format: Single column with 'id' header")
        
        # Fix path discrepancy: replace 'test_data_v2' with 'test_data' in file paths
        test_df['id'] = test_df['id'].str.replace('test_data_v2/', 'test_data/', regex=False)
        print("Fixed file paths: Replaced 'test_data_v2/' with 'test_data/'")
        
        # Save the modified CSV
        modified_csv_path = os.path.join(data_dir, 'modified_' + test_csv)
        test_df.to_csv(modified_csv_path, index=False)
        print(f"Saved modified CSV to {modified_csv_path}")
        
        # Use the modified CSV
        test_csv_path = modified_csv_path
    elif len(test_df.columns) == 1 and test_df.columns[0].startswith('Unnamed'):
        # Rename the column to 'id'
        test_df.columns = ['id']
        
        # Fix path discrepancy
        test_df['id'] = test_df['id'].str.replace('test_data_v2/', 'test_data/', regex=False)
        print("Fixed file paths: Replaced 'test_data_v2/' with 'test_data/'")
        
        # Save the modified CSV
        modified_csv_path = os.path.join(data_dir, 'modified_' + test_csv)
        test_df.to_csv(modified_csv_path, index=False)
        print(f"Saved modified CSV to {modified_csv_path}")
        
        # Use the modified CSV
        test_csv_path = modified_csv_path
    
    test_dataset = ArtDataset(
        csv_file=test_csv_path,
        root_dir=data_dir,
        transform=test_transform,
        is_test=True  # Test set has no labels
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )
    
    return test_loader


def generate_predictions(model, test_loader, device, threshold=0.5):
    """Generate predictions on test data"""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images = batch.to(device)
            
            outputs = model(images)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
    
    return all_preds, all_probs


def save_predictions(test_csv, predictions, probabilities, output_csv):
    """Save predictions to CSV file"""
    # Read test CSV to get image IDs
    test_df = pd.read_csv(test_csv, dtype={0: str})
    
    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        'id': test_df['id'],
        'probability': probabilities,
        'prediction': predictions
    })
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # Create a submission file (only id and prediction)
    submission_path = os.path.splitext(output_csv)[0] + '_submission.csv'
    
    # Check if we need to restore original path format (test_data_v2)
    submission_df = pd.DataFrame()
    
    # If paths were modified to test_data, convert them back for submission
    if 'test_data/' in results_df['id'].iloc[0] and 'test_data_v2/' not in results_df['id'].iloc[0]:
        submission_df['id'] = results_df['id'].str.replace('test_data/', 'test_data_v2/', regex=False)
        print("Restored original file paths for submission (test_data_v2/)")
    else:
        submission_df['id'] = results_df['id']
        
    submission_df['label'] = predictions
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")
    
    # Print prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"Total samples: {len(predictions)}")
    print(f"Predicted as AI-generated (1): {sum(predictions)} ({sum(predictions)/len(predictions)*100:.2f}%)")
    print(f"Predicted as human-made (0): {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.2f}%)")


def main():
    """Main prediction function"""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(TRAIN_CONFIG)
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(model, checkpoint_path=args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}, accuracy: {checkpoint['accuracy']:.4f}")
    
    # Move model to device
    model = model.to(device)
    
    # Create test data loader
    test_loader = create_test_loader(args.data_dir, args.test_csv, args.batch_size)
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Generate predictions
    predictions, probabilities = generate_predictions(model, test_loader, device, args.threshold)
    
    # Save predictions
    save_predictions(
        os.path.join(args.data_dir, args.test_csv),
        predictions,
        probabilities,
        args.output_csv
    )


if __name__ == "__main__":
    main()