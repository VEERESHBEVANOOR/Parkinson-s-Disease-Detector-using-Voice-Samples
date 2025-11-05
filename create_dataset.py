import parselmouth
from parselmouth.praat import call
import pandas as pd
import numpy as np
import glob
import os

# Function to extract features from a single audio file
def extract_features(file_path):
    """
    Extracts vocal features from an audio file using Parselmouth.
    """
    try:
        sound = parselmouth.Sound(file_path)
        
        # Extract pitch and point process for jitter and shimmer
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        # Jitter features
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer features
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Harmonics-to-Noise Ratio (HNR)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = sound.to_mfcc(number_of_coefficients=12)
        # --- THIS IS THE CORRECTED LINE ---
        mfcc_mean = np.mean(mfcc, axis=1) # Get mean of each coefficient
        
        # Create a dictionary of features
        features = {
            'local_jitter': local_jitter,
            'rap_jitter': rap_jitter,
            'local_shimmer': local_shimmer,
            'hnr': hnr
        }
        
        # Add MFCC means to the features dictionary
        for i in range(len(mfcc_mean)):
            features[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
            
        return features

    except Exception as e:
        # Praat errors can occur if the sound is unvoiced or too noisy
        print(f"Could not process {file_path}: {e}")
        return None

# --- Main script ---
# Define paths to your data folders
parkinsons_path = 'parkinsons/*.wav' # Use your folder name
healthy_path = 'healthy/*.wav'     # Use your folder name

# Get list of all audio files
parkinsons_files = glob.glob(parkinsons_path)
healthy_files = glob.glob(healthy_path)

# Check if files were found
if not parkinsons_files and not healthy_files:
    print("Warning: No .wav files found. Check your folder names and paths in the script.")
    exit()

all_files = parkinsons_files + healthy_files

# Create labels (1 for Parkinson's, 0 for healthy)
labels = [1] * len(parkinsons_files) + [0] * len(healthy_files)

# Process all files and store features in a list
feature_list = []
successful_labels = []

for i, file in enumerate(all_files):
    print(f"Processing {file}...")
    features = extract_features(file)
    if features: # Only add if feature extraction was successful
        feature_list.append(features)
        successful_labels.append(labels[i])

# Create a DataFrame
if not feature_list:
     print("\nError: No features were extracted. The dataset is empty. Please check your audio files.")
else:
    df = pd.DataFrame(feature_list)
    df['status'] = successful_labels # Add the labels as the target column

    # Save the DataFrame to a CSV file
    df.to_csv('parkinsons_features.csv', index=False)

    print("\nFeature extraction complete. Dataset saved to parkinsons_features.csv")
    print("Dataset preview:")
    print(df.head())
    print("\nDataset Info:")
    df.info()