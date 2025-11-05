import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import joblib

def extract_features(file_path):
    """
    Extracts the same features used during model training.
    """
    try:
        sound = parselmouth.Sound(file_path)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        mfcc = sound.to_mfcc(number_of_coefficients=12)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        features = {
            'local_jitter': local_jitter, 'rap_jitter': rap_jitter,
            'local_shimmer': local_shimmer, 'hnr': hnr
        }
        for i in range(len(mfcc_mean)):
            features[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
            
        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_parkinsons(audio_file_path):
    """
    Takes a new audio file path and predicts the result.
    """
    # 1. Load the trained model and scaler
    try:
        model = joblib.load('parkinsons_model.joblib')
        scaler = joblib.load('scaler.joblib')
    except FileNotFoundError:
        return "Error: Model or scaler file not found. Please train the model first."

    # 2. Extract features from the new audio file
    print(f"Extracting features from {audio_file_path}...")
    new_features = extract_features(audio_file_path)

    if new_features is None:
        return "Could not process the audio file."

    # 3. Create a DataFrame from the new features
    training_cols = scaler.get_feature_names_out()
    features_df = pd.DataFrame([new_features], columns=training_cols)

    # 4. Handle potential missing values
    features_df.fillna(0, inplace=True)

    # 5. Scale the features using the loaded scaler
    scaled_features = scaler.transform(features_df)
    
    # 6. Make a prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)

    # 7. Interpret and return the result
    if prediction[0] == 1:
        result = "Parkinson's Detected"
        confidence = prediction_proba[0][1]
    else:
        result = "Healthy"
        confidence = prediction_proba[0][0]
        
    return f"Result: {result} (Confidence: {confidence * 100:.2f}%)"

# --- HOW TO USE IT ---
if __name__ == '__main__':
    # !!! THIS LINE IS NOW UPDATED FOR YOUR FILE !!!
    new_voice_sample = 'new_voice_sample.wav' 

    if new_voice_sample == 'path/to/your/new_voice_sample.wav':
        print("Please update the 'new_voice_sample' variable in the script with a real file path.")
    else:
        final_prediction = predict_parkinsons(new_voice_sample)
        print("\n--- Prediction Complete ---")
        print(final_prediction)