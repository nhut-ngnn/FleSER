import os
import torchaudio
import pandas as pd

def preprocess_metadata(metadata_path, min_length=160000):
    """
    Preprocess metadata to exclude audio files shorter than the minimum length.
    
    Args:
        metadata_path (str): Path to the input metadata CSV.
        output_path (str): Path to save the filtered metadata CSV.
        min_length (int): Minimum number of samples required for an audio file.
    """
    metadata = pd.read_csv(metadata_path)
    valid_entries = []

    for _, row in metadata.iterrows():
        audio_path = row['audio_file']
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            total_samples = waveform.size(1)
            
            if total_samples >= min_length:
                valid_entries.append(row)
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")

    filtered_metadata = pd.DataFrame(valid_entries)
    
    filtered_metadata.to_csv(metadata_path, index=False)
    print(f"Filtered metadata saved to {metadata_path}")

if __name__ == "__main__":
    input_metadata_path = "/home/nhut-minh-nguyen/Documents/FuzzyFusion-SER/FlexibleMMSER/metadata/MELD_metadata_val.csv"
    
    preprocess_metadata(input_metadata_path, min_length=160000)
