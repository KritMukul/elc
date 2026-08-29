import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, iirnotch
import sys
import glob

def apply_filters(data, fs):
    """
    Applies a 50Hz notch filter (powerline) and a 0.5-45Hz bandpass filter.
    """
    # 1. Notch Filter at 50Hz
    nyq = 0.5 * fs
    freq = 50.0
    q = 30.0 # Quality factor
    b_notch, a_notch = iirnotch(freq, q, fs)
    y = filtfilt(b_notch, a_notch, data)
    
    # 2. Bandpass Filter (0.5 Hz to 45 Hz)
    low = 0.5 / nyq
    high = 45.0 / nyq
    b_band, a_band = butter(4, [low, high], btype='band')
    y_filtered = filtfilt(b_band, a_band, y)
    
    return y_filtered

def extract_band_power(data, fs):
    """
    Extracts relative power for Delta, Theta, Alpha, Beta, Gamma bands using Welch's PSD.
    """
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta':  (13, 30),
        'Gamma': (30, 45)
    }
    
    # Compute Power Spectral Density
    freqs, psd = welch(data, fs, nperseg=fs*2) # 2-second windows
    
    band_powers = {}
    total_power = np.sum(psd[(freqs >= 1) & (freqs <= 45)])
    
    for band_name, (f_min, f_max) in bands.items():
        idx_band = np.logical_and(freqs >= f_min, freqs <= f_max)
        bp = np.sum(psd[idx_band])
        # Relative power
        band_powers[band_name] = bp / total_power if total_power > 0 else 0
        
    return band_powers

def process_file(csv_file):
    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if len(df) < 100:
        print("Not enough data points in CSV.")
        return None
        
    # Estimate Sampling Frequency (fs)
    # Timestamp is in seconds (since start)
    total_time = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]
    total_samples = len(df)
    fs = int(total_samples / total_time)
    print(f"Estimated Sampling Rate: ~{fs} Hz")
    
    if fs < 50:
        print("Warning: Sampling rate is very low. Frequency extraction might be inaccurate.")
    
    features = {}
    
    # Process each channel
    for ch in ['Ch0_V', 'Ch1_V', 'Ch2_V', 'Ch3_V']:
        if ch in df.columns:
            raw_data = df[ch].values
            
            # Remove NaNs (replace with 0 or mean)
            raw_data = np.nan_to_num(raw_data, nan=np.nanmean(raw_data))
            
            # 1. Filter
            filtered_data = apply_filters(raw_data, fs)
            
            # 2. Extract Power Bands
            bp = extract_band_power(filtered_data, fs)
            
            # Store with channel prefix
            for band_name, val in bp.items():
                features[f"{ch}_{band_name}"] = val
                
    return features

def main():
    # Find the latest raw CSV file
    csv_files = glob.glob("raw_eeg_data_*.csv")
    if not csv_files:
        print("No raw EEG CSV files found! Run 1_acquire_eeg.py first.")
        sys.exit(1)
        
    latest_file = max(csv_files, key=lambda x: x)
    
    # Process the file
    features = process_file(latest_file)
    
    if features:
        # Save features to a new CSV ready for ML
        out_filename = "extracted_features.csv"
        # We append to it so you can build a dataset over multiple recordings
        df_out = pd.DataFrame([features])
        df_out.to_csv(out_filename, mode='a', header=not pd.io.common.file_exists(out_filename), index=False)
        print(f"\nSuccess! Features extracted and saved to {out_filename}")
        print("These features (Theta power, Beta power, etc.) are now ready to be fed into your ML Model!")
        print("\nExtracted Values for this session:")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
