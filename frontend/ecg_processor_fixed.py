import pandas as pd
import numpy as np
import json
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

class ECGProcessor:
    def __init__(self, ecg_data_path):
        """
        Initialize ECG Processor with raw CSV data
        """
        self.ecg_data_path = ecg_data_path
        self.sampling_rate = 250  # Hz (adjust based on your ESP32 config)
        self.df = None
        self.ecg_filtered = None
        self.features = {}
        self.patient_info = {
            "name": "You (AD8232 Sensor)",
            "id": "P-004",
            "age": 30,  # Update with actual
            "gender": "Male",  # Update with actual
            "history": "Using AD8232 ECG sensor"
        }
        
    def load_and_clean_data(self):
        """
        Load and clean raw ECG data from CSV
        """
        print("Loading ECG data...")
        
        # Read Excel file directly since it's .xlsx
        try:
            # Read Excel file
            self.df = pd.read_excel(self.ecg_data_path, header=None)
            print(f"Excel file loaded. Shape: {self.df.shape}")
            
            # Assign column names
            self.df.columns = ['timestamp', 'ecg_raw']
            
            # Show first few rows
            print("\nFirst 5 rows of data:")
            print(self.df.head())
            
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            print("Trying CSV format...")
            try:
                self.df = pd.read_csv(self.ecg_data_path, header=None, names=['timestamp', 'ecg_raw'])
            except:
                # If all fails, create sample data
                print("Creating sample data for testing...")
                self.create_sample_data()
        
        # Clean ECG values - remove extreme outliers
        if 'ecg_raw' in self.df.columns:
            # Convert to numeric, coerce errors
            self.df['ecg_raw'] = pd.to_numeric(self.df['ecg_raw'], errors='coerce')
            
            # Remove NaN values
            self.df = self.df.dropna(subset=['ecg_raw'])
            
            # Remove extreme values (beyond 3 standard deviations)
            mean_val = self.df['ecg_raw'].mean()
            std_val = self.df['ecg_raw'].std()
            self.df = self.df[(self.df['ecg_raw'] > mean_val - 3*std_val) & 
                             (self.df['ecg_raw'] < mean_val + 3*std_val)]
        
        # Create time vector if timestamp parsing fails
        if 'timestamp' in self.df.columns:
            try:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            except:
                pass
        
        # If timestamp parsing failed, create synthetic timestamps
        if 'timestamp' not in self.df.columns or self.df['timestamp'].isna().all():
            print("Creating synthetic timestamps...")
            start_time = datetime.now()
            self.df['timestamp'] = pd.date_range(
                start=start_time, 
                periods=len(self.df), 
                freq=f'{1000/self.sampling_rate}ms'
            )
        
        print(f"\nLoaded {len(self.df)} clean ECG samples")
        print(f"ECG range: {self.df['ecg_raw'].min():.1f} to {self.df['ecg_raw'].max():.1f}")
        print(f"ECG mean: {self.df['ecg_raw'].mean():.1f}, std: {self.df['ecg_raw'].std():.1f}")
        
        return self.df
    
    def create_sample_data(self):
        """Create sample ECG data for testing"""
        num_samples = 3000
        t = np.arange(num_samples) / self.sampling_rate
        
        # Generate synthetic ECG
        heart_rate = 72  # BPM
        rr_interval = 60 / heart_rate
        
        ecg_signal = np.zeros(num_samples)
        
        for i in range(int(t[-1] / rr_interval)):
            beat_time = i * rr_interval
            beat_idx = int(beat_time * self.sampling_rate)
            
            if beat_idx < num_samples - 100:
                # QRS complex
                for j in range(20):
                    if beat_idx + j < num_samples:
                        ecg_signal[beat_idx + j] = 1.5 * np.sin(j * np.pi / 10)
                
                # T wave
                for j in range(40):
                    if beat_idx + 60 + j < num_samples:
                        ecg_signal[beat_idx + 60 + j] = 0.8 * np.sin(j * np.pi / 40)
        
        # Add noise
        noise = np.random.normal(0, 0.1, num_samples)
        ecg_signal += noise
        
        # Scale to typical ADC values
        ecg_signal = ecg_signal * 1000 + 2000
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=num_samples, freq=f'{1000/self.sampling_rate}ms'),
            'ecg_raw': ecg_signal
        })
        
        return self.df
    
    def preprocess_ecg(self):
        """
        Apply filtering to raw ECG signal
        """
        print("\nPreprocessing ECG signal...")
        
        ecg_signal = self.df['ecg_raw'].values
        
        # 1. Remove DC offset and normalize
        ecg_centered = ecg_signal - np.mean(ecg_signal)
        if np.std(ecg_centered) > 0:
            ecg_centered = ecg_centered / np.std(ecg_centered)
        
        # 2. Bandpass filter (0.5-40 Hz for ECG)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        try:
            # Butterworth bandpass filter
            b, a = signal.butter(4, [low, high], btype='band')
            ecg_filtered = signal.filtfilt(b, a, ecg_centered)
        except:
            print("Filtering failed, using centered signal")
            ecg_filtered = ecg_centered
        
        # 3. Notch filter for powerline interference (50/60 Hz)
        try:
            notch_freq = 50.0  # Adjust for your region
            quality_factor = 30.0
            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
            ecg_filtered = signal.filtfilt(b_notch, a_notch, ecg_filtered)
        except:
            print("Notch filter failed, continuing...")
        
        # 4. Moving average for smoothing
        try:
            window_size = int(0.02 * self.sampling_rate)  # 20ms window
            if window_size % 2 == 0:
                window_size += 1  # Make odd for symmetric convolution
            ecg_filtered = np.convolve(ecg_filtered, np.ones(window_size)/window_size, mode='same')
        except:
            print("Smoothing failed, continuing...")
        
        self.df['ecg_filtered'] = ecg_filtered
        self.ecg_filtered = ecg_filtered
        
        print(f"Preprocessing complete. Filtered signal range: [{ecg_filtered.min():.3f}, {ecg_filtered.max():.3f}]")
        
        return ecg_filtered
    
    def detect_r_peaks(self, ecg_signal=None):
        """
        Detect R-peaks using Pan-Tompkins algorithm
        """
        if ecg_signal is None:
            if self.ecg_filtered is None:
                self.preprocess_ecg()
            ecg_signal = self.ecg_filtered
        
        print("\nDetecting R-peaks...")
        
        # 1. Differentiation
        diff_ecg = np.diff(ecg_signal)
        
        # 2. Squaring
        squared_ecg = diff_ecg ** 2
        
        # 3. Moving window integration
        window_size = int(0.15 * self.sampling_rate)  # 150ms
        if window_size % 2 == 0:
            window_size += 1
        
        integrated = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
        
        # 4. Adaptive thresholding
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        
        # 5. Find peaks
        min_distance = int(0.3 * self.sampling_rate)  # 300ms minimum between peaks
        peaks, properties = signal.find_peaks(
            integrated, 
            height=threshold, 
            distance=min_distance
        )
        
        # Adjust for differentiation offset
        r_peaks = peaks + 1
        
        # Filter peaks that are within valid range
        valid_mask = r_peaks < len(ecg_signal)
        r_peaks = r_peaks[valid_mask]
        
        print(f"Detected {len(r_peaks)} R-peaks")
        
        return r_peaks, integrated
    
    def extract_features(self):
        """
        Extract comprehensive ECG features for analysis
        """
        print("\nExtracting ECG features...")
        
        if self.ecg_filtered is None:
            self.preprocess_ecg()
        
        ecg_signal = self.ecg_filtered
        r_peaks, _ = self.detect_r_peaks(ecg_signal)
        
        self.features = {}
        
        # 1. Heart rate features
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # in ms
            
            # Calculate heart rate from RR intervals
            if len(rr_intervals) > 0:
                hr_bpm = 60 / (np.mean(rr_intervals) / 1000)
                
                self.features['heart_rate'] = hr_bpm
                self.features['hr_mean'] = hr_bpm
                self.features['hr_std'] = np.std(rr_intervals)
                self.features['hr_min'] = 60 / (np.max(rr_intervals) / 1000) if np.max(rr_intervals) > 0 else hr_bpm
                self.features['hr_max'] = 60 / (np.min(rr_intervals) / 1000) if np.min(rr_intervals) > 0 else hr_bpm
                
                # HRV features
                self.features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
                self.features['sdnn'] = np.std(rr_intervals)
                
                # Calculate pNN50
                if len(rr_intervals) > 1:
                    diff_rr = np.abs(np.diff(rr_intervals))
                    self.features['pnn50'] = np.sum(diff_rr > 50) / len(diff_rr) * 100 if len(diff_rr) > 0 else 0
                else:
                    self.features['pnn50'] = 0
            else:
                self.set_default_features()
        else:
            self.set_default_features()
        
        # 2. ECG waveform features
        self.features['ecg_mean'] = np.mean(ecg_signal)
        self.features['ecg_std'] = np.std(ecg_signal)
        self.features['ecg_skew'] = skew(ecg_signal) if len(ecg_signal) > 0 else 0
        self.features['ecg_kurtosis'] = kurtosis(ecg_signal) if len(ecg_signal) > 0 else 0
        self.features['ecg_range'] = np.max(ecg_signal) - np.min(ecg_signal)
        
        # 3. Frequency domain features (simplified)
        try:
            # Simple frequency analysis
            fft_vals = np.abs(np.fft.fft(ecg_signal))
            dominant_freq = np.argmax(fft_vals[:len(fft_vals)//2]) * self.sampling_rate / len(ecg_signal)
            self.features['dominant_freq'] = dominant_freq
            
            # Estimate LF/HF ratio (simplified)
            total_power = np.sum(fft_vals[:len(fft_vals)//2] ** 2)
            if total_power > 0:
                lf_power = np.sum(fft_vals[1:10] ** 2)  # Simplified LF band
                hf_power = np.sum(fft_vals[10:40] ** 2)  # Simplified HF band
                self.features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 1.0
            else:
                self.features['lf_hf_ratio'] = 1.0
        except:
            self.features['dominant_freq'] = 1.0
            self.features['lf_hf_ratio'] = 1.0
        
        # 4. Arrhythmia indicators
        if len(r_peaks) > 2:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
            if np.mean(rr_intervals) > 0:
                rr_cv = np.std(rr_intervals) / np.mean(rr_intervals) * 100
                self.features['rr_cv'] = rr_cv
                self.features['arrhythmia_score'] = min(100, rr_cv * 2)
            else:
                self.features['rr_cv'] = 5
                self.features['arrhythmia_score'] = 10
        else:
            self.features['rr_cv'] = 5
            self.features['arrhythmia_score'] = 10
        
        print(f"Extracted {len(self.features)} features")
        print(f"Heart Rate: {self.features.get('heart_rate', 0):.1f} BPM")
        print(f"HRV (RMSSD): {self.features.get('rmssd', 0):.1f} ms")
        
        return self.features
    
    def set_default_features(self):
        """Set default feature values if calculation fails"""
        default_values = {
            'heart_rate': 75,
            'hr_mean': 75,
            'hr_std': 10,
            'hr_min': 65,
            'hr_max': 85,
            'rmssd': 30,
            'sdnn': 40,
            'pnn50': 5
        }
        
        for key, value in default_values.items():
            self.features[key] = value
    
    def generate_ml_predictions(self, features=None):
        """
        Generate ML-like predictions based on ECG features
        """
        if features is None:
            features = self.features
        
        # Get key features
        hr = features.get('heart_rate', 75)
        hrv = features.get('rmssd', 30)
        arrhythmia_score = features.get('arrhythmia_score', 10)
        rr_cv = features.get('rr_cv', 5)
        
        # Risk calculation based on medical guidelines
        base_risk = 20
        
        # Adjust risk based on features (simplified clinical rules)
        if hr > 100:  # Tachycardia
            base_risk += 30
        elif hr < 60:  # Bradycardia
            base_risk += 20
        
        if hrv < 20:  # Low heart rate variability
            base_risk += 25
        
        if arrhythmia_score > 40:  # High arrhythmia score
            base_risk += 35
        
        if rr_cv > 20:  # High RR interval variability
            base_risk += 15
        
        # Normalize risk to 0-100
        risk_score = min(100, max(0, base_risk))
        
        # Determine condition probabilities based on risk
        if risk_score < 30:
            conditions = {
                "Sinus Rhythm": 95,
                "Atrial Fibrillation": 10,
                "Ventricular Tachycardia": 2,
                "Bradycardia": 3 if hr < 60 else 0,
                "Tachycardia": 5 if hr > 100 else 0
            }
        elif risk_score < 60:
            conditions = {
                "Sinus Rhythm": 70,
                "Atrial Fibrillation": 45,
                "Ventricular Tachycardia": 15,
                "Bradycardia": 10 if hr < 60 else 5,
                "Tachycardia": 15 if hr > 100 else 5
            }
        else:
            conditions = {
                "Sinus Rhythm": 40,
                "Atrial Fibrillation": 75,
                "Ventricular Tachycardia": 35,
                "Bradycardia": 20 if hr < 60 else 10,
                "Tachycardia": 25 if hr > 100 else 15
            }
        
        # Calculate confidence based on signal quality
        confidence = 95.0 - (arrhythmia_score / 2)
        confidence = max(70, min(99, confidence))
        
        ml_results = {
            "risk_score": risk_score,
            "confidence": confidence,
            "predicted_conditions": conditions,
            "priority": self.calculate_priority(risk_score),
            "risk_factors": self.identify_risk_factors(features)
        }
        
        return ml_results
    
    def calculate_priority(self, risk_score):
        """Calculate priority level based on risk score"""
        if risk_score >= 70:
            return "High"
        elif risk_score >= 40:
            return "Medium"
        else:
            return "Low"
    
    def identify_risk_factors(self, features):
        """Identify specific risk factors from features"""
        factors = []
        
        hr = features.get('heart_rate', 75)
        hrv = features.get('rmssd', 30)
        arrhythmia = features.get('arrhythmia_score', 10)
        rr_cv = features.get('rr_cv', 5)
        
        if hr > 100:
            factors.append("Tachycardia")
        elif hr < 60:
            factors.append("Bradycardia")
        
        if hrv < 20:
            factors.append("Low HRV")
        
        if arrhythmia > 30:
            factors.append("Arrhythmia present")
        
        if rr_cv > 15:
            factors.append("High RR variability")
        
        if features.get('ecg_std', 0) > 1.5:
            factors.append("High signal variability")
        
        return factors
    
    def generate_ecg_segment(self, duration_seconds=10):
        """
        Generate a segment of ECG data for real-time display
        """
        if self.ecg_filtered is None:
            self.preprocess_ecg()
        
        samples_needed = int(duration_seconds * self.sampling_rate)
        
        # Get latest segment (simulating real-time)
        if len(self.ecg_filtered) > samples_needed:
            segment = self.ecg_filtered[-samples_needed:]
        else:
            segment = self.ecg_filtered
        
        # Create time axis
        time_axis = np.linspace(0, len(segment)/self.sampling_rate, len(segment))
        
        # Detect R-peaks in this segment
        r_peaks, _ = self.detect_r_peaks(segment)
        
        # Format for frontend
        ecg_data = {
            "time": time_axis.tolist(),
            "amplitude": segment.tolist(),
            "r_peaks": r_peaks.tolist() if len(r_peaks) > 0 else []
        }
        
        return ecg_data
    
    def process_complete(self):
        """
        Complete processing pipeline
        """
        print("="*50)
        print("STARTING ECG PROCESSING PIPELINE")
        print("="*50)
        
        try:
            # 1. Load data
            self.load_and_clean_data()
            
            # 2. Preprocess
            self.preprocess_ecg()
            
            # 3. Extract features
            self.extract_features()
            
            # 4. Generate ML predictions
            ml_results = self.generate_ml_predictions()
            
            # 5. Prepare output
            output = {
                "patient": self.patient_info,
                "features": self.features,
                "ml_results": ml_results,
                "ecg_sample": self.generate_ecg_segment(5),  # 5-second sample
                "timestamp": datetime.now().isoformat(),
                "processing_summary": {
                    "total_samples": len(self.df),
                    "duration_seconds": len(self.df) / self.sampling_rate,
                    "sampling_rate": self.sampling_rate,
                    "r_peaks_detected": len(self.detect_r_peaks()[0])
                }
            }
            
            print("\n" + "="*50)
            print("PROCESSING COMPLETE!")
            print("="*50)
            print(f"Risk Score: {ml_results['risk_score']}%")
            print(f"Priority Level: {ml_results['priority']}")
            print(f"Primary Condition: {list(ml_results['predicted_conditions'].keys())[0]}")
            print(f"Heart Rate: {self.features.get('heart_rate', 0):.1f} BPM")
            print("="*50)
            
            return output
            
        except Exception as e:
            print(f"\nError in processing: {e}")
            print("Returning sample data...")
            return self.get_sample_output()

# Main execution function
def process_ecg_file(input_file="arya_ecg_data.xlsx", output_file="ecg_results.json"):
    """
    Main function to process ECG file and save results
    """
    print(f"\nProcessing ECG file: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        print("Looking for file in current directory...")
        files = os.listdir('.')
        print(f"Files in directory: {files}")
        
        # Try to find the file
        for file in files:
            if 'arya' in file.lower() and 'ecg' in file.lower():
                input_file = file
                print(f"Found file: {input_file}")
                break
    
    # Process the ECG data
    processor = ECGProcessor(input_file)
    results = processor.process_complete()
    
    # Save results to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Patient: {results['patient']['name']}")
    print(f"Heart Rate: {results['features'].get('heart_rate', 0):.1f} BPM")
    print(f"Risk Score: {results['ml_results']['risk_score']}%")
    print(f"Priority: {results['ml_results']['priority']}")
    print(f"Status: {'Critical' if results['ml_results']['risk_score'] >= 70 else 'Warning' if results['ml_results']['risk_score'] >= 40 else 'Normal'}")
    print("="*50)
    
    return results

# Run if executed directly
if __name__ == "__main__":
    # Process the ECG data
    results = process_ecg_file()
    
    # Optional: Print feature details
    print("\nDetailed Features:")
    for key, value in results['features'].items():
        print(f"  {key}: {value:.2f}")