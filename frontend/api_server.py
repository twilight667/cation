#!/usr/bin/env python3
"""
Simple API server to serve ECG data to the frontend
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from urllib.parse import urlparse, parse_qs

class ECGAPIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        # Serve API endpoints
        if parsed_path.path == '/api/ecg-data':
            self.serve_ecg_data()
        elif parsed_path.path == '/api/patients':
            self.serve_patients()
        elif parsed_path.path == '/api/real-time':
            self.serve_realtime_data()
        else:
            # Serve static files
            super().do_GET()
    
    def serve_ecg_data(self):
        """Serve processed ECG data"""
        try:
            with open('ecg_results.json', 'r') as f:
                ecg_data = json.load(f)
            
            # Add simulated patients
            ecg_data['all_patients'] = self.get_all_patients(ecg_data)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(ecg_data).encode())
            
        except FileNotFoundError:
            self.send_error(404, "ECG data not found. Run the processor first.")
    
    def serve_patients(self):
        """Serve all patients data"""
        patients = self.get_all_patients()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(patients).encode())
    
    def serve_realtime_data(self):
        """Serve real-time ECG updates (simulated)"""
        import random
        import time
        
        # Generate real-time-like data
        realtime_data = {
            "timestamp": time.time(),
            "heart_rate": 70 + random.randint(-5, 5),
            "ecg_sample": self.generate_ecg_sample(),
            "risk_score": 25 + random.randint(-3, 3)
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(realtime_data).encode())
    
    def get_all_patients(self, ecg_data=None):
        """Get all patients including yourself"""
        if ecg_data is None:
            try:
                with open('ecg_results.json', 'r') as f:
                    ecg_data = json.load(f)
            except:
                ecg_data = {}
        
        # Your data (from sensor)
        your_data = {
            "id": "P-004",
            "name": "You (Live Sensor)",
            "mlRiskScore": ecg_data.get('ml_results', {}).get('risk_score', 28),
            "predictedCondition": self.get_primary_condition(ecg_data.get('ml_results', {}).get('predicted_conditions', {})),
            "heartRate": int(ecg_data.get('features', {}).get('heart_rate', 72)),
            "status": self.get_status(ecg_data.get('ml_results', {}).get('risk_score', 28)),
            "actionRequired": self.get_action_required(ecg_data.get('ml_results', {}).get('priority', 'Medium')),
            "priority": ecg_data.get('ml_results', {}).get('priority', 'Medium'),
            "ecgData": ecg_data.get('ecg_sample', {}),
            "features": ecg_data.get('features', {}),
            "mlResults": ecg_data.get('ml_results', {}),
            "isRealPatient": True
        }
        
        # Simulated patients
        simulated_patients = [
            {
                "id": "P-001",
                "name": "Alex Johnson",
                "mlRiskScore": 42,
                "predictedCondition": "Sinus Rhythm",
                "heartRate": 78,
                "status": "Normal",
                "actionRequired": "Routine Check",
                "priority": "Low",
                "isRealPatient": False
            },
            {
                "id": "P-002",
                "name": "Maria Garcia",
                "mlRiskScore": 68,
                "predictedCondition": "Atrial Fibrillation",
                "heartRate": 112,
                "status": "Warning",
                "actionRequired": "Monitor Closely",
                "priority": "Medium",
                "isRealPatient": False
            },
            {
                "id": "P-003",
                "name": "David Chen",
                "mlRiskScore": 85,
                "predictedCondition": "Ventricular Tachycardia",
                "heartRate": 145,
                "status": "Critical",
                "actionRequired": "Immediate Review",
                "priority": "High",
                "isRealPatient": False
            }
        ]
        
        # Combine and sort by priority
        all_patients = [your_data] + simulated_patients
        return self.sort_patients_by_priority(all_patients)
    
    def get_primary_condition(self, conditions):
        if not conditions:
            return "Sinus Rhythm"
        
        max_prob = 0
        primary = "Sinus Rhythm"
        
        for condition, probability in conditions.items():
            if probability > max_prob:
                max_prob = probability
                primary = condition
        
        return primary
    
    def get_status(self, risk_score):
        if risk_score >= 70:
            return "Critical"
        elif risk_score >= 40:
            return "Warning"
        return "Normal"
    
    def get_action_required(self, priority):
        if priority == "High":
            return "Immediate Review"
        elif priority == "Medium":
            return "Monitor Closely"
        return "Routine Check"
    
    def sort_patients_by_priority(self, patients):
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        return sorted(patients, key=lambda x: priority_order.get(x.get("priority", "Low"), 1), reverse=True)
    
    def generate_ecg_sample(self):
        """Generate a sample ECG waveform"""
        import numpy as np
        
        sampling_rate = 250
        duration = 5
        num_samples = sampling_rate * duration
        
        t = np.linspace(0, duration, num_samples)
        ecg_wave = np.sin(2 * np.pi * 1.2 * t)  # Base frequency
        
        # Add heartbeats
        for i in range(int(duration * 1.2)):  # ~1.2 Hz = 72 BPM
            beat_time = i / 1.2
            if beat_time < duration:
                idx = int(beat_time * sampling_rate)
                if idx + 50 < len(ecg_wave):
                    # Add QRS complex
                    ecg_wave[idx:idx+20] += 1.5 * np.sin(np.linspace(0, np.pi, 20))
        
        return {
            "time": t.tolist(),
            "amplitude": ecg_wave.tolist(),
            "r_peaks": [int(i * sampling_rate / 1.2) for i in range(int(duration * 1.2)) if i * sampling_rate / 1.2 < num_samples]
        }

def run_server(port=8000):
    """Run the API server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ECGAPIHandler)
    
    print(f"ECG API Server running on http://localhost:{port}")
    print(f"Dashboard: http://localhost:{port}/dashboard.html")
    print(f"API Endpoints:")
    print(f"  - GET /api/ecg-data    : Your processed ECG data")
    print(f"  - GET /api/patients    : All patients data")
    print(f"  - GET /api/real-time   : Real-time updates")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    run_server(8000)