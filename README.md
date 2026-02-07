# cation

# 📊 Real-Time ECG Signal Analysis with IoT and 5G Communication

## 📌 Project Overview

This project implements a **real-time ECG monitoring and analysis system** using **IoT devices, 5G communication, signal processing, and machine learning**.

Raw ECG signals are acquired using an **ECG sensor module interfaced with ESP32** and transmitted over the internet using IoT protocols. The live ECG data is visualized on **Ubidots** and stored as CSV files. These CSV files are processed using **Python-based signal processing and ML algorithms** to clean the signal, extract features, classify patient health status, and predict possible cardiac conditions.

The system supports **multiple concurrent ECG streams** and dynamically **prioritizes patients** based on the severity of their condition. A **web-based interface** displays the patient priority list and provides detailed analysis for each patient on request.

---

## 🎯 Objectives

- Real-time ECG data acquisition using ESP32
- Transmission of ECG data using IoT over 5G networks
- Cloud-based visualization of ECG signals
- Signal denoising and preprocessing
- Health status classification (Healthy / Unhealthy)
- Prediction of possible cardiac conditions
- Risk-based patient prioritization
- Centralized monitoring through a web dashboard
- Secure and structured storage of medical data

---

## 🧠 Key Features

- Real-time ECG signal acquisition
- IoT-based data transmission
- Live ECG visualization using Ubidots
- CSV-based ECG data handling
- Signal cleaning and feature extraction
- Machine learning-based health prediction
- Multi-patient priority assignment
- SQL-backed backend for structured data storage
- Interactive web dashboard for monitoring

---

## 🛠️ Technologies Used

### Hardware

- ESP32 Microcontroller
- ECG Sensor Module (AD8232 or equivalent)

### Communication

- IoT Protocols (HTTP / MQTT)
- 5G Communication Network

### Software & Tools

- Python
- NumPy, Pandas, SciPy
- Ubibots (IoT Cloud Platform)
- HTML, CSS, JavaScript (Frontend)
- SQL (MySQL)
- Git & GitHub

---

## 📁 Project File Structure

Cation/
│
├── hardware/
│ ├── python code here
│
├── data/
│ ├── raw/
│ │ └── all raw data here
│ ├── clean/
│ └── all cleaned data here
│
├── backend/
│ ├── backend and database files/scripts here
│
├── frontend/
│ ├── frontend code files/folders here
│
├── README.md

---

## ⚙️ System Workflow

1. ECG sensor captures raw ECG signals
2. ESP32 reads and transmits ECG data via IoT (5G transmission)
3. Data is visualized in real-time on Ubibots
4. ECG data is exported and stored as CSV files
5. Python scripts clean and preprocess ECG signals
6. Features are extracted from cleaned ECG data
7. ML model classifies health status and predicts conditions
8. Risk score is calculated for each patient
9. Patients are prioritized based on severity
10. Results are stored in SQL database
11. Web dashboard displays priority list and patient analysis

---

## 🔐 Data Security and SQL Database Role

The SQL database is used for **secure and structured storage of patient-related data after transmission**, including:

- Patient metadata
- ECG file references
- Analysis results
- Predicted conditions
- Risk scores and priority levels
- Time-stamped medical records

While network-level security mechanisms protect data during transmission, the SQL database ensures:

- Data persistence and integrity
- Controlled backend access
- Historical ECG analysis
- Traceability for clinical decision-making

---

📊 Output & Results

Real-time ECG waveform visualization

Health status classification

Cardiac condition prediction

Dynamic patient priority list

Web-based monitoring dashboard

🚀 Future Enhancements

End-to-end encryption of ECG data

Real-time ML inference without CSV dependency

Deep learning-based ECG analysis

Automated alerts for critical patients

Role-based access control (RBAC)

Compliance with healthcare data standards

Mobile application for clinicians

👤 Author

Anish Thiagarajan
Arya Banerjee
Kinshuk
Swagat Mallik
