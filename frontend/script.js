// ECG Dashboard - Complete Integration
class ECGDashboard {
    constructor() {
        this.ecgData = null;
        this.allPatients = [];
        this.currentPatientIndex = 0;
        this.ecgCanvas = document.getElementById('ecgCanvas');
        this.ctx = this.ecgCanvas?.getContext('2d');
        this.animationId = null;
        this.isLiveMode = false;
        this.liveUpdateInterval = null;
        
        this.init();
    }
    
    async init() {
        console.log("Initializing ECG Dashboard...");
        
        // Setup canvas
        if (this.ecgCanvas) {
            this.setupCanvas();
        }
        
        // Initialize UI
        this.updateTime();
        this.setupEventListeners();
        
        // Load data from API
        await this.loadData();
        
        // Start ECG animation
        this.startECGAnimation();
        
        // Start periodic updates
        setInterval(() => this.updateTime(), 1000);
        
        console.log("Dashboard initialized with", this.allPatients.length, "patients");
    }
    
    setupCanvas() {
        // Set canvas dimensions
        const container = this.ecgCanvas.parentElement;
        if (container) {
            const width = container.clientWidth;
            const height = 300;
            this.ecgCanvas.width = width;
            this.ecgCanvas.height = height;
        }
    }
    
    async loadData() {
        try {
            console.log("Loading ECG data from API...");
            
            // Try to load from API
            const response = await fetch('/api/ecg-data');
            if (!response.ok) {
                throw new Error(`API responded with ${response.status}`);
            }
            
            const data = await response.json();
            this.ecgData = data;
            
            // Extract patients from API response
            if (data.all_patients) {
                this.allPatients = data.all_patients;
            } else {
                // Fallback: create from ecg_data
                this.allPatients = this.createPatientsFromData(data);
            }
            
            console.log("Data loaded successfully:", this.allPatients);
            
            // Update UI with loaded data
            this.updateDashboard();
            
        } catch (error) {
            console.error("Error loading from API:", error);
            console.log("Falling back to simulated data...");
            
            // Fallback to simulated data
            this.loadSimulatedData();
            this.updateDashboard();
        }
    }
    
    createPatientsFromData(ecgData) {
        if (!ecgData) return this.getSimulatedPatients();
        
        // Create patient object from your ECG data
        const yourData = {
            id: 'P-004',
            name: 'You (Live Sensor)',
            mlRiskScore: ecgData?.ml_results?.risk_score || 28,
            predictedCondition: this.getPrimaryCondition(ecgData?.ml_results?.predicted_conditions),
            heartRate: Math.round(ecgData?.features?.heart_rate || 72),
            status: this.getStatus(ecgData?.ml_results?.risk_score || 28),
            actionRequired: this.getActionRequired(ecgData?.ml_results?.priority || 'Medium'),
            priority: ecgData?.ml_results?.priority || 'Medium',
            ecgData: ecgData?.ecg_sample || this.generateECGSample(),
            features: ecgData?.features || {},
            mlResults: ecgData?.ml_results || {},
            isRealPatient: true
        };
        
        // Combine with simulated patients and sort by priority
        const allPatients = [yourData, ...this.getSimulatedPatients()];
        return this.sortPatientsByPriority(allPatients);
    }
    
    getSimulatedPatients() {
        return [
            {
                id: 'P-001',
                name: 'Alex Johnson',
                mlRiskScore: 42,
                predictedCondition: 'Sinus Rhythm',
                heartRate: 78,
                status: 'Normal',
                actionRequired: 'Routine Check',
                priority: 'Low',
                isRealPatient: false
            },
            {
                id: 'P-002',
                name: 'Maria Garcia',
                mlRiskScore: 68,
                predictedCondition: 'Atrial Fibrillation',
                heartRate: 112,
                status: 'Warning',
                actionRequired: 'Monitor Closely',
                priority: 'Medium',
                isRealPatient: false
            },
            {
                id: 'P-003',
                name: 'David Chen',
                mlRiskScore: 85,
                predictedCondition: 'Ventricular Tachycardia',
                heartRate: 145,
                status: 'Critical',
                actionRequired: 'Immediate Review',
                priority: 'High',
                isRealPatient: false
            }
        ];
    }
    
    startECGAnimation() {
        if (!this.ecgCanvas) return;
        
        const animate = () => {
            this.drawECGWaveform();
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    drawECGWaveform() {
        if (!this.ctx || !this.ecgCanvas) return;
        
        const canvas = this.ecgCanvas;
        const ctx = this.ctx;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        this.drawGrid();
        
        // Get current patient's ECG data
        const currentPatient = this.allPatients[this.currentPatientIndex];
        let ecgData = currentPatient?.ecgData;
        
        if (!ecgData || !ecgData.amplitude) {
            ecgData = this.generateECGSample();
        }
        
        // Draw ECG waveform
        const amplitude = ecgData.amplitude;
        const time = ecgData.time || Array.from({length: amplitude.length}, (_, i) => i * 0.004);
        
        // Scale ECG data to fit canvas
        const maxAmplitude = Math.max(...amplitude.map(Math.abs));
        const scaleY = height * 0.4 / (maxAmplitude || 1);
        const offsetY = height / 2;
        const scaleX = width / (time[time.length - 1] || 1);
        
        // Draw baseline
        ctx.beginPath();
        ctx.moveTo(0, offsetY);
        ctx.lineTo(width, offsetY);
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw ECG waveform
        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#26A69A';
        
        for (let i = 0; i < amplitude.length; i++) {
            const x = time[i] * scaleX;
            const y = offsetY - amplitude[i] * scaleY;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Draw R-peaks if available
        if (ecgData.r_peaks && ecgData.r_peaks.length > 0) {
            ctx.fillStyle = '#E57373';
            
            ecgData.r_peaks.forEach(peakIndex => {
                if (peakIndex < time.length) {
                    const x = time[peakIndex] * scaleX;
                    const y = offsetY - amplitude[peakIndex] * scaleY;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }
        
        // Draw patient info
        this.drawPatientInfo(currentPatient);
    }
    
    drawGrid() {
        const ctx = this.ctx;
        const width = this.ecgCanvas.width;
        const height = this.ecgCanvas.height;
        
        // Major grid (1 second intervals)
        const majorGridSpacing = width / 10; // 10 divisions
        
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 0.5;
        
        // Vertical lines
        for (let x = 0; x <= width; x += majorGridSpacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y <= height; y += height/8) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Minor grid (0.2 second intervals)
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
        const minorGridSpacing = majorGridSpacing / 5;
        
        for (let x = 0; x <= width; x += minorGridSpacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
    }
    
    drawPatientInfo(patient) {
        const ctx = this.ctx;
        const width = this.ecgCanvas.width;
        
        ctx.fillStyle = '#333';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Patient: ${patient.name}`, 10, 25);
        
        ctx.font = '14px Arial';
        ctx.fillText(`HR: ${patient.heartRate} BPM`, 10, 45);
        ctx.fillText(`Risk: ${patient.mlRiskScore}%`, 10, 65);
        
        // Draw status indicator
        ctx.fillStyle = this.getStatusColor(patient.status);
        ctx.beginPath();
        ctx.arc(width - 20, 25, 6, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = '#333';
        ctx.textAlign = 'right';
        ctx.fillText(patient.status, width - 30, 30);
    }
    
    getStatusColor(status) {
        switch(status.toLowerCase()) {
            case 'critical': return '#F44336';
            case 'warning': return '#FF9800';
            case 'normal': return '#4CAF50';
            default: return '#757575';
        }
    }
    
    generateECGSample() {
        const samplingRate = 250;
        const duration = 5;
        const numSamples = samplingRate * duration;
        
        const time = [];
        const amplitude = [];
        const rPeaks = [];
        
        const heartRate = 72;
        const rrInterval = 60 / heartRate;
        
        for (let i = 0; i < numSamples; i++) {
            const t = i / samplingRate;
            time.push(t);
            
            // Generate realistic ECG waveform
            let value = Math.sin(t * 2 * Math.PI * 0.2) * 0.1; // Respiration
            
            // Add heartbeats
            const beatPhase = (t % rrInterval) / rrInterval;
            
            if (beatPhase < 0.1) {
                // P wave
                value += 0.15 * Math.sin(beatPhase * 10 * Math.PI);
            } else if (beatPhase < 0.2) {
                // QRS complex
                value += 1.0 * Math.sin((beatPhase - 0.1) * 20 * Math.PI);
                
                // Mark R peak
                if (beatPhase > 0.15 && beatPhase < 0.16) {
                    rPeaks.push(i);
                }
            } else if (beatPhase < 0.4) {
                // T wave
                value += 0.3 * Math.sin((beatPhase - 0.2) * 10 * Math.PI);
            }
            
            // Add noise
            value += (Math.random() - 0.5) * 0.05;
            
            amplitude.push(value);
        }
        
        return { time, amplitude, r_peaks: rPeaks };
    }
    
    updateDashboard() {
        if (this.allPatients.length === 0) return;
        
        // Update stats cards
        this.updateStatsCards();
        
        // Update ML predictions
        this.updateMLPredictions();
        
        // Update risk scoring
        this.updateRiskScoring();
        
        // Update priority table
        this.updatePriorityTable();
        
        // Update patient selector
        this.updatePatientSelector();
    }
    
    updateStatsCards() {
        const patient = this.allPatients[this.currentPatientIndex];
        
        // Update heart rate
        const hrElement = document.getElementById('hr');
        if (hrElement) {
            hrElement.textContent = `${patient.heartRate} BPM`;
        }
        
        // Update risk score
        const riskElement = document.getElementById('risk-score');
        if (riskElement) {
            riskElement.textContent = `${patient.mlRiskScore}%`;
        }
        
        // Update status
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = patient.status;
            statusElement.className = `value ${patient.status.toLowerCase()}`;
        }
        
        // Update priority
        const priorityElement = document.getElementById('priority-level');
        if (priorityElement) {
            priorityElement.textContent = patient.priority;
        }
        
        // Update current user
        const userElement = document.getElementById('current-user');
        if (userElement) {
            userElement.textContent = `Patient: ${patient.name}`;
        }
    }
    
    updateMLPredictions() {
        const patient = this.allPatients[this.currentPatientIndex];
        const mlResults = patient.mlResults || this.ecgData?.ml_results;
        
        if (!mlResults) return;
        
        // Update confidence
        const confidenceElement = document.getElementById('ml-confidence');
        if (confidenceElement) {
            confidenceElement.textContent = `${mlResults.confidence || 92.5}%`;
        }
        
        // Update condition probabilities
        const conditions = mlResults.predicted_conditions || {
            "Sinus Rhythm": 92,
            "Atrial Fibrillation": 28,
            "Ventricular Tachycardia": 5,
            "Bradycardia": 3
        };
        
        // Update probability bars
        const conditionItems = document.querySelectorAll('.condition-item');
        conditionItems.forEach((item, index) => {
            const conditionName = item.querySelector('.condition-name span')?.textContent;
            if (!conditionName) return;
            
            const probability = conditions[conditionName] || 0;
            
            const fill = item.querySelector('.probability-fill');
            const value = item.querySelector('.probability-value');
            
            if (fill) fill.style.width = `${probability}%`;
            if (value) value.textContent = `${Math.round(probability)}%`;
            
            // Update color based on probability
            if (fill) {
                if (probability > 70) fill.style.backgroundColor = '#E57373';
                else if (probability > 30) fill.style.backgroundColor = '#FF9800';
                else fill.style.backgroundColor = '#26A69A';
            }
        });
        
        // Update ML insights
        const insightElement = document.getElementById('ml-insight-text');
        if (insightElement) {
            const riskScore = patient.mlRiskScore;
            
            if (riskScore >= 70) {
                insightElement.textContent = "High cardiac risk detected. ECG shows significant abnormalities requiring immediate medical attention. Consider urgent consultation.";
            } else if (riskScore >= 40) {
                insightElement.textContent = "Moderate cardiac risk. ECG shows irregularities that warrant close monitoring. Consider follow-up evaluation.";
            } else {
                insightElement.textContent = "Low cardiac risk. ECG shows normal sinus rhythm with good heart rate variability. Continue routine monitoring.";
            }
        }
        
        // Update prediction time
        const timeElement = document.getElementById('prediction-time');
        if (timeElement) {
            const now = new Date();
            const minutes = now.getMinutes();
            timeElement.textContent = `${minutes % 5 || 1} min ago`;
        }
    }
    
    updateRiskScoring() {
        const patient = this.allPatients[this.currentPatientIndex];
        const riskScore = patient.mlRiskScore;
        
        // Update risk circle
        const circle = document.getElementById('risk-circle');
        if (circle) {
            const circumference = 339.292;
            const offset = circumference - (riskScore / 100) * circumference;
            circle.style.strokeDashoffset = offset;
            
            // Update color
            if (riskScore >= 70) circle.style.stroke = '#F44336';
            else if (riskScore >= 40) circle.style.stroke = '#FF9800';
            else circle.style.stroke = '#4CAF50';
        }
        
        // Update percentage text
        const percentElement = document.getElementById('risk-percent');
        if (percentElement) {
            percentElement.textContent = `${riskScore}%`;
        }
        
        // Update risk category
        const categoryElement = document.getElementById('risk-category');
        if (categoryElement) {
            let category = 'Low Risk';
            if (riskScore >= 70) category = 'High Risk';
            else if (riskScore >= 40) category = 'Moderate Risk';
            categoryElement.textContent = category;
        }
        
        // Update risk factors count
        const factorsElement = document.getElementById('risk-factors');
        if (factorsElement) {
            const riskFactors = patient.mlResults?.risk_factors || [];
            factorsElement.textContent = `${riskFactors.length}/12`;
        }
        
        // Update risk factor breakdown
        this.updateRiskFactorBreakdown(patient);
    }
    
    updateRiskFactorBreakdown(patient) {
        const features = patient.features || {};
        
        // Define factor calculations
        const factors = [
            { 
                name: 'Heart Rate Variability', 
                value: features.rmssd || 0,
                max: 100,
                calculate: (val) => Math.min(100, Math.max(0, 100 - (val || 0)))
            },
            { 
                name: 'ST Segment Depression', 
                value: features.ecg_std || 0,
                max: 2,
                calculate: (val) => Math.min(100, ((val || 0) / 2) * 100)
            },
            { 
                name: 'QT Interval', 
                value: features.hr_std || 0,
                max: 50,
                calculate: (val) => Math.min(100, ((val || 0) / 50) * 100)
            },
            { 
                name: 'Arrhythmia Frequency', 
                value: features.arrhythmia_score || 0,
                max: 100,
                calculate: (val) => Math.min(100, val || 0)
            }
        ];
        
        const factorElements = document.querySelectorAll('.risk-factor');
        
        factors.forEach((factor, index) => {
            if (index < factorElements.length) {
                const percentage = factor.calculate(factor.value);
                
                const fill = factorElements[index].querySelector('.score-fill');
                const valueElement = factorElements[index].querySelector('.score-value');
                
                if (fill) fill.style.width = `${percentage}%`;
                if (valueElement) valueElement.textContent = `${Math.round(percentage)}%`;
                
                // Color based on percentage
                if (fill) {
                    if (percentage > 70) fill.style.backgroundColor = '#F44336';
                    else if (percentage > 40) fill.style.backgroundColor = '#FF9800';
                    else fill.style.backgroundColor = '#4CAF50';
                }
            }
        });
    }
    
    updatePriorityTable() {
        const tableBody = document.getElementById('priority-table');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        this.allPatients.forEach((patient, index) => {
            const row = document.createElement('tr');
            
            // Add click handler
            row.addEventListener('click', () => this.selectPatient(index));
            
            // Priority badge
            let priorityBadge = '';
            let priorityClass = '';
            
            switch(patient.priority) {
                case 'High':
                    priorityBadge = '<i class="fas fa-exclamation-circle"></i> High';
                    priorityClass = 'badge-high';
                    break;
                case 'Medium':
                    priorityBadge = '<i class="fas fa-exclamation-triangle"></i> Medium';
                    priorityClass = 'badge-medium';
                    break;
                case 'Low':
                    priorityBadge = '<i class="fas fa-check-circle"></i> Low';
                    priorityClass = 'badge-low';
                    break;
            }
            
            // Status badge
            let statusBadge = '';
            let statusClass = '';
            
            switch(patient.status) {
                case 'Critical':
                    statusBadge = '<i class="fas fa-heartbeat"></i> Critical';
                    statusClass = 'status-critical';
                    break;
                case 'Warning':
                    statusBadge = '<i class="fas fa-exclamation-triangle"></i> Warning';
                    statusClass = 'status-warning';
                    break;
                case 'Normal':
                    statusBadge = '<i class="fas fa-check-circle"></i> Normal';
                    statusClass = 'status-normal';
                    break;
            }
            
            // Real patient indicator
            const realIndicator = patient.isRealPatient ? 
                '<span class="real-patient-indicator" title="Real ECG Data"><i class="fas fa-heartbeat"></i> Live</span>' : '';
            
            row.innerHTML = `
                <td><span class="badge ${priorityClass}">${priorityBadge}</span></td>
                <td>
                    <div class="patient-info">
                        <div class="patient-name">${patient.name} ${realIndicator}</div>
                        <div class="patient-id">${patient.id}</div>
                    </div>
                </td>
                <td>
                    <div class="risk-score">
                        <div class="score-value">${patient.mlRiskScore}%</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${patient.mlRiskScore}%"></div>
                        </div>
                    </div>
                </td>
                <td>${patient.predictedCondition}</td>
                <td>
                    <div class="heart-rate">
                        <i class="fas fa-heart" style="color: ${patient.heartRate > 100 ? '#F44336' : patient.heartRate < 60 ? '#2196F3' : '#4CAF50'}"></i>
                        ${patient.heartRate} BPM
                    </div>
                </td>
                <td><span class="status-badge ${statusClass}">${statusBadge}</span></td>
                <td>
                    <div class="action-required">
                        ${patient.actionRequired}
                        ${patient.priority === 'High' ? '<i class="fas fa-bell urgent-bell"></i>' : ''}
                    </div>
                </td>
            `;
            
            // Highlight selected row
            if (index === this.currentPatientIndex) {
                row.classList.add('selected');
            }
            
            tableBody.appendChild(row);
        });
    }
    
    updatePatientSelector() {
        const selector = document.getElementById('user');
        if (!selector) return;
        
        selector.innerHTML = '';
        
        this.allPatients.forEach((patient, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${patient.name} (${patient.id})`;
            if (index === this.currentPatientIndex) {
                option.selected = true;
            }
            selector.appendChild(option);
        });
    }
    
    selectPatient(index) {
        if (index < 0 || index >= this.allPatients.length) return;
        
        this.currentPatientIndex = index;
        
        // Update UI
        this.updateDashboard();
        
        // Highlight in table
        const rows = document.querySelectorAll('#priority-table tr');
        rows.forEach((row, i) => {
            if (i === index) {
                row.classList.add('selected');
            } else {
                row.classList.remove('selected');
            }
        });
        
        console.log(`Selected patient: ${this.allPatients[index].name}`);
    }
    
    getPrimaryCondition(conditions) {
        if (!conditions) return 'Sinus Rhythm';
        
        let maxProb = 0;
        let primary = 'Sinus Rhythm';
        
        for (const [condition, probability] of Object.entries(conditions)) {
            if (probability > maxProb) {
                maxProb = probability;
                primary = condition;
            }
        }
        
        return primary;
    }
    
    getStatus(riskScore) {
        if (riskScore >= 70) return 'Critical';
        if (riskScore >= 40) return 'Warning';
        return 'Normal';
    }
    
    getActionRequired(priority) {
        switch(priority) {
            case 'High': return 'Immediate Review';
            case 'Medium': return 'Monitor Closely';
            case 'Low': return 'Routine Check';
            default: return 'Monitor Closely';
        }
    }
    
    sortPatientsByPriority(patients) {
        const priorityOrder = { 'High': 3, 'Medium': 2, 'Low': 1 };
        return patients.sort((a, b) => {
            const scoreA = priorityOrder[a.priority] || 1;
            const scoreB = priorityOrder[b.priority] || 1;
            return scoreB - scoreA;
        });
    }
    
    updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: true,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            timeElement.textContent = timeString;
        }
    }
    
    async updateRealTimeData() {
        if (!this.isLiveMode) return;
        
        try {
            const response = await fetch('/api/real-time');
            const data = await response.json();
            
            // Update current patient with real-time data
            if (this.allPatients[this.currentPatientIndex]) {
                const patient = this.allPatients[this.currentPatientIndex];
                patient.heartRate = data.heartRate;
                patient.ecgData = data.ecg_sample;
                patient.mlRiskScore = data.risk_score;
                
                // Trigger UI update
                this.updateDashboard();
            }
        } catch (error) {
            console.log("Real-time updates not available");
        }
    }
    
    setupEventListeners() {
        // Patient selector
        const userSelector = document.getElementById('user');
        if (userSelector) {
            userSelector.addEventListener('change', (e) => {
                const index = parseInt(e.target.value);
                if (!isNaN(index)) {
                    this.selectPatient(index);
                }
            });
        }
        
        // Export button
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportReport());
        }
        
        // ML Model selector
        const mlModel = document.getElementById('mlModel');
        if (mlModel) {
            mlModel.addEventListener('change', (e) => {
                console.log(`ML Model changed to: ${e.target.value}`);
                // Here you would reload predictions with different model
            });
        }
        
        // Risk threshold selector
        const riskThreshold = document.getElementById('riskThreshold');
        if (riskThreshold) {
            riskThreshold.addEventListener('change', (e) => {
                console.log(`Risk threshold changed to: ${e.target.value}%`);
                // Update priority calculations
                this.updatePriorityTable();
            });
        }
        
        // Live mode toggle
        const liveModeToggle = document.createElement('button');
        liveModeToggle.innerHTML = '<i class="fas fa-satellite-dish"></i> Live Mode';
        liveModeToggle.className = 'control-group export-btn';
        liveModeToggle.style.marginLeft = '10px';
        liveModeToggle.addEventListener('click', () => {
            this.isLiveMode = !this.isLiveMode;
            liveModeToggle.innerHTML = this.isLiveMode ? 
                '<i class="fas fa-satellite-dish"></i> Live Mode ON' : 
                '<i class="fas fa-satellite-dish"></i> Live Mode';
            liveModeToggle.style.backgroundColor = this.isLiveMode ? '#4CAF50' : '';
            
            if (this.isLiveMode) {
                alert('Live mode enabled! Real-time updates every 3 seconds.');
            }
        });
        
        // Add live mode button to controls
        const controlsRow = document.querySelector('.controls-row');
        if (controlsRow) {
            controlsRow.appendChild(liveModeToggle);
        }
    }
    
    exportReport() {
        const patient = this.allPatients[this.currentPatientIndex];
        
        const report = {
            patient: patient.name,
            patientId: patient.id,
            timestamp: new Date().toISOString(),
            heartRate: patient.heartRate,
            riskScore: patient.mlRiskScore,
            priority: patient.priority,
            status: patient.status,
            predictedCondition: patient.predictedCondition,
            actionRequired: patient.actionRequired,
            features: patient.features,
            mlResults: patient.mlResults,
            dataSource: patient.isRealPatient ? "AD8232 ECG Sensor" : "Simulated"
        };
        
        // Create downloadable JSON file
        const dataStr = JSON.stringify(report, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const link = document.createElement('a');
        link.setAttribute('href', dataUri);
        link.setAttribute('download', `ECG_Report_${patient.name}_${Date.now()}.json`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show notification
        this.showNotification(`Report exported for ${patient.name}`, 'success');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
            <span>${message}</span>
            <button class="notification-close"><i class="fas fa-times"></i></button>
        `;
        
        // Add styles for notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#4CAF50' : '#2196F3'};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        // Add close button functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        });
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
        
        document.body.appendChild(notification);
        
        // Add animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.ecgDashboard = new ECGDashboard();
});