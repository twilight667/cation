// Initialize canvases
const canvas = document.getElementById("ecgCanvas");
const ctx = canvas.getContext("2d");
const riskCircle = document.getElementById("risk-circle");
const riskPercent = document.getElementById("risk-percent");

// DOM Elements
const userSelect = document.getElementById("user");
const hrText = document.getElementById("hr");
const statusText = document.getElementById("status");
const riskScoreText = document.getElementById("risk-score");
const priorityLevelText = document.getElementById("priority-level");
const currentUserElement = document.getElementById("current-user");
const currentTimeElement = document.getElementById("current-time");
const exportBtn = document.getElementById("export-btn");
const hrTrend = document.getElementById("hr-trend");
const riskTrend = document.getElementById("risk-trend");
const mlConfidence = document.getElementById("ml-confidence");
const mlInsightText = document.getElementById("ml-insight-text");
const predictionTime = document.getElementById("prediction-time");
const riskCategory = document.getElementById("risk-category");
const riskFactorsCount = document.getElementById("risk-factors");
const priorityTable = document.getElementById("priority-table");

// ECG Graph Variables
let x = 0;
let prevY = null;
const midY = canvas.height / 2;

// ML Models Configuration
const mlModels = {
  xgboost: {
    name: "XGBoost Classifier",
    accuracy: 94.2,
    conditions: ["Sinus Rhythm", "Atrial Fibrillation", "Ventricular Tachycardia", "Bradycardia"]
  },
  random_forest: {
    name: "Random Forest",
    accuracy: 91.5,
    conditions: ["Sinus Rhythm", "Atrial Fibrillation", "Premature Ventricular Contractions", "Bundle Branch Block"]
  },
  neural_net: {
    name: "Neural Network",
    accuracy: 96.8,
    conditions: ["Sinus Rhythm", "Atrial Fibrillation", "Supraventricular Tachycardia", "QT Prolongation"]
  }
};

// Patient Data with ML Predictions
const patients = [
  { 
    name: "Alex Johnson", 
    id: "P-001",
    baseHR: 72, 
    spikeGap: 90,
    age: 45,
    condition: "Hypertension",
    riskScore: 28,
    riskFactors: ["Hypertension", "Family History", "Elevated LDL"],
    priority: "medium",
    queuePosition: 2,
    lastAnalysis: "2 min ago",
    mlInsights: "ECG shows regular sinus rhythm with occasional PVCs. Heart rate variability within normal limits. Low risk of arrhythmic events in next 24 hours.",
    predictedConditions: [
      { name: "Sinus Rhythm", probability: 92 },
      { name: "Atrial Fibrillation", probability: 28 },
      { name: "Ventricular Tachycardia", probability: 5 },
      { name: "Bradycardia", probability: 3 }
    ]
  },
  { 
    name: "Maria Garcia", 
    id: "P-002",
    baseHR: 88, 
    spikeGap: 70,
    age: 62,
    condition: "Arrhythmia",
    riskScore: 65,
    riskFactors: ["Arrhythmia", "Diabetes", "Obesity", "Previous MI"],
    priority: "high",
    queuePosition: 1,
    lastAnalysis: "1 min ago",
    mlInsights: "ECG shows occasional atrial fibrillation episodes with moderate HRV. Moderate risk of stroke in next 30 days. Recommend anticoagulation therapy review.",
    predictedConditions: [
      { name: "Atrial Fibrillation", probability: 78 },
      { name: "Sinus Rhythm", probability: 65 },
      { name: "Ventricular Tachycardia", probability: 22 },
      { name: "ST Depression", probability: 18 }
    ]
  },
  { 
    name: "David Chen", 
    id: "P-003",
    baseHR: 110, 
    spikeGap: 55,
    age: 58,
    condition: "Post-Op Monitoring",
    riskScore: 82,
    riskFactors: ["Post-CABG", "Hypertension", "Diabetes", "Smoking History", "Renal Impairment"],
    priority: "critical",
    queuePosition: 0,
    lastAnalysis: "30 sec ago",
    mlInsights: "ECG shows sinus tachycardia with occasional PVCs. ST segment changes noted. High risk of ischemic events. Immediate review recommended.",
    predictedConditions: [
      { name: "Sinus Tachycardia", probability: 89 },
      { name: "Ischemic Changes", probability: 45 },
      { name: "Ventricular Tachycardia", probability: 32 },
      { name: "Atrial Fibrillation", probability: 28 }
    ]
  }
];

let patientHRs = patients.map(p => p.baseHR);
let currentModel = "xgboost";

// Initialize ECG Canvas
function initECGCanvas() {
  // Set canvas dimensions
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  
  // Clear canvas with a professional background
  ctx.fillStyle = "#f9f9f9";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Draw baseline
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(canvas.width, midY);
  ctx.strokeStyle = "#E0E0E0";
  ctx.lineWidth = 1;
  ctx.stroke();
}

// Update current time
function updateCurrentTime() {
  const now = new Date();
  const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second: '2-digit'});
  const dateString = now.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'});
  currentTimeElement.textContent = `${dateString} | ${timeString}`;
}

// Generate realistic ECG data with variations
function generateECG(patientIndex) {
  const patient = patients[patientIndex];
  let noise = (Math.random() - 0.5) * 8;
  
  // Simulate different waveform patterns based on patient condition
  let spike = 0;
  if (patientIndex === 1) { // Maria - Arrhythmia
    // Irregular rhythm simulation
    if (Math.random() < 0.1) {
      spike = -80; // Larger QRS for arrhythmia
    } else if (x % (patient.spikeGap + Math.random() * 40 - 20) === 0) {
      spike = -60; // Irregular QRS complex
    }
  } else if (patientIndex === 2) { // David - Tachycardia
    if (x % (patient.spikeGap - 15) === 0) {
      spike = -70; // Faster QRS complexes
    }
  } else { // Alex - Normal
    if (x % patient.spikeGap === 0) {
      spike = -60; // Regular QRS complex
    } else if (x % patient.spikeGap === 20) {
      spike = 20; // P wave
    } else if (x % patient.spikeGap === 70) {
      spike = 30; // T wave
    }
  }
  
  return midY + noise + spike;
}

// Draw ECG waveform with professional styling
function drawECG() {
  const patientIndex = parseInt(userSelect.value);
  
  // Clear a thin vertical slice for the moving graph effect
  ctx.fillStyle = "#f9f9f9";
  ctx.fillRect(x, 0, 3, canvas.height);
  
  // Get current Y position
  const y = generateECG(patientIndex);
  
  // Draw the ECG line
  ctx.beginPath();
  
  if (prevY === null || x === 0) {
    ctx.moveTo(x, y);
  } else {
    ctx.moveTo(x - 2, prevY);
    ctx.lineTo(x, y);
  }
  
  // Set line style based on risk score
  const riskScore = parseInt(riskScoreText.textContent);
  if (riskScore > 70) {
    ctx.strokeStyle = "#E57373"; // High risk - red
    ctx.shadowColor = "rgba(229, 115, 115, 0.5)";
  } else if (riskScore > 40) {
    ctx.strokeStyle = "#FF9800"; // Medium risk - orange
    ctx.shadowColor = "rgba(255, 152, 0, 0.3)";
  } else {
    ctx.strokeStyle = "#26A69A"; // Low risk - teal
    ctx.shadowColor = "rgba(38, 166, 154, 0.3)";
  }
  
  ctx.lineWidth = 2.5;
  ctx.shadowBlur = 8;
  ctx.lineCap = "round";
  ctx.stroke();
  
  // Update position
  prevY = y;
  x += 2;
  
  // Reset position when reaching the end
  if (x >= canvas.width) {
    x = 0;
    prevY = null;
    // Redraw baseline
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(canvas.width, midY);
    ctx.strokeStyle = "#E0E0E0";
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.stroke();
  }
}

// Update risk circle visualization
function updateRiskCircle(score) {
  const circumference = 2 * Math.PI * 54; // 2πr
  const offset = circumference - (score / 100) * circumference;
  
  riskCircle.style.strokeDashoffset = offset;
  riskPercent.textContent = `${score}%`;
  
  // Update color based on risk score
  if (score > 70) {
    riskCircle.style.stroke = "#E57373";
    riskPercent.style.fill = "#E57373";
  } else if (score > 40) {
    riskCircle.style.stroke = "#FF9800";
    riskPercent.style.fill = "#FF9800";
  } else {
    riskCircle.style.stroke = "#26A69A";
    riskPercent.style.fill = "#26A69A";
  }
}

// Calculate ML Risk Score based on various factors
function calculateRiskScore(patientIndex, heartRate) {
  const patient = patients[patientIndex];
  let score = patient.riskScore;
  
  // Add variability based on heart rate
  if (heartRate > 100) {
    score += Math.random() * 15;
  } else if (heartRate > 90) {
    score += Math.random() * 10;
  } else if (heartRate < 60) {
    score += Math.random() * 8;
  }
  
  // Add variability based on model
  if (currentModel === "neural_net") {
    score += Math.random() * 5 - 2.5; // -2.5 to +2.5
  }
  
  // Add time-based variation
  const hour = new Date().getHours();
  if (hour >= 22 || hour <= 6) {
    score += 5; // Higher risk during night
  }
  
  // Ensure score is between 0-100
  score = Math.max(0, Math.min(100, Math.round(score)));
  
  return score;
}

// Update patient priority based on risk score
function updatePriority(riskScore) {
  if (riskScore >= 80) return "critical";
  if (riskScore >= 60) return "high";
  if (riskScore >= 30) return "medium";
  return "low";
}

// Get priority badge class
function getPriorityClass(priority) {
  switch(priority) {
    case "critical": return "priority-critical";
    case "high": return "priority-high";
    case "medium": return "priority-medium";
    case "low": return "priority-low";
    default: return "priority-low";
  }
}

// Get action required based on priority
function getActionRequired(priority) {
  switch(priority) {
    case "critical": return "Immediate Review";
    case "high": return "Urgent Review";
    case "medium": return "Schedule Review";
    case "low": return "Routine Monitor";
    default: return "Routine Monitor";
  }
}

// Get action badge class
function getActionClass(action) {
  switch(action) {
    case "Immediate Review": return "action-immediate";
    case "Urgent Review": return "action-review";
    case "Schedule Review": return "action-review";
    case "Routine Monitor": return "action-monitor";
    default: return "action-monitor";
  }
}

// Update ML predictions display
function updateMLPredictions(patientIndex) {
  const patient = patients[patientIndex];
  const model = mlModels[currentModel];
  
  // Update ML confidence
  mlConfidence.textContent = `${model.accuracy}%`;
  
  // Update prediction time
  const now = new Date();
  const minsAgo = Math.floor(Math.random() * 3) + 1;
  predictionTime.textContent = `${minsAgo} min ago`;
  
  // Update insights
  mlInsightText.textContent = patient.mlInsights;
  
  // Update condition probabilities (simulate ML variability)
  const conditionItems = document.querySelectorAll('.condition-item');
  patient.predictedConditions.forEach((condition, index) => {
    if (conditionItems[index]) {
      const probValue = conditionItems[index].querySelector('.probability-value');
      const probFill = conditionItems[index].querySelector('.probability-fill');
      
      // Add small random variation
      const variation = Math.random() * 6 - 3; // -3 to +3
      const newProb = Math.max(0, Math.min(100, condition.probability + variation));
      
      if (probValue) probValue.textContent = `${Math.round(newProb)}%`;
      if (probFill) {
        probFill.style.width = `${newProb}%`;
        
        // Update color based on probability
        if (newProb > 70) {
          probFill.style.backgroundColor = "#E57373";
        } else if (newProb > 30) {
          probFill.style.backgroundColor = "#FF9800";
        } else {
          probFill.style.backgroundColor = "#26A69A";
        }
      }
      
      // Update condition name if needed
      const conditionName = conditionItems[index].querySelector('.condition-name span');
      if (conditionName && index < model.conditions.length) {
        conditionName.textContent = model.conditions[index];
      }
    }
  });
}

// Update live stats with ML integration
function updateLiveStats(patientIndex) {
  const patient = patients[patientIndex];
  
  // Generate heart rate with realistic fluctuations
  let hr = patient.baseHR + Math.floor(Math.random() * 8 - 4);
  
  // Update UI
  hrText.textContent = hr + " BPM";
  currentUserElement.textContent = `Patient: ${patient.name} (${patient.id})`;
  
  // Calculate risk score
  const riskScore = calculateRiskScore(patientIndex, hr);
  riskScoreText.textContent = `${riskScore}%`;
  
  // Update risk circle visualization
  updateRiskCircle(riskScore);
  
  // Update risk category
  if (riskScore > 70) {
    riskCategory.textContent = "High Risk";
    riskCategory.style.color = "#E57373";
  } else if (riskScore > 40) {
    riskCategory.textContent = "Medium Risk";
    riskCategory.style.color = "#FF9800";
  } else {
    riskCategory.textContent = "Low Risk";
    riskCategory.style.color = "#26A69A";
  }
  
  // Update risk factors count
  const activeFactors = Math.min(patient.riskFactors.length + Math.floor(riskScore/20), 12);
  riskFactorsCount.textContent = `${activeFactors}/12`;
  
  // Update priority
  const priority = updatePriority(riskScore);
  priorityLevelText.textContent = priority.charAt(0).toUpperCase() + priority.slice(1);
  
  // Update trend indicators
  const hrChange = Math.random() > 0.5 ? '+' : '-';
  const hrChangeValue = (Math.random() * 3).toFixed(1);
  hrTrend.innerHTML = `${hrChange}${hrChangeValue}%`;
  hrTrend.className = hrChange === '+' ? 'card-trend trend-up' : 'card-trend trend-down';
  
  const riskChange = Math.random() > 0.3 ? '+' : '-';
  const riskChangeValue = (Math.random() * 4).toFixed(1);
  riskTrend.innerHTML = `${riskChange}${riskChangeValue}%`;
  riskTrend.className = riskChange === '+' ? 'card-trend trend-up' : 'card-trend trend-down';
  
  // Update status with visual feedback
  const statusCard = document.querySelector("#stats .card:nth-child(3)");
  const statusIcon = statusCard.querySelector(".card-icon");
  
  if (riskScore > 70) {
    statusText.textContent = "Critical";
    statusText.className = "value alert pulse-alert";
    statusCard.classList.add("alert");
    statusIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
    statusIcon.style.background = "rgba(229, 115, 115, 0.2)";
    statusIcon.style.color = "#E57373";
  } else if (riskScore > 40) {
    statusText.textContent = "Elevated";
    statusText.className = "value warning";
    statusCard.classList.remove("alert");
    statusCard.classList.add("warning");
    statusIcon.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
    statusIcon.style.background = "rgba(255, 152, 0, 0.2)";
    statusIcon.style.color = "#FF9800";
  } else {
    statusText.textContent = "Normal";
    statusText.className = "value normal";
    statusCard.classList.remove("alert", "warning");
    statusIcon.innerHTML = '<i class="fas fa-bell"></i>';
    statusIcon.style.background = "rgba(187, 222, 251, 0.3)";
    statusIcon.style.color = "#0D47A1";
  }
  
  // Update patient HRs array for comparison
  patientHRs[patientIndex] = hr;
  
  // Update ML predictions
  updateMLPredictions(patientIndex);
}

// Update patient prioritization table
function updatePriorityTable() {
  // Sort patients by risk score (descending)
  const sortedPatients = [...patients].sort((a, b) => b.riskScore - a.riskScore);
  
  // Clear table
  priorityTable.innerHTML = '';
  
  // Add sorted patients to table
  sortedPatients.forEach((patient, index) => {
    const row = document.createElement('tr');
    
    // Priority column
    const priorityCell = document.createElement('td');
    const priorityBadge = document.createElement('span');
    priorityBadge.className = `priority-badge ${getPriorityClass(patient.priority)}`;
    priorityBadge.textContent = patient.priority.toUpperCase();
    priorityCell.appendChild(priorityBadge);
    
    // Patient column
    const patientCell = document.createElement('td');
    patientCell.innerHTML = `
      <div class="user-badge">
        <div class="user-avatar">${patient.name.split(' ').map(n => n[0]).join('')}</div>
        <div>
          <div>${patient.name}</div>
          <div style="font-size: 12px; color: #666;">${patient.id} | ${patient.age}y</div>
        </div>
      </div>
    `;
    
    // Risk Score column
    const riskCell = document.createElement('td');
    riskCell.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-weight: 700; font-size: 18px; color: ${patient.riskScore > 70 ? '#E57373' : patient.riskScore > 40 ? '#FF9800' : '#26A69A'}">
          ${patient.riskScore}%
        </span>
        <div style="flex: 1; height: 6px; background: #eee; border-radius: 3px; overflow: hidden;">
          <div style="height: 100%; width: ${patient.riskScore}%; background: ${patient.riskScore > 70 ? '#E57373' : patient.riskScore > 40 ? '#FF9800' : '#26A69A'}; border-radius: 3px;"></div>
        </div>
      </div>
    `;
    
    // Predicted Condition column
    const conditionCell = document.createElement('td');
    const topCondition = patient.predictedConditions.reduce((prev, current) => 
      (prev.probability > current.probability) ? prev : current
    );
    conditionCell.innerHTML = `
      <div style="font-weight: 600; margin-bottom: 4px;">${topCondition.name}</div>
      <div style="font-size: 12px; color: #666;">${topCondition.probability}% probability</div>
    `;
    
    // Heart Rate column
    const hrCell = document.createElement('td');
    hrCell.innerHTML = `
      <span style="font-weight: 700; font-size: 18px;">${patient.baseHR}</span>
      <span style="font-size: 12px; color: #666;"> BPM</span>
    `;
    
    // Status column
    const statusCell = document.createElement('td');
    const statusClass = patient.riskScore > 70 ? 'status-alert' : patient.riskScore > 40 ? 'status-alert' : 'status-normal';
    const statusText = patient.riskScore > 70 ? 'Critical' : patient.riskScore > 40 ? 'Elevated' : 'Normal';
    statusCell.innerHTML = `<span class="status-badge ${statusClass}">${statusText}</span>`;
    
    // Action Required column
    const actionCell = document.createElement('td');
    const actionRequired = getActionRequired(patient.priority);
    actionCell.innerHTML = `<span class="action-badge ${getActionClass(actionRequired)}">${actionRequired}</span>`;
    
    // Add cells to row
    row.appendChild(priorityCell);
    row.appendChild(patientCell);
    row.appendChild(riskCell);
    row.appendChild(conditionCell);
    row.appendChild(hrCell);
    row.appendChild(statusCell);
    row.appendChild(actionCell);
    
    // Add row to table
    priorityTable.appendChild(row);
  });
}

// Export report functionality
function setupExportButton() {
  exportBtn.addEventListener("click", function() {
    this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';
    this.disabled = true;
    
    // Simulate export process
    setTimeout(() => {
      this.innerHTML = '<i class="fas fa-check"></i> Report Exported';
      this.style.background = "#26A69A";
      
      // Create a temporary download link
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
        exportDate: new Date().toISOString(),
        patients: patients.map(p => ({
          name: p.name,
          id: p.id,
          riskScore: p.riskScore,
          priority: p.priority,
          predictedConditions: p.predictedConditions,
          mlInsights: p.mlInsights
        })),
        mlModel: currentModel,
        threshold: document.getElementById('riskThreshold').value
      }, null, 2));
      
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `Cation-report-${new Date().toISOString().split('T')[0]}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      document.body.removeChild(downloadAnchor);
      
      setTimeout(() => {
        this.innerHTML = '<i class="fas fa-download"></i> Export Report';
        this.style.background = "";
        this.disabled = false;
      }, 2000);
    }, 1500);
  });
}

// ML Model selector functionality
function setupMLModelSelector() {
  document.getElementById("mlModel").addEventListener("change", function() {
    currentModel = this.value;
    const patientIndex = parseInt(userSelect.value);
    updateMLPredictions(patientIndex);
    
    // Update ML badge
    const mlBadge = document.querySelector('.ml-badge span');
    mlBadge.textContent = `${mlModels[currentModel].name} Active`;
    
    console.log(`ML Model changed to: ${mlModels[currentModel].name}`);
  });
}

// Risk threshold selector functionality
function setupRiskThresholdSelector() {
  document.getElementById("riskThreshold").addEventListener("change", function() {
    const selectedThreshold = this.value;
    // Update risk factors display
    console.log(`Risk threshold changed to: ${selectedThreshold}%`);
    
    // Update priority table to reflect new threshold
    updatePriorityTable();
  });
}

// Initialize the application
function initApp() {
  initECGCanvas();
  updateCurrentTime();
  updateLiveStats(0);
  updatePriorityTable();
  
  // Setup event listeners
  setupExportButton();
  setupMLModelSelector();
  setupRiskThresholdSelector();
  
  // Patient selection change
  userSelect.addEventListener("change", function() {
    const patientIndex = parseInt(this.value);
    updateLiveStats(patientIndex);
    
    // Reset ECG drawing position for clean transition
    x = 0;
    prevY = null;
    initECGCanvas();
  });
  
  // Update time every second
  setInterval(updateCurrentTime, 1000);
  
  // Draw ECG at 60 FPS for smooth animation
  setInterval(drawECG, 16);
  
  // Update stats and ML predictions every 3 seconds
  setInterval(() => {
    const patientIndex = parseInt(userSelect.value);
    updateLiveStats(patientIndex);
    updatePriorityTable();
  }, 3000);
}

// Handle window resize
window.addEventListener("resize", function() {
  initECGCanvas();
});

// Initialize the app when page loads
window.addEventListener("load", initApp);