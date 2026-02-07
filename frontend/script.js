// Initialize canvases
const canvas = document.getElementById("ecgCanvas");
const ctx = canvas.getContext("2d");
const compareCanvas = document.getElementById("compareChart");
const compareCtx = compareCanvas.getContext("2d");

// DOM Elements
const userSelect = document.getElementById("user");
const hrText = document.getElementById("hr");
const statusText = document.getElementById("status");
const currentUserElement = document.getElementById("current-user");
const currentTimeElement = document.getElementById("current-time");
const exportBtn = document.getElementById("export-btn");

// ECG Graph Variables
let x = 0;
let prevY = null;
const midY = canvas.height / 2;

// Patient Data
const patients = [
  { 
    name: "Alex Johnson", 
    id: "P-001",
    baseHR: 72, 
    spikeGap: 90,
    age: 45,
    condition: "Hypertension"
  },
  { 
    name: "Maria Garcia", 
    id: "P-002",
    baseHR: 88, 
    spikeGap: 70,
    age: 62,
    condition: "Arrhythmia"
  },
  { 
    name: "David Chen", 
    id: "P-003",
    baseHR: 110, 
    spikeGap: 55,
    age: 58,
    condition: "Post-Op Monitoring"
  }
];

let patientHRs = patients.map(p => p.baseHR);

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
  const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  const dateString = now.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'});
  currentTimeElement.textContent = `${dateString} | ${timeString}`;
}

// Generate realistic ECG data with variations
function generateECG(patientIndex) {
  const patient = patients[patientIndex];
  let noise = (Math.random() - 0.5) * 8;
  
  // Simulate different waveform patterns based on patient
  let spike = 0;
  if (x % patient.spikeGap === 0) {
    spike = -60; // QRS complex
  } else if (x % patient.spikeGap === 20) {
    spike = 20; // P wave
  } else if (x % patient.spikeGap === 70) {
    spike = 30; // T wave
  }
  
  // Add occasional arrhythmia for Maria Garcia (patient 1)
  if (patientIndex === 1 && Math.random() < 0.02) {
    noise += (Math.random() - 0.5) * 40;
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
  
  // Set line style based on heart rate status
  const currentHR = parseInt(hrText.textContent);
  if (currentHR > 100) {
    ctx.strokeStyle = "#E57373"; // Alert color
    ctx.shadowColor = "rgba(229, 115, 115, 0.5)";
  } else {
    ctx.strokeStyle = "#26A69A"; // Normal color
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

// Update live stats with realistic variations
function updateLiveStats(patientIndex) {
  const patient = patients[patientIndex];
  
  // Generate heart rate with realistic fluctuations
  let hr = patient.baseHR + Math.floor(Math.random() * 8 - 4);
  
  // Update UI
  hrText.textContent = hr + " BPM";
  currentUserElement.textContent = `Patient: ${patient.name} (${patient.id})`;
  
  // Update status with visual feedback
  const statusCard = document.querySelector("#stats .card:nth-child(3)");
  const statusIcon = statusCard.querySelector(".card-icon");
  
  if (hr > 100) {
    statusText.textContent = "Elevated";
    statusText.className = "value alert pulse-alert";
    statusCard.classList.add("alert");
    statusIcon.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
    statusIcon.style.background = "rgba(229, 115, 115, 0.2)";
    statusIcon.style.color = "#E57373";
  } else {
    statusText.textContent = "Normal";
    statusText.className = "value normal";
    statusCard.classList.remove("alert");
    statusIcon.innerHTML = '<i class="fas fa-bell"></i>';
    statusIcon.style.background = "rgba(187, 222, 251, 0.3)";
    statusIcon.style.color = "#0D47A1";
  }
  
  // Update patient HRs array for comparison
  patientHRs[patientIndex] = hr;
}

// Draw comparison bar chart
function drawComparisonChart() {
  const width = compareCanvas.width;
  const height = compareCanvas.height;
  
  // Clear canvas
  compareCtx.clearRect(0, 0, width, height);
  
  // Chart configuration
  const barWidth = 70;
  const barSpacing = 40;
  const chartHeight = 180;
  const baseY = height - 50;
  const maxHR = Math.max(...patientHRs) + 20;
  
  // Draw grid lines
  compareCtx.strokeStyle = "#E0E0E0";
  compareCtx.lineWidth = 1;
  
  // Horizontal grid lines
  for (let i = 0; i <= 5; i++) {
    const y = baseY - (i * chartHeight / 5);
    compareCtx.beginPath();
    compareCtx.moveTo(60, y);
    compareCtx.lineTo(width - 30, y);
    compareCtx.stroke();
    
    // Y-axis labels
    compareCtx.fillStyle = "#666";
    compareCtx.font = "12px 'Segoe UI'";
    compareCtx.textAlign = "right";
    compareCtx.fillText(Math.round(maxHR - (i * maxHR / 5)) + " BPM", 55, y + 4);
  }
  
  // Draw bars for each patient
  patientHRs.forEach((hr, i) => {
    const x = 80 + i * (barWidth + barSpacing);
    const barHeight = (hr / maxHR) * chartHeight;
    
    // Choose color based on status
    const color = hr > 100 ? "#E57373" : "#26A69A";
    const shadowColor = hr > 100 ? "rgba(229, 115, 115, 0.3)" : "rgba(38, 166, 154, 0.3)";
    
    // Draw bar with shadow effect
    compareCtx.fillStyle = color;
    compareCtx.shadowColor = shadowColor;
    compareCtx.shadowBlur = 8;
    compareCtx.shadowOffsetY = 4;
    
    compareCtx.fillRect(x, baseY - barHeight, barWidth, barHeight);
    
    // Remove shadow for text
    compareCtx.shadowColor = "transparent";
    compareCtx.shadowBlur = 0;
    compareCtx.shadowOffsetY = 0;
    
    // Draw value on top of bar
    compareCtx.fillStyle = "#333";
    compareCtx.font = "bold 14px 'Segoe UI'";
    compareCtx.textAlign = "center";
    compareCtx.fillText(hr + " BPM", x + barWidth/2, baseY - barHeight - 10);
    
    // Draw patient name below bar
    compareCtx.fillStyle = "#0D47A1";
    compareCtx.font = "13px 'Segoe UI'";
    compareCtx.fillText(patients[i].name.split(" ")[0], x + barWidth/2, baseY + 20);
  });
  
  // Draw X-axis line
  compareCtx.beginPath();
  compareCtx.moveTo(60, baseY);
  compareCtx.lineTo(width - 30, baseY);
  compareCtx.strokeStyle = "#0D47A1";
  compareCtx.lineWidth = 2;
  compareCtx.stroke();
  
  // Chart title
  compareCtx.fillStyle = "#0D47A1";
  compareCtx.font = "bold 16px 'Segoe UI'";
  compareCtx.textAlign = "center";
  compareCtx.fillText("Heart Rate Comparison (BPM)", width/2, 30);
}

// Update comparison table
function updateComparisonTable() {
  patientHRs.forEach((hr, i) => {
    // Update heart rate cells
    document.getElementById(`u${i+1}hr`).textContent = hr;
    
    // Update status cells
    const statusCell = document.getElementById(`u${i+1}status`);
    if (hr > 100) {
      statusCell.textContent = "Alert";
      statusCell.className = "status-badge status-alert";
    } else {
      statusCell.textContent = "Normal";
      statusCell.className = "status-badge status-normal";
    }
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
      
      setTimeout(() => {
        this.innerHTML = '<i class="fas fa-download"></i> Export Report';
        this.style.background = "";
        this.disabled = false;
      }, 2000);
    }, 1500);
  });
}

// Time range selector functionality
function setupTimeRangeSelector() {
  document.getElementById("timeRange").addEventListener("change", function() {
    const selectedRange = this.value;
    // In a real app, this would fetch data for the selected time range
    console.log(`Time range changed to: ${selectedRange}`);
  });
}

// Alert threshold selector functionality
function setupAlertThresholdSelector() {
  document.getElementById("alertThreshold").addEventListener("change", function() {
    const selectedThreshold = this.value;
    // In a real app, this would update alert thresholds
    console.log(`Alert threshold changed to: ${selectedThreshold}`);
  });
}

// Initialize the application
function initApp() {
  initECGCanvas();
  updateCurrentTime();
  updateLiveStats(0);
  updateComparisonTable();
  drawComparisonChart();
  
  // Setup event listeners
  setupExportButton();
  setupTimeRangeSelector();
  setupAlertThresholdSelector();
  
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
  
  // Update stats and comparison every 2 seconds
  setInterval(() => {
    const patientIndex = parseInt(userSelect.value);
    updateLiveStats(patientIndex);
    updateComparisonTable();
    drawComparisonChart();
  }, 2000);
}

// Handle window resize
window.addEventListener("resize", function() {
  initECGCanvas();
  drawComparisonChart();
});

// Initialize the app when page loads
window.addEventListener("load", initApp);