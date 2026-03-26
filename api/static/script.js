// JavaScript for Predictive Maintenance System

let predictionHistory = [];
let probabilityChart = null;

// Navigation Functions

function showHome(){

window.scrollTo({
top:0,
behavior:'smooth'
});

}

function showPredict(){

document.getElementById('predictSection').scrollIntoView({
behavior:'smooth'
});

}

function showPlots(){

document.getElementById('plotsImages').scrollIntoView({
behavior:'smooth'
});



}

// Sync sliders with input fields
document.addEventListener('DOMContentLoaded', function() {
    // Temperature slider
    const tempInput = document.getElementById('temperature');
    const tempSlider = document.getElementById('temp_slider');
    
    tempInput.addEventListener('input', function() {
        tempSlider.value = this.value;
    });
    
    tempSlider.addEventListener('input', function() {
        tempInput.value = this.value;
    });
    
    // Pressure slider
    const pressureInput = document.getElementById('pressure');
    const pressureSlider = document.getElementById('pressure_slider');
    
    pressureInput.addEventListener('input', function() {
        pressureSlider.value = this.value;
    });
    
    pressureSlider.addEventListener('input', function() {
        pressureInput.value = this.value;
    });
    
    // Vibration slider
    const vibrationInput = document.getElementById('vibration');
    const vibrationSlider = document.getElementById('vibration_slider');
    
    vibrationInput.addEventListener('input', function() {
        vibrationSlider.value = this.value;
    });
    
    vibrationSlider.addEventListener('input', function() {
        vibrationInput.value = this.value;
    });
});

// Form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        machine_id: document.getElementById('machine_id').value,
        temperature: parseFloat(document.getElementById('temperature').value),
        pressure: parseFloat(document.getElementById('pressure').value),
        vibration: parseFloat(document.getElementById('vibration').value)
    };
    
    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]');
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';
    
    try {
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayResults(data, formData);
            addToHistory(data, formData);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please try again.');
    } finally {
        // Reset button state
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Machine Health';
    }
});

function displayResults(data, inputData) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update risk indicator
    const riskValue = document.getElementById('riskValue');
    const riskLevel = data.prediction.risk_level;
    riskValue.textContent = riskLevel;
    riskValue.className = 'risk-value ' + riskLevel.toLowerCase();
    
    // Update probability bar
    const probabilityFill = document.getElementById('probabilityFill');
    const failureProb = data.prediction.failure_probability;
    probabilityFill.style.width = (failureProb * 100) + '%';
    
    document.getElementById('failureProbability').textContent = 
        (failureProb * 100).toFixed(1) + '%';
    
    // Update chart
    updateProbabilityChart(data.prediction);
    
    // Display top factors
    displayTopFactors(data.explanation.top_factors);
    
    // Display recommendations
    displayRecommendations(data.explanation.recommendation);
    
    // Update metadata
    document.getElementById('inferenceTime').textContent = 
        data.metadata.inference_time_ms.toFixed(2);
    document.getElementById('timestamp').textContent = 
        new Date(data.metadata.timestamp).toLocaleString();
}

function updateProbabilityChart(prediction) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    probabilityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal Operation', 'Failure Risk'],
            datasets: [{
                data: [
                    (prediction.normal_probability * 100).toFixed(1),
                    (prediction.failure_probability * 100).toFixed(1)
                ],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(220, 53, 69, 0.8)'
                ],
                borderColor: [
                    'rgba(40, 167, 69, 1)',
                    'rgba(220, 53, 69, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            }
        }
    });
}

function displayTopFactors(factors) {
    const container = document.getElementById('topFactors');
    container.innerHTML = '';
    
    factors.forEach(factor => {
        const factorDiv = document.createElement('div');
        factorDiv.className = 'factor-item';
        
        const impactWidth = Math.abs(factor.importance) * 100;
        const impactColor = factor.impact > 0 ? '#dc3545' : '#28a745';
        
        factorDiv.innerHTML = `
            <span class="factor-name">${factor.feature}</span>
            <div class="factor-impact">
                <span>Value: ${factor.value.toFixed(2)}</span>
                <div class="impact-bar">
                    <div class="impact-fill" style="width: ${impactWidth}%; background: ${impactColor}"></div>
                </div>
                <span>${factor.impact > 0 ? '↑' : '↓'} ${Math.abs(factor.impact).toFixed(3)}</span>
            </div>
        `;
        
        container.appendChild(factorDiv);
    });
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    container.innerHTML = '';
    
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        container.appendChild(li);
    });
}

function addToHistory(data, inputData) {
    const historyBody = document.getElementById('historyBody');
    
    // Add to history array
    predictionHistory.unshift({
        timestamp: new Date(data.metadata.timestamp),
        machineId: data.metadata.machine_id,
        riskLevel: data.prediction.risk_level,
        failureProb: data.prediction.failure_probability,
        temperature: inputData.temperature,
        pressure: inputData.pressure,
        vibration: inputData.vibration
    });
    
    // Keep only last 10 entries
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }
    
    // Update table
    historyBody.innerHTML = '';
    predictionHistory.forEach(entry => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${entry.timestamp.toLocaleTimeString()}</td>
            <td>${entry.machineId}</td>
            <td><span class="risk-value ${entry.riskLevel.toLowerCase()}">${entry.riskLevel}</span></td>
            <td>${(entry.failureProb * 100).toFixed(1)}%</td>
            <td>${entry.temperature.toFixed(1)}°C</td>
            <td>${entry.pressure.toFixed(1)} bar</td>
            <td>${entry.vibration.toFixed(2)} mm/s</td>
        `;
        historyBody.appendChild(row);
    });
}