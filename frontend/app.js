/**
 * Breast Cancer Detection System - Frontend JavaScript
 * Handles form validation, API communication, and result display
 */

// ==================== Configuration ====================
const CONFIG = {
    // Update this URL when deploying to production
    API_BASE_URL: '',
    // For production, use your deployed URL:
    // API_BASE_URL: 'https://your-app.onrender.com',
};

// ==================== DOM Elements ====================
const predictionForm = document.getElementById('predictionForm');
const resultsPanel = document.getElementById('resultsPanel');
const submitBtn = document.getElementById('submitBtn');
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');

// Slider elements
const lumpSizeSlider = document.getElementById('lumpSize');
const lumpSizeValue = document.getElementById('lumpSizeValue');
const lumpHardnessSlider = document.getElementById('lumpHardness');
const lumpHardnessValue = document.getElementById('lumpHardnessValue');

// Toggle elements
const toggleInputs = document.querySelectorAll('.toggle input');

// ==================== Event Listeners ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeSliders();
    initializeToggles();
    checkAPIHealth();
});

predictionForm.addEventListener('submit', handleFormSubmit);
predictionForm.addEventListener('reset', handleFormReset);

// ==================== Initialization Functions ====================
function initializeSliders() {
    // Lump Size Slider
    lumpSizeSlider.addEventListener('input', (e) => {
        lumpSizeValue.textContent = `${e.target.value} cm`;
    });
    
    // Lump Hardness Slider
    lumpHardnessSlider.addEventListener('input', (e) => {
        lumpHardnessValue.textContent = `${e.target.value}%`;
    });
}

function initializeToggles() {
    toggleInputs.forEach(toggle => {
        const toggleText = toggle.parentElement.querySelector('.toggle-text');
        
        toggle.addEventListener('change', () => {
            toggleText.textContent = toggle.checked ? 'Yes' : 'No';
        });
    });
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status !== 'healthy') {
            console.warn('API is running but model may not be loaded:', data);
        } else {
            console.log('API health check passed:', data);
        }
    } catch (error) {
        console.warn('API health check failed. Make sure the backend is running:', error);
    }
}

// ==================== Form Handling ====================
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    
    // Collect form data
    const formData = collectFormData();
    
    try {
        // Send prediction request
        const result = await sendPredictionRequest(formData);
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to get prediction. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

function handleFormReset() {
    // Reset slider displays
    lumpSizeValue.textContent = '1.5 cm';
    lumpHardnessValue.textContent = '50%';
    
    // Reset toggle texts
    document.querySelectorAll('.toggle-text').forEach(text => {
        text.textContent = 'No';
    });
    
    // Hide results
    resultsPanel.style.display = 'none';
}

function validateForm() {
    const lumpShape = document.getElementById('lumpShape').value;
    const lumpTexture = document.getElementById('lumpTexture').value;
    const growthRate = document.getElementById('growthRate').value;
    const patientAge = document.getElementById('patientAge').value;
    
    if (!lumpShape) {
        showError('Please select the lump shape.');
        return false;
    }
    
    if (!lumpTexture) {
        showError('Please select the lump texture.');
        return false;
    }
    
    if (!growthRate) {
        showError('Please select the growth rate.');
        return false;
    }
    
    if (!patientAge || patientAge < 18 || patientAge > 120) {
        showError('Please enter a valid age (18-120).');
        return false;
    }
    
    return true;
}

function collectFormData() {
    return {
        lump_size: parseFloat(document.getElementById('lumpSize').value),
        lump_shape: document.getElementById('lumpShape').value,
        lump_texture: document.getElementById('lumpTexture').value,
        lump_hardness: parseFloat(document.getElementById('lumpHardness').value),
        growth_rate: document.getElementById('growthRate').value,
        pain_present: document.getElementById('painPresent').checked,
        skin_changes: document.getElementById('skinChanges').checked,
        nipple_discharge: document.getElementById('nippleDischarge').checked,
        family_history: document.getElementById('familyHistory').checked,
        patient_age: parseInt(document.getElementById('patientAge').value)
    };
}

// ==================== API Communication ====================
async function sendPredictionRequest(formData) {
    const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

// ==================== Results Display ====================
function displayResults(result) {
    // Show results panel
    resultsPanel.style.display = 'block';
    
    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Update classification
    const classificationIcon = document.getElementById('classificationIcon');
    const predictionLabel = document.getElementById('predictionLabel');
    const riskLevel = document.getElementById('riskLevel');
    
    const isBenign = result.prediction === 'Benign';
    
    classificationIcon.className = `classification-icon ${isBenign ? 'benign' : 'malignant'}`;
    classificationIcon.textContent = isBenign ? '✓' : '⚠';
    
    predictionLabel.textContent = result.prediction;
    predictionLabel.className = isBenign ? 'benign' : 'malignant';
    
    riskLevel.textContent = `Risk Level: ${result.risk_level}`;
    
    // Update confidence
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidencePercent = (result.confidence * 100).toFixed(1);
    
    setTimeout(() => {
        confidenceFill.style.width = `${confidencePercent}%`;
    }, 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    
    // Update probability bars
    const probBenignFill = document.getElementById('probBenignFill');
    const probBenignValue = document.getElementById('probBenignValue');
    const probMalignantFill = document.getElementById('probMalignantFill');
    const probMalignantValue = document.getElementById('probMalignantValue');
    
    const benignPercent = (result.probability_benign * 100).toFixed(1);
    const malignantPercent = (result.probability_malignant * 100).toFixed(1);
    
    setTimeout(() => {
        probBenignFill.style.width = `${benignPercent}%`;
        probMalignantFill.style.width = `${malignantPercent}%`;
    }, 100);
    
    probBenignValue.textContent = `${benignPercent}%`;
    probMalignantValue.textContent = `${malignantPercent}%`;
    
    // Update recommendations
    updateRecommendations(result);
    
    // Update result card border color
    const resultCard = document.getElementById('resultCard');
    resultCard.style.borderColor = isBenign ? 'var(--benign-color)' : 'var(--malignant-color)';
}

function updateRecommendations(result) {
    const recommendationsList = document.getElementById('recommendationsList');
    const recommendations = getRecommendations(result);
    
    recommendationsList.innerHTML = recommendations
        .map(rec => `<li>${rec}</li>`)
        .join('');
}

function getRecommendations(result) {
    const recommendations = [];
    
    if (result.prediction === 'Benign' && result.risk_level === 'Low') {
        recommendations.push(
            'Continue regular self-examinations monthly',
            'Maintain routine mammogram schedule (as recommended for your age)',
            'Monitor for any changes in the lump characteristics',
            'Consult your healthcare provider if you notice any changes'
        );
    } else if (result.prediction === 'Benign' && result.risk_level === 'Medium') {
        recommendations.push(
            'Schedule a follow-up with your healthcare provider within 2-4 weeks',
            'Consider additional imaging tests (ultrasound or MRI)',
            'Document any changes in symptoms',
            'Continue monthly self-examinations'
        );
    } else if (result.prediction === 'Malignant' || result.risk_level === 'High') {
        recommendations.push(
            '⚠️ URGENT: Schedule an appointment with a breast specialist immediately',
            'Request a biopsy to confirm diagnosis',
            'Do not delay seeking professional medical evaluation',
            'Bring this assessment report to your appointment',
            'Consider getting a second opinion from another specialist'
        );
    }
    
    // Add general recommendations
    recommendations.push(
        'Remember: This is an AI-assisted screening tool, not a diagnosis',
        'Only a qualified healthcare professional can provide a definitive diagnosis'
    );
    
    return recommendations;
}

// ==================== UI State Functions ====================
function setLoadingState(isLoading) {
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    
    submitBtn.disabled = isLoading;
    btnText.style.display = isLoading ? 'none' : 'inline';
    btnLoader.style.display = isLoading ? 'inline-flex' : 'none';
}

// ==================== Modal Functions ====================
function showError(message) {
    errorMessage.textContent = message;
    errorModal.style.display = 'flex';
}

function closeErrorModal() {
    errorModal.style.display = 'none';
}

// Close modal on outside click
errorModal.addEventListener('click', (e) => {
    if (e.target === errorModal) {
        closeErrorModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && errorModal.style.display === 'flex') {
        closeErrorModal();
    }
});

// ==================== Utility Functions ====================
function resetAssessment() {
    predictionForm.reset();
    handleFormReset();
    
    // Scroll back to form
    predictionForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function printResults() {
    window.print();
}

// ==================== Demo Mode (for testing without backend) ====================
async function demoPredict(formData) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate mock prediction based on inputs
    let riskScore = 0;
    
    // Calculate risk based on inputs
    if (formData.lump_size > 3) riskScore += 20;
    if (formData.lump_shape === 'irregular') riskScore += 25;
    if (formData.lump_texture === 'rough') riskScore += 15;
    if (formData.lump_hardness > 70) riskScore += 15;
    if (formData.growth_rate === 'fast') riskScore += 20;
    if (formData.skin_changes) riskScore += 15;
    if (formData.nipple_discharge) riskScore += 10;
    if (formData.family_history) riskScore += 15;
    if (formData.patient_age > 50) riskScore += 10;
    
    const probMalignant = Math.min(riskScore / 100, 0.95);
    const probBenign = 1 - probMalignant;
    
    const prediction = probBenign > 0.5 ? 'Benign' : 'Malignant';
    const confidence = Math.max(probBenign, probMalignant);
    
    let riskLevel;
    if (probMalignant < 0.3) riskLevel = 'Low';
    else if (probMalignant < 0.6) riskLevel = 'Medium';
    else riskLevel = 'High';
    
    return {
        prediction,
        confidence: parseFloat(confidence.toFixed(4)),
        probability_benign: parseFloat(probBenign.toFixed(4)),
        probability_malignant: parseFloat(probMalignant.toFixed(4)),
        risk_level: riskLevel
    };
}

// Enable demo mode by uncommenting the line below
// CONFIG.DEMO_MODE = true;

// Override sendPredictionRequest for demo mode
const originalSendPredictionRequest = sendPredictionRequest;
async function sendPredictionRequestWithDemo(formData) {
    if (CONFIG.DEMO_MODE) {
        console.log('Demo mode: Using mock prediction');
        return demoPredict(formData);
    }
    return originalSendPredictionRequest(formData);
}

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CONFIG,
        collectFormData,
        validateForm,
        displayResults
    };
}
