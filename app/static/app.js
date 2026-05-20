/* -------------------------------------------------------------
   CARDIO-SHIELD CDSS FRONTEND CONTROLLER & CHART ORCHESTRATOR
   ------------------------------------------------------------- */

let shapChart = null;
let paretoChart = null;
let globalCostSummary = null;

// Determine backend API Base URL based on page origin.
// Enables double-clicking index.html directly from local folder.
const API_BASE = (window.location.protocol === "file:" || window.location.origin === "null") 
    ? "http://127.0.0.1:8001" 
    : "";


document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Dynamic Form Badge Value Listeners
    setupSliderListeners();

    // 2. Fetch System Multi-Model Benchmarks and Cost Statistics
    fetchSystemBenchmarks();

    // 3. Setup Predict Button Click Handler
    document.getElementById("submit-btn").addEventListener("click", runPatientDiagnosis);
});


/**
 * Syncs range slider inputs with their numeric display badges in real-time.
 */
function setupSliderListeners() {
    const sliders = [
        { id: "age", valId: "age-val" },
        { id: "trestbps", valId: "trestbps-val" },
        { id: "chol", valId: "chol-val" },
        { id: "thalach", valId: "thalach-val" },
        { id: "oldpeak", valId: "oldpeak-val" }
    ];

    sliders.forEach(slider => {
        const sliderEl = document.getElementById(slider.id);
        const badgeEl = document.getElementById(slider.valId);
        
        sliderEl.addEventListener("input", (e) => {
            badgeEl.textContent = e.target.value;
        });
    });
}


/**
 * Fetches comparative model performance and global cost statistics on page load.
 */
async function fetchSystemBenchmarks() {
    try {
        // Fetch model performance
        const metricsRes = await fetch(`${API_BASE}/api/metrics`);
        if (metricsRes.ok) {
            const metrics = await metricsRes.json();
            populateMetricsTable(metrics);
        }

        // Fetch cost savings & Pareto
        const costRes = await fetch(`${API_BASE}/api/cost-summary`);
        if (costRes.ok) {
            const summary = await costRes.json();
            globalCostSummary = summary;
            updateParetoSummary(summary);
        }
    } catch (err) {
        console.error("[FRONTEND] Error fetching system metrics:", err);
    }
}


/**
 * Renders the multi-model classification benchmark table dynamically.
 */
function populateMetricsTable(metrics) {
    const tbody = document.getElementById("metrics-table-body");
    tbody.innerHTML = "";

    metrics.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${row.Model.replace("_", " ")}</strong></td>
            <td>${(row.Accuracy * 100).toFixed(2)}%</td>
            <td>${(row.Precision * 100).toFixed(2)}%</td>
            <td>${(row.Recall * 100).toFixed(2)}%</td>
            <td>${(row['F1-score'] * 100).toFixed(2)}%</td>
            <td><span class="badge">${row['ROC-AUC'].toFixed(4)}</span></td>
        `;
        tbody.appendChild(tr);
    });
}


/**
 * Displays global cost summary data on the Pareto Frontier widget.
 */
function updateParetoSummary(summary) {
    document.getElementById("pareto-avg-cost").textContent = `$${summary.staged_simulation.avg_cost.toFixed(2)}`;
    document.getElementById("pareto-savings").textContent = `${(100.0 - (summary.staged_simulation.avg_cost / 990.0) * 100.0).toFixed(1)}%`;
    document.getElementById("pareto-accuracy").textContent = `${(summary.staged_simulation.accuracy * 100.0).toFixed(2)}% Accuracy`;
    renderParetoChart(summary);
}


/**
 * Orchestrates patient risk prediction and dynamic visual elements updates.
 */
async function runPatientDiagnosis() {
    const submitBtn = document.getElementById("submit-btn");
    
    // Add loading states
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Patient...`;

    try {
        // Compile form inputs into patient JSON object
        const patientData = {
            age: parseFloat(document.getElementById("age").value),
            sex: parseFloat(document.querySelector('input[name="sex"]:checked').value),
            cp: parseFloat(document.getElementById("cp").value),
            trestbps: parseFloat(document.getElementById("trestbps").value),
            chol: parseFloat(document.getElementById("chol").value),
            fbs: document.getElementById("fbs").checked ? 1.0 : 0.0,
            restecg: parseFloat(document.getElementById("restecg").value),
            thalach: parseFloat(document.getElementById("thalach").value),
            exang: document.getElementById("exang").checked ? 1.0 : 0.0,
            oldpeak: parseFloat(document.getElementById("oldpeak").value),
            slope: parseFloat(document.getElementById("slope").value),
            ca: parseFloat(document.getElementById("ca").value),
            thal: parseFloat(document.getElementById("thal").value)
        };

        // Post request to predict risk
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patientData)
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        
        // 1. Update Risk Probability Gauge
        updateRiskGauge(data.risk_probability, data.risk_category);
        
        // 2. Update Clinical Advice Text
        document.getElementById("advice-text").innerHTML = `
            <strong>Diagnostic Category:</strong> <span class="badge ${getCategoryColorClass(data.risk_category)}">${data.risk_category}</span><br>
            <span class="mt-1 d-block">${data.clinical_advice}</span>
        `;
        
        // 3. Update Staged Triage Statistics
        document.getElementById("stage-val-text").textContent = data.diagnostic_stage_reached;
        document.getElementById("cost-val-text").textContent = `$${data.total_diagnostic_cost.toFixed(2)}`;
        document.getElementById("savings-val-text").textContent = `${data.savings_vs_full.toFixed(1)}%`;
        
        // 4. Update Staged Stepper Visualizer
        updateStepper(data.escalation_path, data.diagnostic_stage_reached);
        
        // 5. Update Explainable AI Chart
        renderSHAPChart(data.explanations);
        
        // 6. Update Pareto Frontier Chart with Current Patient Highlight
        if (globalCostSummary) {
            renderParetoChart(globalCostSummary, { cost: data.total_diagnostic_cost });
        }

    } catch (err) {
        alert("Clinical diagnosis engine error: " + err.message);
        console.error(err);
    } finally {
        // Restore submit button state
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-shield-heart"></i> Calculate Risk Index`;
    }
}


/**
 * Computes CSS color tags for individual diagnostic categories.
 */
function getCategoryColorClass(category) {
    if (category.includes("LOW")) return "risk-low-badge";
    if (category.includes("INTERMEDIATE")) return "risk-medium-badge";
    return "risk-high-badge";
}


/**
 * Animates the circular clinical risk gauge based on the predicted probability.
 */
function updateRiskGauge(probability, category) {
    const riskPercentEl = document.getElementById("risk-percent");
    const riskBadgeEl = document.getElementById("risk-badge");
    const gaugeFill = document.getElementById("gauge-fill");
    
    const percentage = Math.round(probability * 100);
    riskPercentEl.textContent = `${percentage}%`;
    riskBadgeEl.textContent = category;
    
    // Clear dynamic class lists
    gaugeFill.classList.remove("risk-low", "risk-medium", "risk-high");
    riskBadgeEl.classList.remove("risk-low-badge", "risk-medium-badge", "risk-high-badge");
    
    // Assign corresponding colors
    if (probability < 0.35) {
        gaugeFill.classList.add("risk-low");
        riskBadgeEl.classList.add("risk-low-badge");
    } else if (probability < 0.70) {
        gaugeFill.classList.add("risk-medium");
        riskBadgeEl.classList.add("risk-medium-badge");
    } else {
        gaugeFill.classList.add("risk-high");
        riskBadgeEl.classList.add("risk-high-badge");
    }
    
    // Animate SVG Stroke Offset (283 is the max stroke dasharray circumference of r=45 circle)
    const offset = 283 - (283 * probability);
    gaugeFill.style.strokeDashoffset = offset;
}


/**
 * Renders Staged Stepper highlights to show the escalation path.
 */
function updateStepper(escalationPath, stageReached) {
    const stepperContainer = document.getElementById("stepper-container");
    stepperContainer.innerHTML = ""; // Clear existing steps
    
    const allStages = ['STAGE 1', 'STAGE 2', 'STAGE 3', 'STAGE 4'];
    const activeStageIndex = allStages.indexOf(stageReached);
    
    allStages.forEach((stage, idx) => {
        const stepDiv = document.createElement("div");
        stepDiv.classList.add("step");
        
        let iconClass = "fa-user-nurse";
        if (idx === 1) iconClass = "fa-hospital-user";
        if (idx === 2) iconClass = "fa-wave-square";
        if (idx === 3) iconClass = "fa-stethoscope";
        
        // Mark past steps as complete, final step as active, rest as disabled
        if (idx < activeStageIndex) {
            stepDiv.classList.add("complete");
        } else if (idx === activeStageIndex) {
            stepDiv.classList.add("active");
        } else {
            stepDiv.classList.add("disabled");
        }
        
        stepDiv.innerHTML = `
            <div class="step-icon"><i class="fa-solid ${iconClass}"></i></div>
            <span class="step-label">${stage}</span>
        `;
        
        stepperContainer.appendChild(stepDiv);
        
        // Append connector line if not the final stage
        if (idx < 3) {
            const connector = document.createElement("div");
            connector.classList.add("step-connector");
            if (idx < activeStageIndex) {
                connector.classList.add("complete");
            }
            stepperContainer.appendChild(connector);
        }
    });
}


/**
 * Renders the interactive local SHAP explanation bar chart via Chart.js.
 */
function renderSHAPChart(explanations) {
    const ctx = document.getElementById('shap-chart').getContext('2d');
    
    // Sort features for nice chart layout: highest absolute value first
    const labels = explanations.map(x => `${x.feature} (${x.raw_value.toFixed(1)})`);
    const values = explanations.map(x => x.risk_impact);
    
    // Assign custom clinical colors: red for risk-increasing, blue for protective features
    const colors = values.map(val => val >= 0 ? 'rgba(231, 29, 54, 0.75)' : 'rgba(72, 202, 228, 0.75)');
    const borderColors = values.map(val => val >= 0 ? '#E71D36' : '#48CAE4');
    
    if (shapChart) {
        shapChart.destroy(); // Destroy previous instance to re-render fresh
    }
    
    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Individual Risk Impact Index',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y', // Horizontal bars
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            return `Impact: ${val >= 0 ? '+' : ''}${val.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' },
                    title: {
                        display: true,
                        text: 'Risk Impact (Decreases Risk <-- 0 --> Increases Risk)',
                        color: '#9CA3AF'
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#F3F4F6', font: { weight: 'bold' } }
                }
            }
        }
    });
}


/**
 * Renders the interactive Pareto Frontier scatter and line chart via Chart.js.
 */
function renderParetoChart(summary, currentPatient = null) {
    const ctx = document.getElementById('pareto-chart').getContext('2d');
    
    // Prepare data points for standard variants
    const curvePoints = [
        { x: summary.variants.B.cost, y: summary.variants.B.auc },
        { x: summary.variants.D.cost, y: summary.variants.D.auc },
        { x: summary.variants.C.cost, y: summary.variants.C.auc },
        { x: summary.variants.A.cost, y: summary.variants.A.auc }
    ];
    
    // Sort curve points by cost (x) to draw the connecting line properly
    curvePoints.sort((a, b) => a.x - b.x);
    
    const datasets = [
        // Dataset 0: Connecting dashed line
        {
            label: 'Pareto Frontier Line',
            data: curvePoints,
            type: 'line',
            borderColor: 'rgba(52, 211, 153, 0.3)', // light emerald-500 equivalent
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            showLine: true,
            pointRadius: 0,
            order: 3
        },
        // Dataset 1: Standard Tiers / Variants
        {
            label: 'Standard Tiers',
            data: [
                { x: summary.variants.B.cost, y: summary.variants.B.auc, label: 'Variant B: Low-Cost' },
                { x: summary.variants.D.cost, y: summary.variants.D.auc, label: 'Variant D: Optimized' },
                { x: summary.variants.C.cost, y: summary.variants.C.auc, label: 'Variant C: Immediate' },
                { x: summary.variants.A.cost, y: summary.variants.A.auc, label: 'Variant A: Full-Feature' }
            ],
            type: 'scatter',
            backgroundColor: ['#e76f51', '#2a9d8f', '#f4a261', '#3b82f6'],
            pointRadius: 7,
            pointHoverRadius: 9,
            order: 2
        },
        // Dataset 2: Staged Escalation Average
        {
            label: 'Staged Escalation (Avg)',
            data: [{ x: summary.staged_simulation.avg_cost, y: summary.staged_simulation.auc }],
            type: 'scatter',
            backgroundColor: '#e9c46a',
            pointStyle: 'star',
            pointRadius: 13,
            pointHoverRadius: 15,
            order: 1
        }
    ];

    // Dataset 3: Current Patient Triage Marker (if active)
    if (currentPatient) {
        let patientY = summary.variants.A.auc; // default to Stage 4 AUC
        if (currentPatient.cost <= 35.0) {
            patientY = summary.variants.B.auc;
        } else if (currentPatient.cost <= 120.0) {
            patientY = 0.955;
        } else if (currentPatient.cost <= 400.0) {
            patientY = summary.variants.C.auc;
        }
        
        datasets.push({
            label: 'Current Patient Location',
            data: [{ x: currentPatient.cost, y: patientY }],
            type: 'scatter',
            backgroundColor: '#00F0FF', // Neon Cyan
            borderColor: '#FFFFFF',
            borderWidth: 2,
            pointRadius: 11,
            pointHoverRadius: 13,
            pointStyle: 'rectRot',
            order: 0
        });
    }

    if (paretoChart) {
        paretoChart.destroy();
    }

    paretoChart = new Chart(ctx, {
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const p = context.raw;
                            if (context.dataset.label === 'Current Patient Location') {
                                return `Current Patient Triage: Cost $${p.x}, Stage AUC ${p.y.toFixed(4)}`;
                            }
                            if (context.dataset.label === 'Staged Escalation (Avg)') {
                                return `Staged Escalation (Avg): Cost $${p.x.toFixed(2)}, AUC ${p.y.toFixed(4)}`;
                            }
                            if (p.label) {
                                return `${p.label}: Cost $${p.x}, AUC ${p.y.toFixed(4)}`;
                            }
                            return `Cost $${p.x}, AUC ${p.y.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' },
                    title: {
                        display: true,
                        text: 'Diagnostic Cost (USD)',
                        color: '#9CA3AF',
                        font: { size: 10 }
                    }
                },
                y: {
                    min: 0.75,
                    max: 1.02,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' },
                    title: {
                        display: true,
                        text: 'Diagnostic ROC-AUC',
                        color: '#9CA3AF',
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}
