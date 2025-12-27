// Main Application JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Particles.js
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: "#6366f1" },
                shape: { type: "circle" },
                opacity: { value: 0.5, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: "#6366f1",
                    opacity: 0.2,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: "none",
                    random: true,
                    straight: false,
                    out_mode: "out",
                    bounce: false
                }
            },
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: { enable: true, mode: "repulse" },
                    onclick: { enable: true, mode: "push" }
                }
            },
            retina_detect: true
        });
    }

    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = themeToggle.querySelector('i');
    
    themeToggle.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        themeIcon.className = newTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
        
        // Save preference
        localStorage.setItem('theme', newTheme);
    });

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeIcon.className = savedTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';

    // Navigation - UPDATED VERSION
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target section from data-section attribute
            const targetSection = this.getAttribute('data-section');
            
            // Hide all sections
            document.querySelectorAll('section').forEach(section => {
                section.style.display = 'none';
            });
            
            // Show target section
            const targetElement = document.getElementById(targetSection);
            if (targetElement) {
                targetElement.style.display = 'block';
                // Scroll to the section smoothly
                setTimeout(() => {
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
                
                // Load specific content based on section
                if (targetSection === 'insights') {
                    loadInsights();
                } else if (targetSection === 'history') {
                    loadHistory();
                }
            }
            
            // Update active nav link
            document.querySelectorAll('.nav-link').forEach(navLink => {
                navLink.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    // Load dynamic stats
    loadStats();
    
    // Load history on page load
    loadHistory();

    // Initialize components
    initModelSelection();
    initTextAreaActions();
    initSettings();
    initAnalysis();
    initHistory();

    // Add clear history event listener
    document.getElementById('clearHistory')?.addEventListener('click', clearAllHistory);
});

// Load history function - UPDATED VERSION
function loadHistory() {
    const historyContainer = document.getElementById('historyContainer');
    if (!historyContainer) return;
    
    // Show loading
    historyContainer.innerHTML = `
        <div class="loading-history">
            <div class="spinner-ring"></div>
            <p>Loading your analysis history...</p>
        </div>
    `;
    
    // Fetch history data
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const history = data.history || [];
            
            if (history.length === 0) {
                historyContainer.innerHTML = `
                    <div class="no-history">
                        <i class="fas fa-history"></i>
                        <h3>No Analysis History</h3>
                        <p>Run your first analysis to see it here</p>
                    </div>
                `;
                return;
            }
            
            // Build history HTML
            let historyHTML = '';
            
            history.forEach((item, index) => {
                const date = new Date(item.timestamp);
                const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                const promptPreview = item.prompt ? 
                    (item.prompt.length > 100 ? item.prompt.substring(0, 100) + '...' : item.prompt) :
                    'No prompt available';
                
                const responsePreview = item.response ?
                    (item.response.length > 150 ? item.response.substring(0, 150) + '...' : item.response) :
                    'No response available';
                
                // Get analysis metrics
                const analysis = item.analysis || {};
                const summary = analysis.summary || {};
                
                const totalIssues = summary.total_issues || 0;
                const criticalIssues = summary.critical_issues || 0;
                const overallQuality = summary.overall_quality || 'unknown';
                const errorPropagation = summary.error_propagation ? 'Yes' : 'No';
                
                historyHTML += `
                    <div class="history-item" data-id="${item.task_id || index}">
                        <div class="history-item-header">
                            <div class="history-item-title">Analysis #${history.length - index}</div>
                            <div class="history-item-date">${formattedDate}</div>
                        </div>
                        
                        <div class="history-item-preview">
                            <strong>Prompt:</strong> ${promptPreview}<br>
                            <strong>Response:</strong> ${responsePreview}
                        </div>
                        
                        <div class="history-item-metrics">
                            <div class="metric">
                                <span class="metric-value">${totalIssues}</span>
                                <span class="metric-label">Total Issues</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">${criticalIssues}</span>
                                <span class="metric-label">Critical</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">${overallQuality}</span>
                                <span class="metric-label">Quality</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">${errorPropagation}</span>
                                <span class="metric-label">Propagation</span>
                            </div>
                        </div>
                        
                        <div class="history-item-actions">
                            <button class="action-btn view-btn" onclick="viewAnalysis(${index})">
                                <i class="fas fa-eye"></i> Details
                            </button>
                            <button class="action-btn delete-btn" onclick="deleteAnalysis(${index})">
                                <i class="fas fa-trash"></i> Remove
                            </button>
                        </div>
                    </div>
                `;
            });
            
            historyContainer.innerHTML = historyHTML;
        })
        .catch(error => {
            console.error('Error loading history:', error);
            historyContainer.innerHTML = `
                <div class="no-history">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Error Loading History</h3>
                    <p>Could not load analysis history. Please try again.</p>
                </div>
            `;
        });
}

// View analysis details
function viewAnalysis(index) {
    // Switch to results section and load this analysis
    document.querySelectorAll('section').forEach(section => {
        section.style.display = 'none';
    });
    
    const resultsSection = document.getElementById('results');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
    
    // Update nav link
    document.querySelectorAll('.nav-link').forEach(navLink => {
        navLink.classList.remove('active');
        if (navLink.getAttribute('data-section') === 'results') {
            navLink.classList.add('active');
        }
    });
    
    // Fetch and display this specific analysis
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const history = data.history || [];
            if (history[index]) {
                displayAnalysisResults(history[index]);
            }
        });
}

// Helper function to display analysis results from history
function displayAnalysisResults(data) {
    // This function should display the results similar to displayResults()
    // You might need to adapt it based on your data structure
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Clear results container
    resultsContainer.innerHTML = `
        <div class="loading-results">
            <div class="spinner-ring"></div>
            <p>Loading analysis results...</p>
        </div>
    `;
    
    // Display the analysis results
    displayResults(data);
}

// Delete analysis
function deleteAnalysis(index) {
    if (!confirm('Are you sure you want to delete this analysis?')) {
        return;
    }
    
    fetch(`/api/history/${index}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Reload history
            loadHistory();
            // Reload stats
            loadStats();
            // Reload insights
            loadInsights();
        } else {
            alert('Error deleting analysis: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error deleting analysis:', error);
        alert('Error deleting analysis');
    });
}

// Clear all history
function clearAllHistory() {
    if (!confirm('Are you sure you want to clear ALL analysis history? This cannot be undone.')) {
        return;
    }
    
    fetch('/api/history/clear', {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Reload everything
            loadHistory();
            loadStats();
            loadInsights();
        } else {
            alert('Error clearing history: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error clearing history:', error);
        alert('Error clearing history');
    });
}

// Load dynamic stats - UPDATED VERSION
function loadStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Update the quick stats
            document.getElementById('avgTime').textContent = `${data.avg_analysis_time}s`;
            document.getElementById('successRate').textContent = `${data.success_rate}%`;
            document.getElementById('totalAnalyses').textContent = data.total_analyses;
        })
        .catch(error => {
            console.error('Error loading stats:', error);
            // Keep default values if API fails
            document.getElementById('avgTime').textContent = '15s';
            document.getElementById('successRate').textContent = '99%';
            document.getElementById('totalAnalyses').textContent = '0';
        });
}

// Load insights
function loadInsights() {
    const insightsContainer = document.getElementById('insightsContainer');
    if (!insightsContainer) return;
    
    // Show loading
    insightsContainer.innerHTML = `
        <div class="loading-insights">
            <div class="spinner-ring"></div>
            <p>Loading insights from your analyses...</p>
        </div>
    `;
    
    // Fetch insights data
    fetch('/api/insights')
        .then(response => response.json())
        .then(insights => {
            if (!insights || insights.length === 0) {
                insightsContainer.innerHTML = `
                    <div class="insight-card major">
                        <div class="insight-icon">
                            <i class="fas fa-chart-bar"></i>
                        </div>
                        <div class="insight-content">
                            <h3>No Insights Yet</h3>
                            <p>Run some analyses to discover patterns in LLM outputs</p>
                            <div class="insight-metric">
                                <span class="metric-value">0</span>
                                <span class="metric-label">analyses needed</span>
                            </div>
                        </div>
                    </div>
                `;
                return;
            }
            
            // Build insights HTML
            let insightsHTML = '';
            
            // Major insight (first one)
            if (insights.major) {
                insightsHTML += `
                    <div class="insight-card major">
                        <div class="insight-icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <div class="insight-content">
                            <h3>${insights.major.title}</h3>
                            <p>${insights.major.description}</p>
                            <div class="insight-metric">
                                <span class="metric-value">${insights.major.value}</span>
                                <span class="metric-label">${insights.major.label}</span>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            // Regular insights grid
            if (insights.regular && insights.regular.length > 0) {
                insightsHTML += '<div class="insight-grid">';
                insights.regular.forEach(insight => {
                    insightsHTML += `
                        <div class="insight-card">
                            <div class="insight-icon">
                                <i class="${insight.icon}"></i>
                            </div>
                            <div class="insight-content">
                                <h4>${insight.title}</h4>
                                <p>${insight.description}</p>
                                <div class="insight-metric">
                                    <span class="metric-value">${insight.value}</span>
                                    <span class="metric-label">${insight.label}</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
                insightsHTML += '</div>';
            }
            
            insightsContainer.innerHTML = insightsHTML;
        })
        .catch(error => {
            console.error('Error loading insights:', error);
            insightsContainer.innerHTML = `
                <div class="insight-card major">
                    <div class="insight-icon">
                        <i class="fas fa-exclamation-circle"></i>
                    </div>
                    <div class="insight-content">
                        <h3>Error Loading Insights</h3>
                        <p>Could not load analysis insights. Try running some analyses first.</p>
                    </div>
                </div>
            `;
        });
}

// Initialize model selection
async function initModelSelection() {
    const modelGrid = document.getElementById('modelGrid');
    
    try {
        const response = await fetch('/api/available-models');
        const data = await response.json();
        
        modelGrid.innerHTML = '';
        
        data.models.forEach(model => {
            const provider = model.includes('gemini') ? 'Google' : 
                            model.includes('grok') ? 'xAI Grok' : 'Other';
            const icon = model.includes('gemini') ? 'fab fa-google' : 
                        model.includes('grok') ? 'fas fa-brain' : 'fas fa-robot';
            
            const modelCard = document.createElement('div');
            modelCard.className = 'model-card';
            modelCard.innerHTML = `
                <div class="model-icon">
                    <i class="${icon}"></i>
                </div>
                <div class="model-info">
                    <h4>${model}</h4>
                    <p>${provider}</p>
                </div>
            `;
            
            modelCard.addEventListener('click', function() {
                document.querySelectorAll('.model-card').forEach(card => {
                    card.classList.remove('selected');
                });
                this.classList.add('selected');
            });
            
            modelGrid.appendChild(modelCard);
        });
        
        // Select first model by default
        if (modelGrid.firstChild) {
            modelGrid.firstChild.classList.add('selected');
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        modelGrid.innerHTML = '<p class="error">Failed to load models</p>';
    }
}

// Text area actions
function initTextAreaActions() {
    const promptInput = document.getElementById('promptInput');
    const clearBtn = document.getElementById('clearPrompt');
    const sampleBtn = document.getElementById('samplePrompt');
    
    clearBtn.addEventListener('click', function() {
        promptInput.value = '';
        promptInput.focus();
    });
    
    sampleBtn.addEventListener('click', function() {
        const samples = [
            "Explain quantum computing to a 10-year-old",
            "What are the main causes of climate change and how do they interact?",
            "Write a short story about a robot learning to paint",
            "Describe the process of photosynthesis in detail",
            "If all dogs are mammals and all mammals are animals, are all dogs animals?",
            "Explain the theory of relativity in simple terms",
            "Compare and contrast the economic systems of capitalism and socialism",
            "How does a computer processor work from transistors to executing programs?",
            "Discuss the ethical implications of artificial intelligence in healthcare",
            "What would life be like on Mars in 100 years?"
        ];
        
        const randomSample = samples[Math.floor(Math.random() * samples.length)];
        promptInput.value = randomSample;
    });
}

// Settings toggle
function initSettings() {
    // Settings removed - simple interface only
    console.log("Settings simplified - no temperature/token controls");
    
    // Check if settings elements exist before trying to access them
    const toggleBtn = document.getElementById('toggleSettings');
    const settingsContent = document.getElementById('settingsContent');
    
    if (!toggleBtn || !settingsContent) {
        return; // Settings section doesn't exist
    }
    
    const toggleIcon = toggleBtn.querySelector('.toggle-icon');
    let settingsVisible = false;
    
    toggleBtn.addEventListener('click', function() {
        settingsVisible = !settingsVisible;
        
        if (settingsVisible) {
            settingsContent.style.display = 'block';
            toggleIcon.className = 'fas fa-chevron-up';
        } else {
            settingsContent.style.display = 'none';
            toggleIcon.className = 'fas fa-chevron-down';
        }
    });
}

// Analysis functionality
function initAnalysis() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnLoader = document.getElementById('btnLoader');
    const promptInput = document.getElementById('promptInput');
    
    analyzeBtn.addEventListener('click', async function() {
        const prompt = promptInput.value.trim();
        const selectedModel = document.querySelector('.model-card.selected h4')?.textContent;
        
        if (!prompt) {
            showNotification('Please enter a prompt', 'error');
            return;
        }
        
        if (!selectedModel) {
            showNotification('Please select a model', 'error');
            return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        btnLoader.style.display = 'block';
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model: selectedModel
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Switch to results section
            document.querySelector('.nav-link[data-section="results"]').click();
            
            // Start polling for results
            pollResults(data.task_id);
            
            showNotification('Analysis started successfully', 'success');
            
        } catch (error) {
            console.error('Analysis error:', error);
            showNotification('Failed to start analysis: ' + error.message, 'error');
        } finally {
            analyzeBtn.disabled = false;
            btnLoader.style.display = 'none';
        }
    });
}

// Poll for analysis results
async function pollResults(taskId) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultStatus = document.getElementById('resultStatus');
    
    // Show loading state
    resultsContainer.innerHTML = `
        <div class="loading-results">
            <div class="spinner-ring"></div>
            <p>Analyzing at multiple granularity levels...</p>
            <div class="progress-steps">
                <div class="step active" id="step1">
                    <span class="step-number">1</span>
                    <span class="step-label">Token Analysis</span>
                </div>
                <div class="step" id="step2">
                    <span class="step-number">2</span>
                    <span class="step-label">Sentence Analysis</span>
                </div>
                <div class="step" id="step3">
                    <span class="step-number">3</span>
                    <span class="step-label">Reasoning Analysis</span>
                </div>
                <div class="step" id="step4">
                    <span class="step-number">4</span>
                    <span class="step-label">Cross-Granular</span>
                </div>
            </div>
        </div>
    `;
    
    const steps = ['step1', 'step2', 'step3', 'step4'];
    let currentStep = 0;
    
    // Animate steps
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            document.getElementById(steps[currentStep])?.classList.add('active');
            currentStep++;
        }
    }, 1500);
    
    // Poll for results
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'complete') {
                clearInterval(pollInterval);
                clearInterval(stepInterval);
                resultStatus.textContent = 'Complete';
                resultStatus.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                
                // Display results
                displayResults(data);
                
                // Update history and insights
                loadHistory();
                
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                clearInterval(stepInterval);
                resultStatus.textContent = 'Error';
                resultStatus.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                
                resultsContainer.innerHTML = `
                    <div class="error-state">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h3>Analysis Failed</h3>
                        <p>${data.error || 'Unknown error occurred'}</p>
                        <button onclick="location.reload()" class="retry-btn">
                            <i class="fas fa-redo"></i> Try Again
                        </button>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

// Display results - UPDATED VERSION
function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    const analysis = data.analysis;
    const summary = data.summary;
    const visualizations = data.visualizations;
    
    // Get factual analysis data
    const sentenceAnalysis = analysis.sentence_level || {};
    const factualClaims = sentenceAnalysis.factual_claims || [];
    const factVerification = sentenceAnalysis.fact_verification || {};
    const contradictions = sentenceAnalysis.contradictions || [];
    
    // Calculate real factual accuracy
    const contradictedCount = factVerification.contradicted_claims?.length || 0;
    const totalClaims = factualClaims.length || 1; // Use 1 to avoid division by zero
    const factualAccuracy = 100 * (1 - (contradictedCount / totalClaims));
    
    // Update the display
    const factualAccuracyValue = Math.max(0, Math.min(100, Math.round(factualAccuracy)));
    
    // Update factual accuracy display from visualizations
    const factualStats = data.visualizations.factual_stats || {};
    const finalFactualAccuracy = Math.round(factualStats.accuracy || factualAccuracyValue);
    
    // Extract factual issues
    const factualIssues = [];
    if (contradictedCount > 0) {
        factVerification.contradicted_claims?.forEach(claim => {
            factualIssues.push({
                description: `Contradicted claim: "${claim.simplified || claim.claim_text}"`,
                severity: 'high',
                type: 'factual_error'
            });
        });
    }
    
    if (contradictions.length > 0) {
        contradictions.forEach(contradiction => {
            factualIssues.push({
                description: `Contradiction detected between sentences ${contradiction.sentence1_idx + 1} and ${contradiction.sentence2_idx + 1}`,
                severity: 'high',
                type: 'logical_contradiction'
            });
        });
    }
    
    // Create results HTML
    resultsContainer.innerHTML = `
        <div class="results-grid">
            <!-- Summary Card -->
            <div class="summary-card">
                <div class="summary-header">
                    <h3><i class="fas fa-chart-pie"></i> Analysis Summary</h3>
                    <div class="overall-score">
                        <div class="score-circle">
                            <span class="score-value">${Math.round(visualizations.summary_scores.overall * 100)}</span>
                            <span class="score-label">Overall</span>
                        </div>
                    </div>
                </div>
                <div class="summary-metrics">
                    <div class="metric">
                        <span class="metric-label">Token Quality</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${visualizations.summary_scores.token * 100}%"></div>
                        </div>
                        <span class="metric-value">${Math.round(visualizations.summary_scores.token * 100)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sentence Coherence</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${visualizations.summary_scores.sentence * 100}%"></div>
                        </div>
                        <span class="metric-value">${Math.round(visualizations.summary_scores.sentence * 100)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Reasoning Quality</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${visualizations.summary_scores.reasoning * 100}%"></div>
                        </div>
                        <span class="metric-value">${Math.round(visualizations.summary_scores.reasoning * 100)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Factual Accuracy</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${finalFactualAccuracy}%"></div>
                        </div>
                        <span class="metric-value factual-accuracy-value">${finalFactualAccuracy}%</span>
                    </div>
                </div>
                <div class="summary-issues">
                    <h4><i class="fas fa-exclamation-circle"></i> Issues Found</h4>
                    <div class="issues-grid">
                        <div class="issue-count">
                            <span class="count">${summary.critical_issues || 0}</span>
                            <span class="label">Critical</span>
                        </div>
                        <div class="issue-count">
                            <span class="count">${visualizations.issues.token || 0}</span>
                            <span class="label">Token Level</span>
                        </div>
                        <div class="issue-count">
                            <span class="count">${visualizations.issues.sentence || 0}</span>
                            <span class="label">Sentence Level</span>
                        </div>
                        <div class="issue-count">
                            <span class="count">${visualizations.issues.reasoning || 0}</span>
                            <span class="label">Reasoning Level</span>
                        </div>
                        <div class="issue-count">
                            <span class="count" id="factualIssuesCount">${factualStats.contradicted || factualIssues.length || 0}</span>
                            <span class="label">Factual Issues</span>
                        </div>
                    </div>
                </div>
                <!-- Factual Analysis Details -->
                <div class="factual-analysis">
                    <h4><i class="fas fa-search"></i> Factual Analysis</h4>
                    <div class="factual-stats">
                        <div class="stat">
                            <span class="stat-label">Claims Found</span>
                            <span class="stat-value">${factualStats.total || totalClaims}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Verified</span>
                            <span class="stat-value success">${factVerification.verified_claims?.length || 0}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Contradicted</span>
                            <span class="stat-value error">${factualStats.contradicted || contradictedCount}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Accuracy</span>
                            <span class="stat-value">${finalFactualAccuracy}%</span>
                        </div>
                    </div>
                    ${factualIssues.length > 0 ? `
                    <div class="factual-issues">
                        <h5><i class="fas fa-exclamation-triangle"></i> Factual Issues</h5>
                        <ul>
                            ${factualIssues.map(issue => `<li class="${issue.severity}">${issue.description}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>
            </div>
            
            <!-- Radar Chart -->
            <div class="chart-card">
                <h3><i class="fas fa-radar"></i> Quality Radar</h3>
                <canvas id="radarChart"></canvas>
            </div>
            
            <!-- Issues Breakdown -->
            <div class="issues-card">
                <h3><i class="fas fa-bug"></i> Issues Breakdown</h3>
                <div class="issues-list">
                    ${generateIssuesList(analysis, factualIssues)}
                </div>
            </div>
            
            <!-- Response Preview -->
            <div class="response-card">
                <h3><i class="fas fa-comment-alt"></i> LLM Response</h3>
                <div class="response-content">
                    <p>${data.response}</p>
                </div>
                <div class="response-meta">
                    <span class="meta-item">
                        <i class="fas fa-robot"></i> ${data.model_info.provider}
                    </span>
                    <span class="meta-item">
                        <i class="fas fa-clock"></i> ${new Date(data.timestamp).toLocaleTimeString()}
                    </span>
                    <span class="meta-item">
                        <i class="fas fa-ruler-horizontal"></i> ${data.response.length} chars
                    </span>
                </div>
            </div>
        </div>
    `;
    
    // Initialize radar chart with factual accuracy
    initRadarChart(visualizations.radar, finalFactualAccuracy / 100);
}

// Generate issues list HTML
function generateIssuesList(analysis, factualIssues = []) {
    const issues = [];
    
    // Token issues
    const tokenIssues = analysis.token_level?.issues || [];
    tokenIssues.forEach(issue => {
        issues.push({
            level: 'Token',
            severity: issue.severity || 'medium',
            description: issue.description || issue.type || 'Token issue detected',
            type: issue.type || 'token_issue'
        });
    });
    
    // Sentence issues
    const sentenceIssues = analysis.sentence_level?.issues || [];
    sentenceIssues.forEach(issue => {
        issues.push({
            level: 'Sentence',
            severity: issue.severity || 'medium',
            description: issue.description || issue.type || 'Sentence issue detected',
            type: issue.type || 'sentence_issue'
        });
    });
    
    // Reasoning issues
    const reasoningIssues = analysis.reasoning_level?.issues || [];
    reasoningIssues.forEach(issue => {
        issues.push({
            level: 'Reasoning',
            severity: issue.severity || 'medium',
            description: issue.description || issue.type || 'Reasoning issue detected',
            type: issue.type || 'reasoning_issue'
        });
    });
    
    // Add factual issues
    factualIssues.forEach(issue => {
        issues.push({
            level: 'Factual',
            severity: issue.severity || 'high',
            description: issue.description,
            type: issue.type || 'factual_error'
        });
    });
    
    // Sort by severity for better display
    issues.sort((a, b) => {
        const severityOrder = { high: 0, medium: 1, low: 2, info: 3 };
        return severityOrder[a.severity] - severityOrder[b.severity];
    });
    
    // Group by type for better organization
    const groupedIssues = {};
    issues.forEach(issue => {
        if (!groupedIssues[issue.type]) {
            groupedIssues[issue.type] = [];
        }
        groupedIssues[issue.type].push(issue);
    });
    
    // Generate HTML with better organization
    if (issues.length === 0) {
        return '<div class="no-issues">No issues detected! Response appears coherent at all levels.</div>';
    }
    
    let html = '';
    
    // Show critical issues first
    const criticalIssues = issues.filter(i => i.severity === 'high');
    if (criticalIssues.length > 0) {
        html += '<div class="critical-section">';
        html += '<h4><i class="fas fa-exclamation-triangle"></i> Critical Issues</h4>';
        criticalIssues.forEach(issue => {
            html += `
                <div class="issue-item critical">
                    <div class="issue-header">
                        <span class="issue-level">${issue.level}</span>
                        <span class="issue-severity high">CRITICAL</span>
                    </div>
                    <p class="issue-desc">${issue.description}</p>
                    ${issue.type.includes('fact') ? '<span class="fact-error"><i class="fas fa-times-circle"></i> Factual Error</span>' : ''}
                </div>
            `;
        });
        html += '</div>';
    }
    
    // Show medium severity issues
    const mediumIssues = issues.filter(i => i.severity === 'medium');
    if (mediumIssues.length > 0) {
        html += '<div class="medium-section">';
        html += '<h4><i class="fas fa-exclamation-circle"></i> Moderate Issues</h4>';
        mediumIssues.forEach(issue => {
            html += `
                <div class="issue-item medium">
                    <div class="issue-header">
                        <span class="issue-level">${issue.level}</span>
                        <span class="issue-severity medium">MODERATE</span>
                    </div>
                    <p class="issue-desc">${issue.description}</p>
                </div>
            `;
        });
        html += '</div>';
    }
    
    // Show low severity issues
    const lowIssues = issues.filter(i => i.severity === 'low');
    if (lowIssues.length > 0) {
        html += '<div class="low-section">';
        html += '<h4><i class="fas fa-info-circle"></i> Minor Issues</h4>';
        lowIssues.forEach(issue => {
            html += `
                <div class="issue-item low">
                    <div class="issue-header">
                        <span class="issue-level">${issue.level}</span>
                        <span class="issue-severity low">MINOR</span>
                    </div>
                    <p class="issue-desc">${issue.description}</p>
                </div>
            `;
        });
        html += '</div>';
    }
    
    return html;
}

// Initialize radar chart with factual data
function initRadarChart(radarData, factualAccuracy = 0) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    // Get scores from existing radar data or use defaults
    const tokenScore = radarData?.datasets?.[0]?.data?.[0] || 0.8;
    const sentenceScore = radarData?.datasets?.[0]?.data?.[1] || 0.8;
    const reasoningScore = radarData?.datasets?.[0]?.data?.[2] || 0.7;
    const inferenceScore = radarData?.datasets?.[0]?.data?.[3] || 0.6;
    
    // Create updated radar data with factual accuracy
    const updatedRadarData = {
        labels: ['Token Stability', 'Sentence Coherence', 'Reasoning Quality', 'Inference Validity', 'Factual Accuracy'],
        datasets: [{
            'label': 'Quality Scores',
            'data': [
                tokenScore,
                sentenceScore,
                reasoningScore,
                inferenceScore,
                factualAccuracy || 0.75
            ],
            'fill': true,
            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
            'borderColor': 'rgb(54, 162, 235)',
            'pointBackgroundColor': 'rgb(54, 162, 235)',
            'pointBorderColor': '#fff',
            'pointHoverBackgroundColor': '#fff',
            'pointHoverBorderColor': 'rgb(54, 162, 235)'
        }]
    };
    
    new Chart(ctx, {
        type: 'radar',
        data: updatedRadarData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scale: {
                ticks: {
                    beginAtZero: true,
                    max: 1,
                    stepSize: 0.2,
                    backdropColor: 'transparent'
                },
                pointLabels: {
                    font: {
                        size: 12,
                        family: "'Poppins', sans-serif"
                    },
                    color: 'var(--text-secondary)'
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            elements: {
                line: {
                    tension: 0.3
                }
            }
        }
    });
}

// Initialize history
function initHistory() {
    // This function is now handled by loadHistory() and the DOMContentLoaded event
    console.log("History initialization handled by loadHistory()");
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
    
    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });
}

// Add notification CSS
const notificationCSS = document.createElement('style');
notificationCSS.textContent = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--surface-primary);
    border-left: 4px solid var(--accent-primary);
    padding: 1rem 1.5rem;
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    display: flex;
    align-items: center;
    gap: 1rem;
    z-index: 9999;
    animation: slideInRight 0.3s ease;
    max-width: 400px;
}

.notification.success {
    border-left-color: var(--accent-success);
}

.notification.error {
    border-left-color: var(--accent-danger);
}

.notification i {
    font-size: 1.25rem;
}

.notification.success i {
    color: var(--accent-success);
}

.notification.error i {
    color: var(--accent-danger);
}

.notification-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.25rem;
    margin-left: auto;
}

.notification-close:hover {
    color: var(--text-primary);
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.fade-out {
    animation: fadeOut 0.3s ease forwards;
}

@keyframes fadeOut {
    to {
        opacity: 0;
        transform: translateX(100%);
    }
}
`;
document.head.appendChild(notificationCSS);

// Make functions available globally
window.viewAnalysis = viewAnalysis;
window.deleteAnalysis = deleteAnalysis;
window.displayAnalysisResults = displayAnalysisResults;