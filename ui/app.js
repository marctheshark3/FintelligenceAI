// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global variable to store research context
let currentResearchContext = {
    originalQuery: '',
    results: null,
    summary: '',
    extractedFields: {
        useCase: '',
        complexityLevel: '',
        requirements: [],
        constraints: []
    }
};

// Global variables for job tracking
let activeIngestionJobs = {}; // Store job objects, not interval IDs
let jobIntervals = {}; // Store interval IDs separately

// Tab switching
function switchTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => tab.classList.remove('active'));

    // Remove active class from all tabs
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Show selected tab content
    document.getElementById(tabName).classList.add('active');

    // Add active class to selected tab
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Auto-load stats when knowledge tab is opened
    if (tabName === 'knowledge') {
        loadKnowledgeStats();
        // Sync jobs first, then load them
        syncJobsWithBackend().then(() => loadActiveJobs());
        // Update category options
        updateCategoryOptions();
        // Start auto-refresh for knowledge tab
        startKnowledgeStatsAutoRefresh();
        startActiveJobsAutoRefresh();
    } else {
        // Stop auto-refresh when leaving knowledge tab
        stopKnowledgeStatsAutoRefresh();
        stopActiveJobsAutoRefresh();
    }

    // Update category options for research and code generation tabs
    if (tabName === 'research' || tabName === 'generate') {
        updateCategoryOptions();
    }
}

// Auto-refresh functionality for knowledge stats
let knowledgeStatsInterval = null;

function startKnowledgeStatsAutoRefresh() {
    // Clear any existing interval
    stopKnowledgeStatsAutoRefresh();

    // Refresh every 30 seconds when on knowledge tab
    knowledgeStatsInterval = setInterval(() => {
        loadKnowledgeStats();
        loadActiveJobs();
    }, 30000);
}

function stopKnowledgeStatsAutoRefresh() {
    if (knowledgeStatsInterval) {
        clearInterval(knowledgeStatsInterval);
        knowledgeStatsInterval = null;
    }
}

let activeJobsRefreshInterval = null;

function startActiveJobsAutoRefresh() {
    // Clear any existing interval
    stopActiveJobsAutoRefresh();

    // Refresh active jobs every 3 seconds, sync with backend every 30 seconds
    activeJobsRefreshInterval = setInterval(() => {
        const now = Date.now();
        if (!window.lastJobSync || now - window.lastJobSync > 30000) {
            syncJobsWithBackend().then(() => loadActiveJobs());
            window.lastJobSync = now;
        } else {
            loadActiveJobs();
        }
    }, 3000);
}

function stopActiveJobsAutoRefresh() {
    if (activeJobsRefreshInterval) {
        clearInterval(activeJobsRefreshInterval);
        activeJobsRefreshInterval = null;
    }
}

// API Status Check
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const statusElement = document.getElementById('apiStatus');

        if (response.ok) {
            statusElement.className = 'api-status online';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> API Online';
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        const statusElement = document.getElementById('apiStatus');
        statusElement.className = 'api-status offline';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> API Offline';
    }
}

// List item management
function addListItem(event, listType) {
    if (event.key === 'Enter' && event.target.value.trim()) {
        event.preventDefault();
        const value = event.target.value.trim();
        event.target.value = '';

        const item = document.createElement('div');
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.gap = '10px';
        item.innerHTML = `
            <input type="text" value="${value}" readonly style="flex: 1; background: #f8f9fa;">
            <button type="button" onclick="this.parentElement.remove()" style="padding: 5px 10px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer;">
                <i class="fas fa-times"></i>
            </button>
        `;

        const container = document.getElementById(`${listType}List`);
        container.insertBefore(item, container.lastElementChild);
    }
}

function addRequirement() {
    const input = document.querySelector('#requirementsList input');
    if (input.value.trim()) {
        addListItem({key: 'Enter', target: input, preventDefault: () => {}}, 'requirements');
    }
}

function addConstraint() {
    const input = document.querySelector('#constraintsList input');
    if (input.value.trim()) {
        addListItem({key: 'Enter', target: input, preventDefault: () => {}}, 'constraints');
    }
}

// Function to extract fields from research context
function extractFieldsFromResearch(researchData, originalQuery) {
    const extracted = {
        useCase: '',
        complexityLevel: 'intermediate',
        requirements: [],
        constraints: []
    };

    const queryLower = originalQuery.toLowerCase();
    const summaryLower = researchData.toLowerCase();

    // Extract use case based on keywords
    const useCaseKeywords = {
        'web_app': ['web application', 'web app', 'website', 'frontend', 'backend', 'full stack', 'react', 'vue', 'angular', 'express', 'django', 'flask'],
        'api': ['api', 'rest', 'graphql', 'endpoint', 'microservice', 'service', 'backend'],
        'data_processing': ['data processing', 'etl', 'data pipeline', 'analytics', 'big data', 'pandas', 'numpy'],
        'automation': ['automation', 'script', 'cron', 'scheduler', 'workflow', 'pipeline'],
        'blockchain': ['blockchain', 'smart contract', 'ethereum', 'solidity', 'web3', 'defi', 'crypto', 'ergo', 'ergoscript'],
        'ml_ai': ['machine learning', 'artificial intelligence', 'ai', 'ml', 'tensorflow', 'pytorch', 'model', 'neural network'],
        'database': ['database', 'sql', 'mongodb', 'postgresql', 'mysql', 'orm', 'migration'],
        'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin'],
        'testing': ['test', 'testing', 'unit test', 'integration test', 'jest', 'pytest', 'selenium']
    };

    for (const [useCase, keywords] of Object.entries(useCaseKeywords)) {
        if (keywords.some(keyword => queryLower.includes(keyword) || summaryLower.includes(keyword))) {
            extracted.useCase = useCase;
            break;
        }
    }

    // Extract complexity level
    if (queryLower.includes('simple') || queryLower.includes('basic') || queryLower.includes('beginner')) {
        extracted.complexityLevel = 'beginner';
    } else if (queryLower.includes('advanced') || queryLower.includes('complex') || queryLower.includes('enterprise')) {
        extracted.complexityLevel = 'advanced';
    }

    // Extract requirements from research content
    const requirementIndicators = [
        'must have', 'should include', 'requires', 'needs to', 'essential',
        'important to', 'necessary', 'feature', 'functionality'
    ];

    const lines = summaryLower.split('\n');
    lines.forEach(line => {
        if (requirementIndicators.some(indicator => line.includes(indicator))) {
            const cleaned = line.replace(/[^\w\s]/g, '').trim();
            if (cleaned.length > 10 && cleaned.length < 100) {
                extracted.requirements.push(cleaned);
            }
        }
    });

    // Extract constraints
    const constraintIndicators = [
        'limitation', 'constraint', 'cannot', 'should not', 'avoid',
        'performance', 'memory', 'security', 'compatibility'
    ];

    lines.forEach(line => {
        if (constraintIndicators.some(indicator => line.includes(indicator))) {
            const cleaned = line.replace(/[^\w\s]/g, '').trim();
            if (cleaned.length > 10 && cleaned.length < 100) {
                extracted.constraints.push(cleaned);
            }
        }
    });

    // Limit to reasonable numbers
    extracted.requirements = extracted.requirements.slice(0, 5);
    extracted.constraints = extracted.constraints.slice(0, 3);

    return extracted;
}

// Function to populate form fields with extracted data
function populateFormFields(extractedFields) {
    // Set use case
    if (extractedFields.useCase) {
        document.getElementById('useCase').value = extractedFields.useCase;
    }

    // Set complexity level
    document.getElementById('complexity').value = extractedFields.complexityLevel;

    // Clear existing requirements and constraints
    document.querySelectorAll('#requirementsList div').forEach(div => {
        if (div.querySelector('input[readonly]')) {
            div.remove();
        }
    });
    document.querySelectorAll('#constraintsList div').forEach(div => {
        if (div.querySelector('input[readonly]')) {
            div.remove();
        }
    });

    // Add requirements
    extractedFields.requirements.forEach(req => {
        const input = document.querySelector('#requirementsList input:not([readonly])');
        input.value = req;
        addRequirement();
    });

    // Add constraints
    extractedFields.constraints.forEach(constraint => {
        const input = document.querySelector('#constraintsList input:not([readonly])');
        input.value = constraint;
        addConstraint();
    });
}

// Form submission handlers
document.addEventListener('DOMContentLoaded', () => {
    // Code Generation Form
    document.getElementById('generateForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(e.target);
        const requirements = Array.from(document.querySelectorAll('#requirementsList input[readonly]'))
            .map(input => input.value).filter(v => v.trim());
        const constraints = Array.from(document.querySelectorAll('#constraintsList input[readonly]'))
            .map(input => input.value).filter(v => v.trim());

        const data = {
            description: formData.get('description'),
            use_case: formData.get('use_case') || null,
            complexity_level: formData.get('complexity_level'),
            requirements: requirements,
            constraints: constraints
        };

        await submitRequest('/agents/generate-code', data, 'generateLoading', 'generateResponse', 'generateContent', renderCodeResponse);
    });

    // Research Form
    document.getElementById('researchForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(e.target);
        const data = {
            query: formData.get('query'),
            scope: formData.get('scope'),
            include_examples: formData.get('include_examples') === 'true'
        };

        // Store original query for context
        currentResearchContext.originalQuery = data.query;

        await submitRequest('/agents/research', data, 'researchLoading', 'researchResponse', 'researchContent', renderResearchResponse);
    });

    // Validation Form
    document.getElementById('validateForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(e.target);
        const validationCriteria = {
            syntax_check: document.getElementById('syntaxCheck').checked,
            semantic_check: document.getElementById('semanticCheck').checked,
            security_check: document.getElementById('securityCheck').checked,
            gas_estimation: document.getElementById('gasEstimation').checked
        };

        const data = {
            code: formData.get('code'),
            use_case: formData.get('use_case') || null,
            validation_criteria: validationCriteria
        };

        await submitRequest('/agents/validate-code', data, 'validateLoading', 'validateResponse', 'validateContent', renderValidateResponse);
    });

    // Initialize API status checking
    checkApiStatus();
    setInterval(checkApiStatus, 30000); // Check every 30 seconds
});

// Generic request handler
async function submitRequest(endpoint, data, loadingId, responseId, contentId, renderFunction) {
    const loadingElement = document.getElementById(loadingId);
    const responseElement = document.getElementById(responseId);
    const contentElement = document.getElementById(contentId);

    // Show loading
    loadingElement.classList.add('show');
    responseElement.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Hide loading
        loadingElement.classList.remove('show');

        if (response.ok) {
            responseElement.style.display = 'block';
            responseElement.className = 'response-section success';
            renderFunction(result, contentElement);
        } else {
            responseElement.style.display = 'block';
            responseElement.className = 'response-section error';
            contentElement.innerHTML = `
                <h4>Error</h4>
                <p>${result.detail || 'An error occurred while processing your request.'}</p>
            `;
        }
    } catch (error) {
        // Hide loading
        loadingElement.classList.remove('show');

        responseElement.style.display = 'block';
        responseElement.className = 'response-section error';
        contentElement.innerHTML = `
            <h4>Connection Error</h4>
            <p>Failed to connect to the API. Please check if the server is running.</p>
            <p><strong>Error:</strong> ${error.message}</p>
        `;
    }
}

// Response renderers
function renderCodeResponse(data, container) {
    container.innerHTML = `
        <div class="code-block">
            <pre><code class="language-scala">${escapeHtml(data.generated_code)}</code></pre>
        </div>

        <div class="response-grid">
            <div class="response-item">
                <h4><i class="fas fa-info-circle"></i> Explanation</h4>
                <p>${escapeHtml(data.explanation || 'No explanation provided.')}</p>
            </div>

            <div class="response-item">
                <h4><i class="fas fa-lightbulb"></i> Use Case</h4>
                <p><strong>${escapeHtml(data.use_case || 'General')}</strong></p>
            </div>

            <div class="response-item">
                <h4><i class="fas fa-clock"></i> Generation Time</h4>
                <p><strong>${data.generation_time || 'N/A'}</strong></p>
            </div>

            <div class="response-item">
                <h4><i class="fas fa-check-circle"></i> Confidence</h4>
                <p><strong>${data.confidence_score ? (data.confidence_score * 100).toFixed(1) + '%' : 'N/A'}</strong></p>
            </div>
        </div>

        ${data.documentation_links && data.documentation_links.length > 0 ? `
        <div class="response-item" style="margin-top: 15px;">
            <h4><i class="fas fa-link"></i> Documentation Links</h4>
            <ul>
                ${data.documentation_links.map(link => `<li><a href="${escapeHtml(link)}" target="_blank">${escapeHtml(link)}</a></li>`).join('')}
            </ul>
        </div>
        ` : ''}
    `;

    // Re-run Prism highlighting
    if (typeof Prism !== 'undefined') {
        Prism.highlightAllUnder(container);
    }
}

function renderResearchResponse(data, container) {
    let content = '';

    // Handle different response formats
    if (typeof data === 'string') {
        content = data;
    } else if (data.summary) {
        content = data.summary;
    } else if (data.response) {
        content = data.response;
    } else {
        content = JSON.stringify(data, null, 2);
    }

    // Store research context for potential code generation workflow
    currentResearchContext = {
        originalQuery: data.topic || 'Research Query',
        findings: data.findings || content,
        sources: data.sources || [],
        recommendations: data.recommendations || '',
        rawResearch: data.raw_research || {}
    };

    container.innerHTML = `
        <div class="research-workflow-container">
            <div class="response-item">
                <h4><i class="fas fa-search"></i> Research Results</h4>
                <div style="white-space: pre-wrap; line-height: 1.6;">${escapeHtml(content)}</div>
            </div>

            ${data.examples ? `
            <div class="response-item" style="margin-top: 15px;">
                <h4><i class="fas fa-code"></i> Code Examples</h4>
                <div class="code-block">
                    <pre><code class="language-scala">${escapeHtml(data.examples)}</code></pre>
                </div>
            </div>
            ` : ''}

            ${data.sources && Array.isArray(data.sources) ? `
            <div class="response-item" style="margin-top: 15px;">
                <h4><i class="fas fa-link"></i> Sources</h4>
                <ul>
                    ${data.sources.map(source => `<li>${escapeHtml(source)}</li>`).join('')}
                </ul>
            </div>
            ` : ''}

            <!-- Summary Workflow Section -->
            <div class="workflow-section" style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px; border-left: 4px solid #667eea;">
                <h4><i class="fas fa-magic"></i> Research Summary & Code Generation</h4>
                <p style="margin-bottom: 15px; color: #6c757d;">Generate a focused summary and use it to create code based on your research with auto-populated fields.</p>

                <div class="workflow-buttons">
                    <button class="btn workflow-btn" onclick="generateResearchSummary()" id="summaryBtn">
                        <i class="fas fa-compress-alt"></i> Generate Summary
                    </button>
                    <button class="btn workflow-btn" onclick="generateCodeFromResearch()" id="generateCodeBtn" style="display: none;">
                        <i class="fas fa-code"></i> Generate Code from Research
                    </button>
                </div>

                <div id="summaryContainer" style="display: none; margin-top: 15px;">
                    <div class="loading" id="summaryLoading" style="margin: 10px 0;">
                        <div class="spinner"></div>
                        <p>Generating research summary...</p>
                    </div>
                    <div id="summaryResult" style="display: none;">
                        <h5><i class="fas fa-file-alt"></i> Research Summary</h5>
                        <div id="summaryContent" style="background: white; padding: 15px; border-radius: 4px; border: 1px solid #dee2e6;"></div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Re-run Prism highlighting
    if (typeof Prism !== 'undefined') {
        Prism.highlightAllUnder(container);
    }
}

// Research summary generation function
async function generateResearchSummary() {
    if (!currentResearchContext) {
        alert('No research context available. Please run a research query first.');
        return;
    }

    const summaryBtn = document.getElementById('summaryBtn');
    const summaryContainer = document.getElementById('summaryContainer');
    const summaryLoading = document.getElementById('summaryLoading');
    const summaryResult = document.getElementById('summaryResult');
    const summaryContent = document.getElementById('summaryContent');
    const generateCodeBtn = document.getElementById('generateCodeBtn');

    // Show container and loading
    summaryContainer.style.display = 'block';
    summaryLoading.style.display = 'block';
    summaryResult.style.display = 'none';
    summaryBtn.disabled = true;
    summaryBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

    try {
        const summaryData = {
            query: `Create a focused summary for code generation based on this research about: ${currentResearchContext.originalQuery}`,
            research_context: {
                findings: currentResearchContext.findings,
                sources: currentResearchContext.sources,
                recommendations: currentResearchContext.recommendations
            },
            scope: "focused_summary",
            include_examples: true
        };

        const response = await fetch(`${API_BASE_URL}/agents/research-summary`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(summaryData)
        });

        let result;
        if (response.ok) {
            result = await response.json();
        } else {
            // Fallback: use the research agent with a summary-focused query
            const fallbackData = {
                query: `Summarize the key points for code generation: ${currentResearchContext.findings.substring(0, 500)}`,
                scope: "focused",
                include_examples: true
            };

            const fallbackResponse = await fetch(`${API_BASE_URL}/agents/research`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(fallbackData)
            });

            if (fallbackResponse.ok) {
                result = await fallbackResponse.json();
            } else {
                throw new Error('Failed to generate summary');
            }
        }

        // Hide loading, show result
        summaryLoading.style.display = 'none';
        summaryResult.style.display = 'block';

        // Display summary
        let summaryText = '';
        if (result.findings) {
            summaryText = result.findings;
        } else if (result.summary) {
            summaryText = result.summary;
        } else if (typeof result === 'string') {
            summaryText = result;
        } else {
            summaryText = JSON.stringify(result, null, 2);
        }

        summaryContent.innerHTML = `<div style="white-space: pre-wrap; line-height: 1.6;">${escapeHtml(summaryText)}</div>`;

        // Store summary in research context
        currentResearchContext.summary = summaryText;

        // Extract fields from research data
        currentResearchContext.extractedFields = extractFieldsFromResearch(
            summaryText,
            currentResearchContext.originalQuery
        );

        // Show generate code button
        generateCodeBtn.style.display = 'inline-flex';

    } catch (error) {
        summaryLoading.style.display = 'none';
        summaryResult.style.display = 'block';
        summaryContent.innerHTML = `<div style="color: #dc3545;"><i class="fas fa-exclamation-triangle"></i> Error generating summary: ${error.message}</div>`;
    } finally {
        summaryBtn.disabled = false;
        summaryBtn.innerHTML = '<i class="fas fa-compress-alt"></i> Generate Summary';
    }
}

// Function to navigate to code generation with research context
function generateCodeFromResearch() {
    if (!currentResearchContext || !currentResearchContext.summary) {
        alert('Please generate a research summary first.');
        return;
    }

    // Switch to generate code tab
    switchTab('generate');

    // Pre-fill the description with research context
    const descriptionField = document.getElementById('description');
    const researchSummary = currentResearchContext.summary;
    const originalQuery = currentResearchContext.originalQuery;

    const enhancedDescription = `Based on research about "${originalQuery}":

RESEARCH SUMMARY:
${researchSummary}

IMPLEMENTATION REQUEST:
Please create code that implements the concepts and patterns described in the research summary above.`;

    descriptionField.value = enhancedDescription;

    // Auto-populate form fields based on extracted data
    if (currentResearchContext.extractedFields) {
        populateFormFields(currentResearchContext.extractedFields);
    }

    // Scroll to top of generate section
    document.getElementById('generate').scrollIntoView({ behavior: 'smooth' });

    // Add visual indicator
    descriptionField.style.borderColor = '#667eea';
    descriptionField.style.boxShadow = '0 0 0 0.2rem rgba(102, 126, 234, 0.25)';

    // Remove visual indicator after a few seconds
    setTimeout(() => {
        descriptionField.style.borderColor = '';
        descriptionField.style.boxShadow = '';
    }, 3000);

    // Show success message with more details
    const extractedCount = (currentResearchContext.extractedFields?.requirements?.length || 0) +
                          (currentResearchContext.extractedFields?.constraints?.length || 0);
    const message = `Research context loaded! ${extractedCount > 0 ? `Auto-populated ${extractedCount} field(s).` : ''}`;
    showWorkflowNotification(message, 'success');
}

// Utility function to show workflow notifications
function showWorkflowNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#d1ecf1'};
        color: ${type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#0c5460'};
        border: 1px solid ${type === 'success' ? '#c3e6cb' : type === 'error' ? '#f5c6cb' : '#bee5eb'};
        border-radius: 8px;
        padding: 15px 20px;
        max-width: 300px;
        z-index: 1000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    `;

    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
        ${message}
    `;

    // Add animation styles
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

    document.body.appendChild(notification);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
            if (style.parentNode) {
                style.parentNode.removeChild(style);
            }
        }, 300);
    }, 4000);
}

function renderValidateResponse(data, container) {
    container.innerHTML = `
        <div class="response-grid">
            <div class="response-item">
                <h4><i class="fas fa-check-circle"></i> Validation Status</h4>
                <p><strong>${data.success ? 'PASSED' : 'FAILED'}</strong></p>
            </div>

            ${data.error_message ? `
            <div class="response-item">
                <h4><i class="fas fa-exclamation-triangle"></i> Error Message</h4>
                <p style="color: #c53030;">${escapeHtml(data.error_message)}</p>
            </div>
            ` : ''}

            ${data.execution_time_seconds ? `
            <div class="response-item">
                <h4><i class="fas fa-clock"></i> Execution Time</h4>
                <p><strong>${data.execution_time_seconds}s</strong></p>
            </div>
            ` : ''}
        </div>

        ${data.result ? `
        <div class="response-item" style="margin-top: 15px;">
            <h4><i class="fas fa-list-alt"></i> Detailed Results</h4>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 4px; white-space: pre-wrap; overflow-x: auto;">${JSON.stringify(data.result, null, 2)}</pre>
        </div>
        ` : ''}
    `;
}

// Agent status check
async function checkAgentStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/agents/status`);
        const loadingElement = document.getElementById('statusLoading');
        const responseElement = document.getElementById('statusResponse');
        const contentElement = document.getElementById('statusContent');

        loadingElement.classList.add('show');
        responseElement.style.display = 'none';

        const result = await response.json();

        loadingElement.classList.remove('show');

        if (response.ok) {
            responseElement.style.display = 'block';
            responseElement.className = 'response-section success';
            renderStatusResponse(result, contentElement);
        } else {
            responseElement.style.display = 'block';
            responseElement.className = 'response-section error';
            contentElement.innerHTML = `
                <h4>Error</h4>
                <p>Failed to fetch agent status: ${result.detail || 'Unknown error'}</p>
            `;
        }
    } catch (error) {
        const loadingElement = document.getElementById('statusLoading');
        const responseElement = document.getElementById('statusResponse');
        const contentElement = document.getElementById('statusContent');

        loadingElement.classList.remove('show');
        responseElement.style.display = 'block';
        responseElement.className = 'response-section error';
        contentElement.innerHTML = `
            <h4>Connection Error</h4>
            <p>Failed to connect to the API: ${error.message}</p>
        `;
    }
}

function renderStatusResponse(data, container) {
    container.innerHTML = `
        <div class="response-item">
            <h4><i class="fas fa-robot"></i> Orchestrator</h4>
            <p><strong>Status:</strong> ${data.orchestrator.status}</p>
            <p><strong>Active Tasks:</strong> ${data.orchestrator.active_tasks}</p>
            <p><strong>Completed Tasks:</strong> ${data.orchestrator.completed_tasks}</p>
        </div>

        <div style="margin-top: 20px;">
            <h4><i class="fas fa-cogs"></i> Individual Agents</h4>
            <div class="response-grid">
                ${Object.entries(data.agents).map(([key, agent]) => `
                    <div class="response-item">
                        <h4>${agent.name}</h4>
                        <p><strong>Role:</strong> ${agent.role}</p>
                        <p><strong>Status:</strong> ${agent.status}</p>
                        <p><strong>Active:</strong> ${agent.active_tasks}</p>
                        <p><strong>Completed:</strong> ${agent.completed_tasks}</p>
                        <p><strong>Supports:</strong> ${agent.supported_task_types.join(', ')}</p>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// Utility functions
function escapeHtml(text) {
    if (typeof text !== 'string') {
        text = String(text);
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Sample data helper functions
function fillSampleCodeGeneration() {
    document.getElementById('description').value = 'Create a REST API with authentication middleware, user management endpoints, and data validation for a task management application.';
    document.getElementById('useCase').value = 'api';
    document.getElementById('complexity').value = 'intermediate';

    // Add sample requirements
    const requirementsInput = document.querySelector('#requirementsList input');
    requirementsInput.value = 'JWT-based authentication';
    addRequirement();
    requirementsInput.value = 'CRUD operations for tasks';
    addRequirement();
    requirementsInput.value = 'Input validation and sanitization';
    addRequirement();

    // Add sample constraints
    const constraintsInput = document.querySelector('#constraintsList input');
    constraintsInput.value = 'Follow RESTful principles';
    addConstraint();
    constraintsInput.value = 'Handle errors gracefully';
    addConstraint();
}

function fillSampleResearch() {
    document.getElementById('researchQuery').value = 'What are the best practices for building secure REST APIs with authentication and authorization?';
    document.getElementById('researchScope').value = 'web';
    document.getElementById('includeExamples').value = 'true';
}

function fillSampleValidation() {
    document.getElementById('validateCode').value = `from flask import Flask, request, jsonify
from functools import wraps
import jwt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/tasks', methods=['GET'])
@token_required
def get_tasks():
    return jsonify({'tasks': []})`;
    document.getElementById('validateUseCase').value = 'api';
}

// Add sample data buttons
document.addEventListener('DOMContentLoaded', () => {
    // Add sample buttons to each form
    const generateForm = document.getElementById('generateForm');
    const sampleGenBtn = document.createElement('button');
    sampleGenBtn.type = 'button';
    sampleGenBtn.className = 'btn add-item';
    sampleGenBtn.innerHTML = '<i class="fas fa-file-alt"></i> Fill Sample Data';
    sampleGenBtn.onclick = fillSampleCodeGeneration;
    generateForm.appendChild(sampleGenBtn);

    const researchForm = document.getElementById('researchForm');
    const sampleResBtn = document.createElement('button');
    sampleResBtn.type = 'button';
    sampleResBtn.className = 'btn add-item';
    sampleResBtn.innerHTML = '<i class="fas fa-file-alt"></i> Fill Sample Query';
    sampleResBtn.onclick = fillSampleResearch;
    researchForm.appendChild(sampleResBtn);

    const validateForm = document.getElementById('validateForm');
    const sampleValBtn = document.createElement('button');
    sampleValBtn.type = 'button';
    sampleValBtn.className = 'btn add-item';
    sampleValBtn.innerHTML = '<i class="fas fa-file-alt"></i> Fill Sample Code';
    sampleValBtn.onclick = fillSampleValidation;
    validateForm.appendChild(sampleValBtn);
});

// Knowledge Base Management Functions

// Load knowledge base statistics
async function loadKnowledgeStats() {
    const contentElement = document.getElementById('knowledgeStatsContent');

    // Show loading indicator
    contentElement.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
            <div class="spinner" style="margin-right: 10px;"></div>
            <span>Loading knowledge base statistics...</span>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/knowledge/stats`);

        if (response.ok) {
            const stats = await response.json();
            console.log('Knowledge stats received:', stats); // Debug log

            // Ensure arrays exist and are valid
            const availableCategories = stats.available_categories || [];
            const collections = stats.collections || {};

            // Get current timestamp for last refreshed
            const now = new Date();
            const timestamp = now.toLocaleTimeString();

            contentElement.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 0.9rem; color: #666;">
                        <i class="fas fa-clock"></i> Last refreshed: ${timestamp}
                    </span>
                    <button onclick="loadKnowledgeStats()" class="btn add-item" style="padding: 6px 12px; font-size: 0.8rem;">
                        <i class="fas fa-refresh"></i> Refresh Now
                    </button>
                </div>

                <div class="response-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                    <div class="response-item">
                        <h4><i class="fas fa-file-alt"></i> Total Documents</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">${stats.total_documents || 0}</p>
                    </div>
                    <div class="response-item">
                        <h4><i class="fas fa-cubes"></i> Total Chunks</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">${stats.total_chunks || 0}</p>
                    </div>
                    <div class="response-item">
                        <h4><i class="fas fa-hdd"></i> Storage Size</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">${(stats.storage_size_mb || 0).toFixed(1)} MB</p>
                    </div>
                    <div class="response-item">
                        <h4><i class="fas fa-clock"></i> Last Updated</h4>
                        <p style="font-size: 0.9rem;">${stats.last_updated || 'Never'}</p>
                    </div>
                </div>

                ${availableCategories.length > 0 ? `
                <div class="response-item" style="margin-top: 15px;">
                    <h4><i class="fas fa-tags"></i> Available Categories</h4>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        ${availableCategories.map(cat =>
                            `<span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8rem;">${cat}</span>`
                        ).join('')}
                    </div>
                </div>
                ` : ''}

                ${Object.keys(collections).length > 0 ? `
                <div class="response-item" style="margin-top: 15px;">
                    <h4><i class="fas fa-database"></i> Collections</h4>
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${Object.entries(collections).map(([name, info]) =>
                            `<div style="margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                                <strong>${name}</strong>: ${info.count || 0} items
                            </div>`
                        ).join('')}
                    </div>
                </div>
                ` : ''}
            `;
        } else {
            const errorText = await response.text();
            console.error('Failed to load stats:', response.status, errorText);
            contentElement.innerHTML = `
                <div class="response-item error">
                    <h4>Error Loading Stats</h4>
                    <p>Failed to load knowledge base statistics (${response.status})</p>
                    <small>${errorText}</small>
                </div>
            `;
        }
    } catch (error) {
        console.error('Knowledge stats error:', error);
        contentElement.innerHTML = `
            <div class="response-item error">
                <h4>Connection Error</h4>
                <p>Failed to connect to the API: ${error.message}</p>
                <small>Check if the server is running on ${API_BASE_URL}</small>
            </div>
        `;
    }
}

// Handle file upload
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }

    // Handle upload type switching
    const uploadTypeRadios = document.querySelectorAll('input[name="uploadType"]');
    if (uploadTypeRadios.length > 0) {
        uploadTypeRadios.forEach(radio => {
            radio.addEventListener('change', handleUploadTypeChange);
        });
    }

    // Setup file upload area interactions
    setupFileUploadArea();
});

// Setup file upload area with drag and drop
function setupFileUploadArea() {
    const uploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileUpload');

    if (!uploadArea || !fileInput) return;

    // Click to select files
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change handler
    fileInput.addEventListener('change', handleFileSelectionChange);

    // Drag and drop handlers
    uploadArea.addEventListener('dragenter', handleDragEnter);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
}

// Handle upload type change
function handleUploadTypeChange(event) {
    const uploadType = event.target.value;
    const fileInput = document.getElementById('fileUpload');
    const helperText = document.getElementById('uploadHelperText');
    const uploadText = document.getElementById('uploadText');
    const uploadIcon = document.getElementById('uploadIcon');
    const structureOption = document.getElementById('structureOption');

    if (uploadType === 'directory') {
        // Enable directory upload
        fileInput.setAttribute('webkitdirectory', '');
        fileInput.removeAttribute('multiple');
        helperText.textContent = 'Select a directory to upload all its files and subdirectories';
        uploadText.textContent = 'Click to select a directory or drag & drop folder';
        uploadIcon.innerHTML = '<i class="fas fa-folder"></i>';
        structureOption.style.display = 'block';
    } else {
        // Enable file upload
        fileInput.removeAttribute('webkitdirectory');
        fileInput.setAttribute('multiple', '');
        helperText.textContent = 'Supported formats: TXT, MD, PDF, DOCX, PY, JS, JSON, RST, TEX';
        uploadText.textContent = 'Click to select multiple files or drag & drop';
        uploadIcon.innerHTML = '<i class="fas fa-cloud-upload-alt"></i>';
        structureOption.style.display = 'none';
    }

    // Clear current selection
    fileInput.value = '';
    clearFilePreview();
}

// Handle file selection change
function handleFileSelectionChange(event) {
    const files = event.target.files;
    updateFilePreview(files);
}

// Drag and drop handlers
function handleDragEnter(event) {
    event.preventDefault();
    event.stopPropagation();
    const uploadArea = document.getElementById('fileUploadArea');
    uploadArea.style.borderColor = '#4CAF50';
    uploadArea.style.backgroundColor = '#f0f8f0';
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    const uploadArea = document.getElementById('fileUploadArea');
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.backgroundColor = '#f8f9fa';
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();

    const uploadArea = document.getElementById('fileUploadArea');
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.backgroundColor = '#f8f9fa';

    const files = event.dataTransfer.files;
    const fileInput = document.getElementById('fileUpload');

    // Update file input with dropped files
    fileInput.files = files;
    updateFilePreview(files);
}

// Update file preview
function updateFilePreview(files) {
    const previewList = document.getElementById('filePreviewList');
    const previewContent = document.getElementById('filePreviewContent');

    if (!files || files.length === 0) {
        previewList.style.display = 'none';
        return;
    }

    previewList.style.display = 'block';

    let html = '';
    const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);

    // Add summary
    html += `<div style="margin-bottom: 15px; padding: 10px; background: #e3f2fd; border-radius: 6px; border-left: 4px solid #2196f3;">
        <strong>${files.length} file${files.length > 1 ? 's' : ''} selected</strong>
        (Total: ${formatFileSize(totalSize)})
    </div>`;

    // Add individual files
    Array.from(files).forEach((file, index) => {
        const fileIcon = getFileIcon(file.name);
        const relativePathDisplay = file.webkitRelativePath || file.name;

        html += `
            <div style="display: flex; align-items: center; padding: 8px; margin: 4px 0; background: #f8f9fa; border-radius: 4px; border: 1px solid #e9ecef;">
                <span style="font-size: 1.2rem; margin-right: 10px; color: #666;">${fileIcon}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 500;">${relativePathDisplay}</div>
                    <small style="color: #666;">${formatFileSize(file.size)} • ${file.type || 'Unknown type'}</small>
                </div>
            </div>
        `;
    });

    previewContent.innerHTML = html;
}

// Clear file preview
function clearFilePreview() {
    const previewList = document.getElementById('filePreviewList');
    previewList.style.display = 'none';
}

// Clear file selection
function clearFileSelection() {
    const fileInput = document.getElementById('fileUpload');
    fileInput.value = '';
    clearFilePreview();
}

// Get file icon based on extension
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const iconMap = {
        'md': '📝',
        'txt': '📄',
        'pdf': '📕',
        'docx': '📘',
        'doc': '📘',
        'py': '🐍',
        'js': '📜',
        'json': '📋',
        'html': '🌐',
        'css': '🎨',
        'rst': '📝',
        'tex': '📄'
    };
    return iconMap[ext] || '📄';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function handleFileUpload(event) {
    event.preventDefault();

    const fileInput = document.getElementById('fileUpload');
    const categorySelect = document.getElementById('fileCategory');
    const preserveStructureCheckbox = document.getElementById('preserveStructure');
    const progressDiv = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('uploadProgressBar');
    const statusElement = document.getElementById('uploadStatus');
    const resultsDiv = document.getElementById('uploadResults');

    const files = fileInput.files;
    if (files.length === 0) {
        alert('Please select at least one file or directory to upload.');
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    progressBar.style.width = '0%';
    statusElement.textContent = 'Starting upload...';

    try {
        // Prepare form data for bulk upload
        const formData = new FormData();

        // Add all files
        for (let file of files) {
            formData.append('files', file);
        }

        // Add other parameters
        formData.append('category', categorySelect.value);
        formData.append('preserve_structure', preserveStructureCheckbox.checked);

        statusElement.textContent = `Uploading ${files.length} files...`;
        progressBar.style.width = '30%';

        const response = await fetch(`${API_BASE_URL}/knowledge/upload-files`, {
            method: 'POST',
            body: formData
        });

        progressBar.style.width = '90%';
        statusElement.textContent = 'Processing upload results...';

        const uploadedFiles = await response.json();

        // Update progress
        progressBar.style.width = '100%';
        statusElement.textContent = `Upload completed! Processing ${uploadedFiles.length} files...`;
        resultsDiv.style.display = 'block';

        // Filter successful uploads and errors
        const successfulFiles = uploadedFiles.filter(file => file.success);
        const failedFiles = uploadedFiles.filter(file => !file.success);

        if (successfulFiles.length > 0) {
            // Show success message with clear next steps
            document.getElementById('uploadResults').innerHTML = `
                <div class="response-item success" style="margin-bottom: 15px;">
                    <h4>✅ Upload Complete</h4>
                    <p><strong>Successfully uploaded ${successfulFiles.length} files!</strong></p>
                    <div style="background: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #28a745;">
                        <p style="margin: 0; font-weight: bold; color: #155724;">📋 Next Step: Click "Process Uploaded Files" to add these documents to your knowledge base.</p>
                    </div>
                </div>

                <div style="max-height: 200px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 8px; background: white;">
                    <div style="background: #f8f9fa; padding: 10px; border-bottom: 1px solid #dee2e6; font-weight: bold;">
                        📁 Uploaded Files (${successfulFiles.length} total)
                    </div>
                    ${successfulFiles.map((file, index) => `
                        <div style="padding: 12px; border-bottom: 1px solid #f0f0f0; display: flex; align-items: center; gap: 10px;">
                            <span style="background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold;">✓</span>
                            <div style="flex: 1;">
                                <div style="font-weight: 500; color: #333;">${file.filename}</div>
                                <div style="font-size: 0.8rem; color: #666;">
                                    ${(file.size_bytes / 1024).toFixed(1)} KB • ${getFileIcon(file.filename)} ${file.filename.split('.').pop().toUpperCase()}
                                </div>
                                <div style="font-size: 0.7rem; color: #999;">Path: ${file.file_path}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        if (failedFiles.length > 0) {
            // Show error messages
            const errorHtml = `
                <div class="response-item error" style="margin-top: 15px;">
                    <h4>⚠️ Upload Issues (${failedFiles.length} files)</h4>
                    ${failedFiles.map(file => `<p style="margin: 5px 0;">• ${file.filename}: ${file.message || 'Upload failed'}</p>`).join('')}
                </div>
            `;
            document.getElementById('uploadResults').innerHTML += errorHtml;
        }

        // Reset form and hide progress
        if (successfulFiles.length > 0) {
            setTimeout(() => {
                document.getElementById('uploadForm').reset();
                clearFilePreview();
                progressDiv.style.display = 'none';
            }, 2000);
        }

    } catch (error) {
        progressBar.style.width = '100%';
        statusElement.textContent = 'Upload failed!';
        resultsDiv.style.display = 'block';
        resultsDiv.innerHTML = `
            <div class="response-section error">
                <h4><i class="fas fa-exclamation-triangle"></i> Upload Failed</h4>
                <p>${error.message}</p>
            </div>
        `;
    }
}


// Ingestion functions
async function ingestErgoScript() {
    const jobId = await startIngestion({
        source_type: 'ergoscript'
    });

    if (jobId) {
        showIngestionProgress('Ingesting ErgoScript knowledge base...');
        monitorIngestionJob(jobId);
    }
}

async function ingestUploadedFiles() {
    // Get list of uploaded files from the knowledge-base directory
    const fileList = await getUploadedFilesList();

    if (fileList.length === 0) {
        alert('No files found to process. Please upload some files first.');
        return;
    }

    const jobId = await startIngestion({
        source_type: 'files',
        file_paths: fileList
    });

    if (jobId) {
        showIngestionProgress('Processing uploaded files...');
        monitorIngestionJob(jobId);
    }
}

// Refresh knowledge base
async function refreshKnowledgeBase() {
    try {
        showIngestionProgress('🔄 Starting knowledge base refresh...');

        const response = await fetch(`${API_BASE_URL}/knowledge/refresh`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const result = await response.json();
            const jobId = result.job_id;

            showIngestionProgress('📊 Refresh initiated successfully! Monitoring progress...');

            // Monitor the refresh job
            await monitorIngestionJob(jobId);

            // Auto-reload stats after successful refresh
            setTimeout(() => {
                loadKnowledgeStats();
            }, 2000);

        } else {
            const error = await response.json();
            updateIngestionProgress({
                status: 'failed',
                message: `❌ Refresh failed: ${error.detail || 'Unknown error'}`,
                progress: 0
            });
        }
    } catch (error) {
        updateIngestionProgress({
            status: 'failed',
            message: `❌ Network error during refresh: ${error.message}`,
            progress: 0
        });
    }
}

async function startIngestion(request) {
    try {
        const response = await fetch(`${API_BASE_URL}/knowledge/ingest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });

        const result = await response.json();

        if (response.ok) {
            // Store the initial job object
            activeIngestionJobs[result.job_id] = result;
            // Update the active jobs display immediately
            await loadActiveJobs();
            return result.job_id;
        } else {
            throw new Error(result.detail || 'Failed to start ingestion');
        }
    } catch (error) {
        alert(`Failed to start ingestion: ${error.message}`);
        return null;
    }
}

function showIngestionProgress(message) {
    const statusDiv = document.getElementById('ingestionStatus');
    const progressBar = document.getElementById('ingestionProgressBar');
    const messageElement = document.getElementById('ingestionMessage');

    statusDiv.style.display = 'block';
    progressBar.style.width = '0%';
    messageElement.textContent = message;
}

async function monitorIngestionJob(jobId) {
    let consecutiveErrors = 0;
    const maxConsecutiveErrors = 5; // Allow up to 5 consecutive errors before giving up

    const checkInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/knowledge/ingest/${jobId}`);

            if (response.ok) {
                const job = await response.json();

                // Reset error counter on successful response
                consecutiveErrors = 0;

                // Update the job object in our tracking
                activeIngestionJobs[jobId] = job;

                // Update the progress display
                updateIngestionProgress(job);

                // Update the active jobs display
                await loadActiveJobs();

                if (job.status === 'completed' || job.status === 'failed') {
                    clearInterval(checkInterval);
                    delete jobIntervals[jobId];

                    // Keep the job in activeIngestionJobs for a bit longer to show completion
                    setTimeout(() => {
                        delete activeIngestionJobs[jobId];
                        loadActiveJobs(); // Refresh the display
                    }, 5000); // Remove after 5 seconds

                    // Refresh stats after completion with improved timing
                    if (job.status === 'completed') {
                        console.log('Ingestion completed, refreshing stats...');

                        // Single immediate refresh
                        await loadKnowledgeStats();

                        // One additional refresh after a short delay to ensure all operations are complete
                        setTimeout(async () => {
                            console.log('Final stats refresh after ingestion completion');
                            await loadKnowledgeStats();
                        }, 3000); // 3 second delay instead of multiple rapid refreshes
                    }
                }
            } else {
                consecutiveErrors++;
                console.warn(`Failed to get job status (attempt ${consecutiveErrors}/${maxConsecutiveErrors}):`, response.statusText);

                // Update job status to show connection issues
                if (activeIngestionJobs[jobId]) {
                    activeIngestionJobs[jobId].message = `Connection issues (attempt ${consecutiveErrors}/${maxConsecutiveErrors})...`;
                    await loadActiveJobs();
                }

                // Only remove job after multiple consecutive failures
                if (consecutiveErrors >= maxConsecutiveErrors) {
                    console.error(`Job ${jobId} monitoring failed after ${maxConsecutiveErrors} consecutive errors`);
                    clearInterval(checkInterval);
                    delete jobIntervals[jobId];

                    // Mark job as having connection issues instead of deleting it
                    if (activeIngestionJobs[jobId]) {
                        activeIngestionJobs[jobId].status = 'connection_error';
                        activeIngestionJobs[jobId].message = 'Lost connection to job - may still be running';
                        await loadActiveJobs();
                    }
                }
            }
        } catch (error) {
            consecutiveErrors++;
            console.warn(`Error monitoring ingestion job (attempt ${consecutiveErrors}/${maxConsecutiveErrors}):`, error);

            // Update job status to show connection issues
            if (activeIngestionJobs[jobId]) {
                activeIngestionJobs[jobId].message = `Network error (attempt ${consecutiveErrors}/${maxConsecutiveErrors})...`;
                await loadActiveJobs();
            }

            // Only remove job after multiple consecutive failures
            if (consecutiveErrors >= maxConsecutiveErrors) {
                console.error(`Job ${jobId} monitoring failed after ${maxConsecutiveErrors} consecutive network errors`);
                clearInterval(checkInterval);
                delete jobIntervals[jobId];

                // Mark job as having connection issues instead of deleting it
                if (activeIngestionJobs[jobId]) {
                    activeIngestionJobs[jobId].status = 'connection_error';
                    activeIngestionJobs[jobId].message = 'Network error - job may still be running';
                    await loadActiveJobs();
                }
            }
        }
    }, 2000); // Check every 2 seconds

    // Store the interval ID separately
    jobIntervals[jobId] = checkInterval;
}

function updateIngestionProgress(job) {
    const progressElement = document.getElementById('ingestionProgressBar');
    const statusElement = document.getElementById('ingestionStatus');
    const messageElement = document.getElementById('ingestionMessage');

    if (!progressElement || !statusElement || !messageElement) return;

    // Update progress bar
    const progressPercent = Math.round(job.progress * 100);
    progressElement.style.width = `${progressPercent}%`;

    // Update status with better messaging
    let statusClass = '';
    let statusIcon = '';
    let headerIcon = '';
    let statusMessage = job.message || 'Processing...';

    switch (job.status) {
        case 'pending':
            statusClass = 'status-pending';
            statusIcon = '⏳';
            headerIcon = '<i class="fas fa-clock"></i>';
            break;
        case 'running':
            statusClass = 'status-running';
            statusIcon = '🔄';
            headerIcon = '<i class="fas fa-spinner fa-spin"></i>';
            // Add more specific messaging based on progress
            if (job.progress < 0.3) {
                statusMessage = job.message || 'Initializing ingestion process...';
            } else if (job.progress < 0.6) {
                statusMessage = job.message || 'Processing documents...';
            } else if (job.progress < 0.9) {
                statusMessage = job.message || 'Creating embeddings and storing in vector database...';
            } else if (job.progress < 1.0) {
                statusMessage = job.message || 'Finalizing database operations...';
            }
            break;
        case 'completed':
            statusClass = 'status-completed';
            statusIcon = '✅';
            headerIcon = '<i class="fas fa-check-circle"></i>';
            statusMessage = job.message || 'Ingestion completed successfully!';
            // Auto-hide after 10 seconds
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 10000);
            break;
        case 'failed':
            statusClass = 'status-failed';
            statusIcon = '❌';
            headerIcon = '<i class="fas fa-exclamation-circle"></i>';
            statusMessage = job.message || 'Ingestion failed';
            break;
        default:
            statusClass = 'status-unknown';
            statusIcon = '❓';
            headerIcon = '<i class="fas fa-question-circle"></i>';
    }

    // Update the header with appropriate icon
    const headerElement = statusElement.querySelector('h4');
    if (headerElement) {
        headerElement.innerHTML = `${headerIcon} Ingestion Progress`;
    }

    // Update the message
    messageElement.innerHTML = `
        <div class="ingestion-status ${statusClass}">
            <span class="status-icon">${statusIcon}</span>
            <span class="status-text">${statusMessage}</span>
            <span class="status-progress">${progressPercent}%</span>
        </div>
    `;

    // Show completion results if available
    if (job.status === 'completed' && job.result) {
        const result = job.result;
        const completionElement = document.getElementById('ingestionComplete');
        if (completionElement) {
            completionElement.innerHTML = `
                <div class="completion-stats">
                    <div class="stat-item">
                        <span class="stat-value">${result.documents_processed || 0}</span>
                        <span class="stat-label">Documents Processed</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${result.chunks_created || 0}</span>
                        <span class="stat-label">Chunks Created</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${result.chunks_stored || 0}</span>
                        <span class="stat-label">Chunks Stored</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${result.processing_time_seconds ? result.processing_time_seconds.toFixed(1) : '0.0'}s</span>
                        <span class="stat-label">Processing Time</span>
                    </div>
                </div>
            `;
        }
    }
}

// Get list of uploaded files (simplified - in real implementation, this would call an API)
async function getUploadedFilesList() {
    // This is a simplified implementation
    // In a real scenario, you'd call an API to get the list of files in the knowledge-base directory
    return [
        'knowledge-base/documents/uploaded_file_1.txt',
        'knowledge-base/documents/uploaded_file_2.md'
    ];
}

// Sync jobs with backend to catch any orphaned jobs
async function syncJobsWithBackend() {
    try {
        const response = await fetch(`${API_BASE_URL}/knowledge/ingest`);
        if (response.ok) {
            const backendJobs = await response.json();

            // Add any backend jobs that aren't being tracked locally
            for (const job of backendJobs) {
                if (!activeIngestionJobs[job.job_id]) {
                    console.log(`Found orphaned job: ${job.job_id}, adding to tracking`);
                    activeIngestionJobs[job.job_id] = job;

                    // Start monitoring if job is still active
                    if (job.status === 'pending' || job.status === 'running') {
                        monitorIngestionJob(job.job_id);
                    }
                }
            }

            // Remove any local jobs that no longer exist on backend
            for (const jobId of Object.keys(activeIngestionJobs)) {
                const backendJob = backendJobs.find(j => j.job_id === jobId);
                if (!backendJob) {
                    console.log(`Local job ${jobId} no longer exists on backend, removing`);
                    delete activeIngestionJobs[jobId];
                    if (jobIntervals[jobId]) {
                        clearInterval(jobIntervals[jobId]);
                        delete jobIntervals[jobId];
                    }
                }
            }
        }
    } catch (error) {
        console.warn('Failed to sync jobs with backend:', error);
    }
}

// Load available categories
async function loadAvailableCategories() {
    try {
        const response = await fetch(`${API_BASE_URL}/knowledge/categories`);
        if (response.ok) {
            const result = await response.json();
            return result.categories || [];
        }
    } catch (error) {
        console.warn('Failed to load categories:', error);
    }
    return [];
}

// Update category datalist with available categories
async function updateCategoryOptions() {
    const categories = await loadAvailableCategories();
    const datalistIds = ['categoryOptions', 'researchCategoryOptions', 'codeCategoryOptions'];

    datalistIds.forEach(datalistId => {
        const datalist = document.getElementById(datalistId);

        if (datalist && categories.length > 0) {
            // Keep existing options and add new ones
            const existingOptions = Array.from(datalist.querySelectorAll('option')).map(opt => opt.value);

            categories.forEach(category => {
                if (!existingOptions.includes(category)) {
                    const option = document.createElement('option');
                    option.value = category;
                    datalist.appendChild(option);
                }
            });
        }
    });
}

// Load active jobs
async function loadActiveJobs() {
    const activeJobsDiv = document.getElementById('activeJobs');

    if (!activeJobsDiv) return;

    // First sync with backend to catch orphaned jobs
    await syncJobsWithBackend();

    const jobCount = Object.keys(activeIngestionJobs).length;

    if (jobCount === 0) {
        activeJobsDiv.innerHTML = '<p style="color: #666; font-style: italic; margin: 0;">No active jobs</p>';
        return;
    }

    const jobsHtml = Object.entries(activeIngestionJobs).map(([jobId, job]) => {
        const progressPercent = Math.round((job.progress || 0) * 100);

        // Status styling
        let statusColor = '#6c757d';
        let statusBg = '#f8f9fa';
        switch (job.status) {
            case 'pending':
                statusColor = '#856404';
                statusBg = '#fff3cd';
                break;
            case 'running':
                statusColor = '#004085';
                statusBg = '#cce7ff';
                break;
            case 'completed':
                statusColor = '#155724';
                statusBg = '#d4edda';
                break;
            case 'failed':
                statusColor = '#721c24';
                statusBg = '#f8d7da';
                break;
            case 'connection_error':
                statusColor = '#856404';
                statusBg = '#fff3cd';
                break;
        }

        return `
            <div class="job-item" style="
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; font-size: 0.9rem; color: #333; margin-bottom: 4px;">
                            Job ${jobId.substring(0, 8)}...
                        </div>
                        <div style="font-size: 0.8rem; color: #666; line-height: 1.3;">
                            ${job.message || 'Processing...'}
                        </div>
                    </div>
                    <div style="text-align: right; margin-left: 12px;">
                        <span style="
                            padding: 3px 8px;
                            background: ${statusBg};
                            color: ${statusColor};
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 500;
                            text-transform: uppercase;
                        ">
                            ${job.status || 'unknown'}
                        </span>
                    </div>
                </div>
                <div style="margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 0.75rem; color: #666;">Progress</span>
                        <span style="font-size: 0.75rem; font-weight: 600; color: #333;">${progressPercent}%</span>
                    </div>
                    <div style="
                        width: 100%;
                        height: 6px;
                        background: #e9ecef;
                        border-radius: 3px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: ${progressPercent}%;
                            height: 100%;
                            background: linear-gradient(90deg, #007bff, #0056b3);
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                    ${job.status === 'connection_error' ? `
                        <div style="margin-top: 8px;">
                            <button onclick="retryJobMonitoring('${jobId}')" style="
                                background: #ffc107;
                                color: #212529;
                                border: none;
                                border-radius: 4px;
                                padding: 4px 8px;
                                font-size: 0.75rem;
                                cursor: pointer;
                            ">
                                🔄 Retry Connection
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');

    activeJobsDiv.innerHTML = `
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.9rem; font-weight: 600; color: #333;">
                    Active Jobs (${jobCount})
                </span>
                <div style="display: flex; gap: 4px;">
                    <button onclick="loadActiveJobs()" style="
                        background: none;
                        border: 1px solid #dee2e6;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-size: 0.75rem;
                        color: #6c757d;
                        cursor: pointer;
                    " title="Refresh jobs">
                        🔄
                    </button>
                    <button onclick="syncJobsWithBackend().then(() => loadActiveJobs())" style="
                        background: none;
                        border: 1px solid #dee2e6;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-size: 0.75rem;
                        color: #6c757d;
                        cursor: pointer;
                    " title="Sync with backend">
                        🔄🔗
                    </button>
                </div>
            </div>
        </div>
        <div style="max-height: 300px; overflow-y: auto;">
            ${jobsHtml}
        </div>
    `;
}

// Retry monitoring for jobs with connection errors
function retryJobMonitoring(jobId) {
    if (activeIngestionJobs[jobId]) {
        // Reset job status
        activeIngestionJobs[jobId].status = 'running';
        activeIngestionJobs[jobId].message = 'Reconnecting to job...';

        // Start monitoring again
        monitorIngestionJob(jobId);

        // Update display
        loadActiveJobs();
    }
}

// Clear knowledge base
async function clearKnowledgeBase() {
    const confirmed = confirm(
        '⚠️ WARNING: This will permanently delete ALL documents from the knowledge base.\n\n' +
        'This action cannot be undone. Are you sure you want to continue?'
    );

    if (!confirmed) {
        return;
    }

    try {
        showIngestionProgress('🗑️ Clearing knowledge base...');

        const response = await fetch(`${API_BASE_URL}/knowledge/clear`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const result = await response.json();

            updateIngestionProgress({
                status: 'completed',
                message: '✅ Knowledge base cleared successfully!',
                progress: 1.0
            });

            // Refresh stats after clearing
            setTimeout(() => {
                loadKnowledgeStats();
            }, 2000);

        } else {
            const error = await response.json();
            updateIngestionProgress({
                status: 'failed',
                message: `❌ Failed to clear knowledge base: ${error.detail || 'Unknown error'}`,
                progress: 0
            });
        }
    } catch (error) {
        updateIngestionProgress({
            status: 'failed',
            message: `❌ Network error while clearing knowledge base: ${error.message}`,
            progress: 0
        });
    }
}

// Clear knowledge base by category
async function clearKnowledgeBaseByCategory(category) {
    const confirmed = confirm(
        `⚠️ WARNING: This will permanently delete ALL documents from the "${category}" category.\n\n` +
        'This action cannot be undone. Are you sure you want to continue?'
    );

    if (!confirmed) {
        return;
    }

    try {
        showIngestionProgress(`🗑️ Clearing category "${category}"...`);

        const response = await fetch(`${API_BASE_URL}/knowledge/clear/${encodeURIComponent(category)}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const result = await response.json();

            updateIngestionProgress({
                status: 'completed',
                message: `✅ ${result.message}`,
                progress: 1.0
            });

            // Refresh stats after clearing
            setTimeout(() => {
                loadKnowledgeStats();
            }, 2000);

        } else {
            const error = await response.json();
            updateIngestionProgress({
                status: 'failed',
                message: `❌ Failed to clear category: ${error.detail || 'Unknown error'}`,
                progress: 0
            });
        }
    } catch (error) {
        updateIngestionProgress({
            status: 'failed',
            message: `❌ Network error while clearing category: ${error.message}`,
            progress: 0
        });
    }
}
