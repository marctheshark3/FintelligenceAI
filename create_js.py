#!/usr/bin/env python3

js_content = """// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    document.getElementById(tabName).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
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

// Generic request handler
async function submitRequest(endpoint, data, loadingId, responseId, contentId, renderFunction) {
    const loadingElement = document.getElementById(loadingId);
    const responseElement = document.getElementById(responseId);
    const contentElement = document.getElementById(contentId);

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
                <p>${escapeHtml(data.explanation)}</p>
            </div>

            <div class="response-item">
                <h4><i class="fas fa-chart-line"></i> Complexity Score</h4>
                <p><strong>${data.complexity_score}/10</strong></p>
            </div>

            ${data.gas_estimate ? `
            <div class="response-item">
                <h4><i class="fas fa-gas-pump"></i> Gas Estimate</h4>
                <p><strong>${data.gas_estimate}</strong></p>
            </div>
            ` : ''}

            ${data.optimization_suggestions && data.optimization_suggestions.length > 0 ? `
            <div class="response-item">
                <h4><i class="fas fa-lightbulb"></i> Optimization Suggestions</h4>
                <ul>
                    ${data.optimization_suggestions.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
        </div>

        ${data.example_usage ? `
        <div class="response-item" style="margin-top: 15px;">
            <h4><i class="fas fa-play"></i> Example Usage</h4>
            <div class="code-block">
                <pre><code class="language-scala">${escapeHtml(data.example_usage)}</code></pre>
            </div>
        </div>
        ` : ''}
    `;

    if (typeof Prism !== 'undefined') {
        Prism.highlightAllUnder(container);
    }
}

function renderResearchResponse(data, container) {
    let content = '';

    if (typeof data === 'string') {
        content = data;
    } else if (data.summary) {
        content = data.summary;
    } else if (data.response) {
        content = data.response;
    } else {
        content = JSON.stringify(data, null, 2);
    }

    container.innerHTML = `
        <div class="response-item">
            <h4><i class="fas fa-search"></i> Research Results</h4>
            <div style="white-space: pre-wrap; line-height: 1.6;">${escapeHtml(content)}</div>
        </div>
    `;
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
            contentElement.innerHTML = `<h4>Error</h4><p>Failed to fetch agent status</p>`;
        }
    } catch (error) {
        const loadingElement = document.getElementById('statusLoading');
        const responseElement = document.getElementById('statusResponse');
        const contentElement = document.getElementById('statusContent');

        loadingElement.classList.remove('show');
        responseElement.style.display = 'block';
        responseElement.className = 'response-section error';
        contentElement.innerHTML = `<h4>Connection Error</h4><p>${error.message}</p>`;
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

// Initialize when DOM is loaded
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
    setInterval(checkApiStatus, 30000);
});
"""

with open("ui/app.js", "w") as f:
    f.write(js_content)

print("JavaScript file created successfully!")
