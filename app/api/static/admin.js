const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const systemStatusSummary = document.getElementById('system-status-summary');
const searchBtn = document.getElementById('search-btn');
const queryInput = document.getElementById('query-input');
const downloadBtn = document.getElementById('download-report-btn');
let lastQueryResult = null;

// --- IMPROVEMENT 2: Session ID for Conversation Memory ---
// Unique per browser tab/session — persists until page is refreshed
const SESSION_ID = 'session_' + Math.random().toString(36).substr(2, 9);

// Neural Query Logic
if (searchBtn && queryInput) {
    searchBtn.addEventListener('click', performNeuralQuery);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performNeuralQuery();
    });
}

if (downloadBtn) {
    downloadBtn.addEventListener('click', downloadReport);
}

// --- Neural Flow Control ---
function setFlowMode(mode, content = "") {
    const paths = document.querySelectorAll('.flow-path');
    const nodes = document.querySelectorAll('.flow-node');
    const container = document.getElementById('neural-flow-map');
    
    // Reset
    paths.forEach(p => { p.className = 'flow-path'; });
    nodes.forEach(n => { n.classList.remove('active'); });
    document.querySelectorAll('.data-packet').forEach(p => p.remove());

    const snippet = content ? content.substring(0, 100) + "..." : "Technical Data Fragment...";

    if (mode === 'write') {
        paths.forEach(p => { 
            p.classList.add('pulse-forward'); 
            const packet = document.createElement('div');
            packet.className = 'data-packet streaming';
            packet.innerText = snippet;
            p.appendChild(packet);
        });
        document.getElementById('node-input').classList.add('active');
        document.getElementById('node-process').classList.add('active');
        document.getElementById('node-memory').classList.add('active');
    } else if (mode === 'read') {
        paths.forEach(p => { 
            p.classList.add('pulse-backward'); 
            const packet = document.createElement('div');
            packet.className = 'data-packet streaming-backward';
            packet.innerText = snippet;
            p.appendChild(packet);
        });
        document.getElementById('node-output').classList.add('active');
        document.getElementById('node-brain').classList.add('active');
        document.getElementById('node-memory').classList.add('active');
    } else if (mode === 'api') {
        paths.forEach(p => { 
            p.classList.add('pulse-api'); 
            const packet = document.createElement('div');
            packet.className = 'data-packet streaming';
            packet.innerText = "API LOG: [RNOC] SYNCING DATA STREAM...";
            p.appendChild(packet);
        });
        document.getElementById('node-api').classList.add('active');
        document.getElementById('node-process').classList.add('active');
        document.getElementById('node-memory').classList.add('active');
    }
}

function resetFlow() {
    const paths = document.querySelectorAll('.flow-path');
    const nodes = document.querySelectorAll('.flow-node');
    paths.forEach(p => { p.className = 'flow-path'; });
    nodes.forEach(n => { n.classList.remove('active'); });
}

async function performNeuralQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    const retrievedContainer = document.getElementById('retrieved-data');
    const responseBox = document.getElementById('ai-response');

    // Reset UI
    retrievedContainer.innerHTML = '<div class="empty-state">Searching neural pathways...</div>';
    responseBox.innerHTML = '<p class="loading">Deep Neural Search initiated... Analyizing vector harmonics...</p>';
    responseBox.classList.add('loading');
    searchBtn.disabled = true;
    downloadBtn.disabled = true;
    setFlowMode('read', query);

    try {
        const response = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}&limit=10&session_id=${SESSION_ID}`);
        const data = await response.json();
        lastQueryResult = { query, ...data };
        // 1. Show Retrieved Knowledge (X-Algo Evidence)
        if (data.context && data.context.length > 0) {
            retrievedContainer.innerHTML = data.context.map(m => `
                <div class="memory-item ${m.metadata.is_visual ? 'visual-item' : ''}">
                    <div class="memory-header">
                        <span class="score-pill">Score: ${m.score.toFixed(2)}</span>
                        <span class="source-link">${m.metadata.filename || 'System Source'}</span>
                    </div>
                    ${m.retrieval_reason ? `<div style="font-size:0.72rem;color:#94a3b8;margin:4px 0 6px 0;padding:2px 8px;background:#1e293b;border-radius:4px;display:inline-block">🔍 ${m.retrieval_reason}</div>` : ''}
                    ${m.metadata.is_visual && m.metadata.media_url ? `
                        <div class="memory-image">
                            <img src="${API_BASE}${m.metadata.media_url}" alt="Neural Evidence" onclick="window.open(this.src)">
                        </div>
                    ` : ''}
                    <div class="memory-content">${m.content}</div>
                    <div class="feedback-controls">
                        <button class="feedback-btn up" onclick="submitFeedback('${m.id}', 1.0, this)">
                            👍 Helpful
                        </button>
                        <button class="feedback-btn down" onclick="submitFeedback('${m.id}', -1.0, this)">
                            👎 Irrelevant
                        </button>
                        <span class="feedback-status" id="feedback-status-${m.id}"></span>
                    </div>
                </div>
            `).join('');
        } else {
            retrievedContainer.innerHTML = '<div class="empty-state">No relevant evidence fragments found in neural core.</div>';
        }

        // 2. Show Confidence Badge + AI Answer
        const confidence = data.confidence || 'MEDIUM';
        const confidenceReason = data.context && data.context.length > 0 ? data.context[0].confidence_reason : '';
        
        const confidenceMap = {
            'EXCEPTIONAL': { icon: '💎', label: 'Exceptional Accuracy', color: '#8b5cf6' },
            'HIGH':        { icon: '🟢', label: 'High Confidence',      color: '#22c55e' },
            'MEDIUM':      { icon: '🟡', label: 'Medium Confidence',    color: '#f59e0b' },
            'LOW':         { icon: '🔴', label: 'Low Confidence',       color: '#ef4444' },
            'NONE':        { icon: '⚪', label: 'No Evidence',          color: '#94a3b8' }
        };
        
        const conf = confidenceMap[confidence] || confidenceMap['MEDIUM'];
        const confidenceBadge = `
            <div class="confidence-container" style="margin-bottom: 16px;">
                <div style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;background:${conf.color}22;border:1px solid ${conf.color};font-size:0.8rem;font-weight:600;color:${conf.color}">
                    ${conf.icon} ${conf.label}
                </div>
                ${confidenceReason ? `<div style="font-size:0.75rem;color:#94a3b8;margin-top:6px;font-style:italic;">ℹ️ ${confidenceReason}</div>` : ''}
            </div>`;

        if (data.answer) {
            const answerHtml = typeof marked !== 'undefined' ? marked.parse(data.answer) : data.answer;
            responseBox.innerHTML = confidenceBadge + answerHtml;
            downloadBtn.disabled = false;
        } else {
            responseBox.innerHTML = `${confidenceBadge}<div class="empty-state">Knowledge gap detected. The AI could not synthesize a confirmed answer from the retrieved context.</div>`;
        }

    } catch (err) {
        console.error("Query error:", err);
        responseBox.innerHTML = `<div class="error-msg">Error: Neural server heartbeat lost. Ensure FastAPI is running.</div>`;
    } finally {
        responseBox.classList.remove('loading');
        searchBtn.disabled = false;
        resetFlow();
    }
}

async function submitFeedback(docId, score, btn) {
    if (btn.classList.contains('active')) return;
    
    const container = btn.closest('.feedback-controls');
    const statusEl = document.getElementById(`feedback-status-${docId}`);
    const allBtns = container.querySelectorAll('.feedback-btn');
    
    // Optimistic UI
    allBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (statusEl) statusEl.innerText = 'Syncing...';

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                doc_id: docId,
                query: queryInput.value,
                score: score
            })
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            if (statusEl) statusEl.innerText = 'Learned! ✓';
            btn.style.borderColor = score > 0 ? '#10b981' : '#ef4444';
        } else {
            if (statusEl) statusEl.innerText = 'Failed';
        }
    } catch (err) {
        console.error("Feedback error:", err);
        if (statusEl) statusEl.innerText = 'Error';
    }
}

function downloadReport() {
    if (!lastQueryResult) return;

    const { query, answer, context } = lastQueryResult;
    let report = `# AI TECHNICAL ANALYSIS REPORT\n\n`;
    report += `**Query:** ${query}\n`;
    report += `**Date:** ${new Date().toLocaleString()}\n\n`;
    report += `## EXECUTIVE SUMMARY\n\n${answer}\n\n`;
    report += `## SOURCE EVIDENCE (X-ALGO RETRIEVAL)\n\n`;

    context.forEach((m, i) => {
        report += `### [${i + 1}] Source: ${m.metadata.filename || 'System'}\n`;
        report += `**Relevance Score:** ${m.score.toFixed(4)}\n`;
        report += `**Excerpt:**\n> ${m.content.replace(/\n/g, '\n> ')}\n\n`;
    });

    report += `\n---\n*Generated by AI Knowledge Beyond Neural Core*`;

    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI_Report_${query.substring(0, 20).replace(/\s+/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

// Dynamic API Base Detection
const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';

// Tab Switching Logic
tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');

        // Update Buttons
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update Content
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === `${tabId}-tab`) {
                content.classList.add('active');
            }
        });

        // Load data if needed
        if (tabId === 'explorer') fetchMemories();
        if (tabId === 'graph') fetchGraphData();
        if (tabId === 'system') fetchSystemHealth();
    });
});

// Periodic Health Check
setInterval(fetchSystemHealth, 5000);
fetchSystemHealth(); // Initial check

async function fetchSystemHealth() {
    try {
        const response = await fetch(`${API_BASE}/system/health`);
        const data = await response.json();

        updateHealthCard('database', data.database);
        if (data.database_type) {
            const dbHealth = document.getElementById('database-health');
            if (dbHealth) dbHealth.innerText = `${data.database_type} (${data.database})`;
        }
        updateHealthCard('redis', data.redis);
        updateHealthCard('graph_rag', data.graph_rag);
        updateHealthCard('groq_cloud', data.groq_cloud ? 'online' : 'offline');

        // Update Knowledge Base Count
        const storageText = document.getElementById('storage-health');
        if (storageText) {
            storageText.innerText = `${data.memory_count || 0} Fragments`;
        }

        // Update header summary
        const allOnline = data.database === 'online' &&
            data.redis === 'online' &&
            data.graph_rag.includes('online');
        systemStatusSummary.innerHTML = allOnline
            ? '<span class="dot green"></span> System Online'
            : '<span class="dot red"></span> Service Interruption';
    } catch (err) {
        systemStatusSummary.innerHTML = '<span class="dot red"></span> Server Offline';
    }
}

function updateHealthCard(service, status) {
    const card = document.querySelector(`.health-card[data-service="${service}"]`);
    const text = document.getElementById(`${service}-health`);
    if (card && text) {
        card.className = `health-card ${status}`;
        text.innerText = status.charAt(0).toUpperCase() + status.slice(1);
    }
}

async function fetchMemories() {
    const memoryList = document.getElementById('memory-list');
    try {
        const response = await fetch(`${API_BASE}/memories`);
        const memories = await response.json();

        if (memories.length === 0) {
            memoryList.innerHTML = '<div class="empty-state">No memories found in neural core.</div>';
            return;
        }

        memoryList.innerHTML = memories.map(m => `
            <div class="memory-item">
                <div class="memory-header">
                    <span>ID: ${m.id}</span>
                    <span>${new Date(m.created_at).toLocaleString()}</span>
                </div>
                <div class="memory-content">${m.content}</div>
                <div class="memory-meta">
                    <span class="meta-tag">Model: ${m.metadata.embedding_model || 'unknown'}</span>
                    <span class="meta-tag">Source: ${m.metadata.filename || 'manual'}</span>
                    <span class="meta-tag">Authority: ${m.metadata.authority || '1.0'}</span>
                </div>
            </div>
        `).join('');
    } catch (err) {
        memoryList.innerHTML = '<div class="empty-state">Failed to fetch neural data.</div>';
    }
}

const apiSyncBtn = document.getElementById('api-sync-btn');
if (apiSyncBtn) {
    apiSyncBtn.addEventListener('click', async () => {
        apiSyncBtn.disabled = true;
        apiSyncBtn.innerText = '⌛ Syncing...';
        setFlowMode('api');

        try {
            const response = await fetch(`${API_BASE}/ingest/api-sync`, { method: 'POST' });
            const result = await response.json();
            alert(`Sync Complete! ${result.processed || 0} logs analyzed and stored.`);
        } catch (err) {
            console.error("Sync error:", err);
            alert("API Sync failed. Check server logs.");
        } finally {
            apiSyncBtn.disabled = false;
            apiSyncBtn.innerText = '🔄 Sync with External API (T-API)';
            resetFlow();
        }
    });
}

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const pulseList = document.getElementById('pulse-list');
const heavyParsingToggle = document.getElementById('heavy-parsing-toggle');

// Drag and Drop Handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    const files = e.dataTransfer.files;
    handleFiles(files);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
});

function handleFiles(files) {
    if (files.length === 0) return;

    // Clear empty state if it exist
    const emptyState = document.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    Array.from(files).forEach(file => {
        uploadFile(file);
    });
}

async function uploadFile(file) {
    const heavyParsing = heavyParsingToggle.checked;
    const itemId = Math.random().toString(36).substring(7);

    // Create UI Item
    const item = document.createElement('div');
    item.className = 'pulse-item';
    item.id = `item-${itemId}`;
    setFlowMode('write', file.name);
    item.innerHTML = `
        <div class="item-main">
            <div class="file-info">
                <span class="file-icon">${getFileIcon(file.name)}</span>
                <span class="file-name">${file.name}</span>
            </div>
            <div class="item-actions">
                <span class="badge ${heavyParsing ? 'heavy' : ''}">${heavyParsing ? 'HEAVY' : 'STD'}</span>
            </div>
        </div>
        <div class="progress-container">
            <div class="progress-bar" id="progress-${itemId}"></div>
        </div>
        <div class="status-text">
            <span id="status-${itemId}">Initializing transmission...</span>
            <span id="percent-${itemId}">0%</span>
        </div>
    `;
    pulseList.prepend(item);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('heavy_parsing', heavyParsing);
    formData.append('metadata_json', JSON.stringify({
        source: 'admin_dashboard',
        ingested_at: new Date().toISOString()
    }));

    try {
        const progressBar = document.getElementById(`progress-${itemId}`);
        const statusText = document.getElementById(`status-${itemId}`);
        const percentText = document.getElementById(`percent-${itemId}`);

        // Update progress simulated while uploading
        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 10;
                updateProgress(itemId, Math.min(progress, 90), "Transmitting neural data...");
            }
        }, 300);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        clearInterval(interval);

        if (response.ok) {
            const result = await response.json();
            updateProgress(itemId, 50, "Task queued. Synchronizing with Vector DB...");

            if (result.task_id) {
                // Start pulling the real status from Celery
                pollTaskStatus(result.task_id, itemId);
            } else {
                updateProgress(itemId, 100, `Success: ${result.chunks_identified} chunks ingested.`);
                item.classList.add('done');
                resetFlow();
            }
        } else {
            const error = await response.json();
            updateProgress(itemId, 100, `Error: ${error.detail || 'Upload failed'}`);
            item.style.borderColor = "#ef4444";
        }
    } catch (err) {
        updateProgress(itemId, 0, `Failed: Connection error`);
        console.error(err);
    }
}

async function pollTaskStatus(taskId, itemId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/task/${taskId}`);
            const data = await response.json();

            if (data.status === 'PROGRESS') {
                const meta = data.meta || {};
                const percent = meta.total > 0 ? (meta.current / meta.total) * 100 : 50;
                updateProgress(itemId, percent, meta.status || "Processing...");
            } else if (data.status === 'SUCCESS') {
                clearInterval(pollInterval);
                const count = data.result ? data.result.count : '?';
                updateProgress(itemId, 100, `Success: ${count} chunks stored in Vector DB.`);
                document.getElementById(`item-${itemId}`).classList.add('done');
            } else if (data.status === 'FAILURE') {
                clearInterval(pollInterval);
                updateProgress(itemId, 100, `Error: ${data.error || 'Ingestion failed'}`);
                document.getElementById(`item-${itemId}`).style.borderColor = "#ef4444";
            }
        } catch (err) {
            console.error("Polling error:", err);
            updateProgress(itemId, 0, "Error: Connection lost or restricted.");
            clearInterval(pollInterval);
        }
    }, 1000);
}

function updateProgress(id, percent, text) {
    const bar = document.getElementById(`progress-${id}`);
    const status = document.getElementById(`status-${id}`);
    const pct = document.getElementById(`percent-${id}`);

    if (bar) bar.style.width = `${percent}%`;
    if (status) status.innerText = text;
    if (pct) pct.innerText = `${Math.round(percent)}%`;

    // Protocol lock removed to allow direct file usage.
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    switch (ext) {
        case 'pdf': return '📄';
        case 'csv':
        case 'xlsx':
        case 'xls': return '📊';
        case 'json': return '📦';
        case 'docx': return '📝';
        default: return '📁';
    }
}
// Knowledge Graph Logic
let network = null;
let allFacts = [];
let factsCurrentPage = 1;
const factsPageSize = 10;

async function fetchGraphData() {
    const container = document.getElementById('graph-container');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/system/graph`);
        const data = await response.json();

        if (!data.nodes || data.nodes.length === 0) {
            container.innerHTML = '<div class="empty-state">No technical relationships extracted yet. Ingest more data to see the graph.</div>';
            return;
        }

        allFacts = data.edges || [];
        factsCurrentPage = 1;

        const visData = {
            nodes: new vis.DataSet(data.nodes),
            edges: new vis.DataSet(data.edges)
        };

        const options = {
            nodes: {
                shape: 'dot',
                size: 20,
                font: { size: 14, color: '#ffffff' },
                borderWidth: 2,
                color: {
                    background: '#2563eb',
                    border: '#3b82f6',
                    highlight: { background: '#60a5fa', border: '#93c5fd' }
                }
            },
            edges: {
                width: 2,
                color: { color: 'rgba(255, 255, 255, 0.3)', highlight: '#ffffff' },
                font: { size: 10, color: '#94a3b8', align: 'top' },
                arrows: { to: { enabled: true, scaleFactor: 0.5 } }
            },
            physics: {
                enabled: true,
                barnesHut: { gravitationalConstant: -2000, centralGravity: 0.3, springLength: 95 },
                stabilization: { iterations: 100 }
            },
            interaction: { hover: true, tooltipDelay: 200 }
        };

        if (network) {
            network.destroy();
        }
        network = new vis.Network(container, visData, options);

        renderFactsPage();

    } catch (err) {
        console.error("Graph fetch error:", err);
        container.innerHTML = '<div class="error-msg">Failed to connect to Neural Graph Engine.</div>';
    }
}

function renderFactsPage() {
    const factsBody = document.getElementById('facts-body');
    const factCount = document.getElementById('fact-count');
    const indicator = document.getElementById('page-indicator');
    const prevBtn = document.getElementById('prev-facts-btn');
    const nextBtn = document.getElementById('next-facts-btn');

    if (!factsBody) return;

    const totalPages = Math.ceil(allFacts.length / factsPageSize) || 1;
    if (factsCurrentPage > totalPages) factsCurrentPage = totalPages;

    const start = (factsCurrentPage - 1) * factsPageSize;
    const end = start + factsPageSize;
    const pageData = allFacts.slice(start, end);

    factCount.innerText = `${allFacts.length} Facts Discovered`;
    indicator.innerText = `Page ${factsCurrentPage} of ${totalPages}`;

    factsBody.innerHTML = pageData.map(edge => `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.2s;">
            <td style="padding: 12px; font-weight: 500;">${edge.from}</td>
            <td style="padding: 12px;"><span style="background: rgba(37, 99, 235, 0.2); color: var(--primary); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;">${edge.label}</span></td>
            <td style="padding: 12px; font-weight: 500;">${edge.to}</td>
        </tr>
    `).join('');

    prevBtn.style.opacity = factsCurrentPage > 1 ? "1" : "0.3";
    nextBtn.style.opacity = factsCurrentPage < totalPages ? "1" : "0.3";
    prevBtn.disabled = factsCurrentPage === 1;
    nextBtn.disabled = factsCurrentPage === totalPages;
}

// Pagination Listeners
document.getElementById('prev-facts-btn')?.addEventListener('click', () => {
    if (factsCurrentPage > 1) {
        factsCurrentPage--;
        renderFactsPage();
    }
});

document.getElementById('next-facts-btn')?.addEventListener('click', () => {
    const totalPages = Math.ceil(allFacts.length / factsPageSize);
    if (factsCurrentPage < totalPages) {
        factsCurrentPage++;
        renderFactsPage();
    }
});

const refreshGraphBtn = document.getElementById('refresh-graph-btn');
if (refreshGraphBtn) {
    refreshGraphBtn.addEventListener('click', fetchGraphData);
}
