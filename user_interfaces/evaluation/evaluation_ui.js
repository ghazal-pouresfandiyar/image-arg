// ============================================================================
// IMAGE-ARGUMENT EVALUATION UI — LOGIC
// ============================================================================
'use strict';

// ---------------------------------------------------------------------------
// STATE
// ---------------------------------------------------------------------------
let DATA = null;                  // from /api/data
let evaluations = {};             // { imageId: { modelKey: { metricKey: value, comment: '' } } }
let currentImageId = null;
let currentModel = null;
let saveTimer = null;

const MODEL_KEYS = ['llava', 'minicpm', 'nemotron'];
const GROUP_ORDER = ['Output Format', 'Premises', 'Conclusion', 'Human Alignment'];
const GROUP_COLORS = {
    'Output Format': '#d4a5a5',
    'Premises': '#9bb8d3',
    'Conclusion': '#a8c99e',
    'Human Alignment': '#e8a44c',
};

// ---------------------------------------------------------------------------
// HELPERS
// ---------------------------------------------------------------------------
function escapeHtml(s) {
    return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function getModelEval(imageId, modelKey) {
    if (!evaluations[imageId]) evaluations[imageId] = {};
    if (!evaluations[imageId][modelKey]) evaluations[imageId][modelKey] = {};
    return evaluations[imageId][modelKey];
}

// ---------------------------------------------------------------------------
// SAVE / RESTORE / RESET / EXPORT
// ---------------------------------------------------------------------------
function showSaveStatus(msg, isError) {
    const el = document.getElementById('save-status');
    el.textContent = msg;
    el.className = 'text-xs ' + (isError ? 'text-red-400' : 'text-gray-400');
}

function scheduleSave() {
    clearTimeout(saveTimer);
    showSaveStatus('Saving...');
    saveTimer = setTimeout(saveNow, 600);
}

async function saveNow() {
    clearTimeout(saveTimer);
    try {
        const resp = await fetch('/api/evaluations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ evaluations }),
        });
        if (!resp.ok) throw new Error(resp.status);
        showSaveStatus('Saved ' + new Date().toLocaleTimeString());
    } catch (e) {
        showSaveStatus('Save failed: ' + e.message, true);
    }
}

async function resetAll() {
    if (!confirm('Delete ALL evaluation data? This cannot be undone.')) return;
    await fetch('/api/evaluations', { method: 'DELETE' });
    evaluations = {};
    renderAll();
    showSaveStatus('Reset');
}

function exportJSON() {
    const blob = new Blob([JSON.stringify({ evaluations, updated_at: new Date().toISOString() }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'evaluations_export.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

function importJSON() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const parsed = JSON.parse(reader.result);
                const evals = parsed.evaluations || parsed;
                if (typeof evals !== 'object' || Array.isArray(evals)) throw new Error('bad structure');
                evaluations = evals;
                renderAll();
                saveNow();
                showSaveStatus('Imported & saved');
            } catch (err) {
                showSaveStatus('Import failed: ' + err.message, true);
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

// ---------------------------------------------------------------------------
// RENDERING
// ---------------------------------------------------------------------------
function renderAll() {
    renderGallery();
    renderModelTabs();
    renderCurrent();
    updateProgress();
}

function renderGallery() {
    const g = document.getElementById('gallery');
    g.innerHTML = '';
    DATA.images.forEach(img => {
        const im = document.createElement('div');
        const isActive = img.id === currentImageId;
        const anyRated = evaluations[img.id] && Object.values(evaluations[img.id]).some(m => Object.keys(m).length > 0);
        im.className = 'thumb shrink-0' + (isActive ? ' active' : '') + (anyRated ? ' rated' : '');
        im.innerHTML = `<img src="/api/images/${img.id}.jpg" class="w-24 h-16 object-cover block" alt="" /><div class="text-[10px] text-center bg-gray-100 py-0.5">${img.id}</div>`;
        im.onclick = () => { currentImageId = img.id; renderAll(); };
        g.appendChild(im);
        if (isActive) im.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
    });
}

function renderModelTabs() {
    const t = document.getElementById('model-tabs');
    t.innerHTML = '';
    MODEL_KEYS.forEach(key => {
        const label = (DATA.models && DATA.models[key]) || key;
        const b = document.createElement('button');
        b.className = 'model-tab bg-gray-200 ' + (key === currentModel ? ' active' : ' text-gray-500 hover:text-gray-700');
        b.textContent = label;
        b.onclick = () => { currentModel = key; renderCurrent(); updateModelDots(); };
        t.appendChild(b);
    });
}

function renderModelDots() {
    const img = DATA.images.find(i => i.id === currentImageId);
    if (!img) return;
    const dots = document.getElementById('model-dots');
    dots.innerHTML = MODEL_KEYS.map(key => {
        const ev = evaluations[img.id] && evaluations[img.id][key];
        const rated = !!ev && Object.keys(ev).length > 0;
        return `<span title="${(DATA.models && DATA.models[key]) || key}${rated ? ' (rated)' : ' (not rated)'}" class="inline-block w-3 h-3 rounded-full ${rated ? 'bg-green-500' : 'bg-gray-400'}"></span>`;
    }).join('');
}

function renderCurrent() {
    const img = DATA.images.find(i => i.id === currentImageId);
    if (!img) return;

    document.getElementById('cur-id').textContent = img.id;
    document.getElementById('main-image').src = `/api/images/${img.id}.jpg`;

    const m = img.metadata || {};
    const md = document.getElementById('metadata');
    const pairs = [
        ['Type', m.type], ['Setting', m.setting], ['Source', m.source_file],
        ['Animals', m.animals], ['Consequences', m.consequences], ['Climate action', m.climateaction],
    ];
    md.innerHTML = pairs.filter(p => p[1]).map(([k, v]) =>
        `<span class="bg-gray-100 border border-gray-200 rounded px-2 py-0.5"><b>${k}:</b> ${escapeHtml(v)}</span>`
    ).join('') || '<span class="text-gray-400 italic">No metadata</span>';

    renderModelDots();
    renderHumanCards(img);
    renderModelOutput(img);
    renderRatingForm(img);

    const ev = getModelEval(img.id, currentModel);
    document.getElementById('comment-box').value = ev.comment || '';
}

// --- Human cards -----------------------------------------------------------
function renderHumanCards(img) {
    const cont = document.getElementById('human-cards');
    const humans = [
        { key: 'human1', title: 'Human Evaluation 1' },
        { key: 'human2', title: 'Human Evaluation 2' },
    ];
    cont.innerHTML = humans.map(h => {
        const d = img[h.key] || { premises: [], conclusions: [], notes: '' };
        const prem = d.premises.length
            ? d.premises.map((p, i) => `<li class="bg-green-50 border border-green-200 rounded p-2 text-sm">${escapeHtml(p)}</li>`).join('')
            : '<li class="text-sm text-gray-400 italic">No premises</li>';
        const conc = d.conclusions.length
            ? d.conclusions.map((c, i) => `<li class="bg-green-50 border border-green-200 rounded p-2 text-sm font-medium">${escapeHtml(c)}</li>`).join('')
            : '<li class="text-sm text-gray-400 italic">No conclusions</li>';
        return `
        <div class="card p-3 border-l-4 border-l-green-500">
            <h3 class="font-bold text-green-800 text-sm mb-2">${h.title}</h3>
            <div class="mb-2">
                <div class="text-xs font-semibold text-gray-500 uppercase mb-1">Premises</div>
                <ul class="space-y-1">${prem}</ul>
            </div>
            <div>
                <div class="text-xs font-semibold text-gray-500 uppercase mb-1">Conclusions</div>
                <ul class="space-y-1">${conc}</ul>
            </div>
            ${d.notes ? `<p class="text-xs text-gray-500 mt-2 italic">Note: ${escapeHtml(d.notes)}</p>` : ''}
        </div>`;
    }).join('');
}

// --- Model output ----------------------------------------------------------
function renderModelOutput(img) {
    const mo = img.models[currentModel] || { plan: '', premises: [], conclusions: [], raw: '' };
    const cont = document.getElementById('model-output');
    const prem = mo.premises.length
        ? mo.premises.map((p, i) => `<li class="bg-gray-50 border border-gray-200 rounded p-2 text-sm">${escapeHtml(p)}</li>`).join('')
        : '<li class="text-sm text-gray-400 italic">No premises parsed</li>';
    const conc = mo.conclusions.length
        ? mo.conclusions.map((c, i) => `<li class="bg-gray-50 border border-gray-200 rounded p-2 text-sm font-medium">${escapeHtml(c)}</li>`).join('')
        : '<li class="text-sm text-gray-400 italic">No conclusions parsed</li>';
    cont.innerHTML = `
        <h3 class="font-bold text-indigo-800 text-sm mb-3">${(DATA.models && DATA.models[currentModel]) || currentModel} &mdash; Output</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
                <div class="text-xs font-semibold text-gray-500 uppercase mb-1">Premises (${mo.premises.length})</div>
                <ul class="space-y-1">${prem}</ul>
            </div>
            <div>
                <div class="text-xs font-semibold text-gray-500 uppercase mb-1">Conclusions (${mo.conclusions.length})</div>
                <ul class="space-y-1">${conc}</ul>
            </div>
        </div>
        ${mo.plan ? `<details class="mt-3"><summary class="cursor-pointer text-xs font-semibold text-gray-500 hover:text-gray-700">Reasoning plan</summary><p class="text-sm text-gray-600 mt-1 whitespace-pre-wrap bg-gray-50 rounded p-2 border border-gray-200">${escapeHtml(mo.plan)}</p></details>` : ''}
        ${mo.raw ? `<details class="mt-2"><summary class="cursor-pointer text-xs font-semibold text-gray-500 hover:text-gray-700">Raw output</summary><pre class="text-xs text-gray-600 mt-1 whitespace-pre-wrap bg-gray-50 rounded p-2 border border-gray-200 max-h-60 overflow-auto">${escapeHtml(mo.raw)}</pre></details>` : ''}
    `;
}

// --- Rating form -----------------------------------------------------------
function renderRatingForm(img) {
    const ev = getModelEval(img.id, currentModel);
    const mo = img.models[currentModel] || { premises: [], conclusions: [] };
    const totalPremises = (mo.premises || []).length;
    const totalConclusions = (mo.conclusions || []).length;
    const cont = document.getElementById('rating-form');

    let html = `<div class="flex items-center justify-between mb-3">
        <h3 class="font-bold text-gray-800 text-sm">Metrics <span class="font-normal text-gray-400">(based on metrics.html)</span></h3>
        <button class="group-btn" onclick="clearGroup('${currentModel}')" title="Clear all metrics for this model-pair">Clear all for ${(DATA.models && DATA.models[currentModel]) || currentModel}</button>
    </div>`;

    for (const group of GROUP_ORDER) {
        const metrics = (DATA.metrics[group] || []).filter(m => m.key !== 'premise_coverage');
        html += `<div class="mt-4">
            <div class="flex items-center gap-2 mb-2">
                <span class="metric-group-title" style="background:${GROUP_COLORS[group]}">${group}</span>
                <span class="text-[10px] text-gray-400">(${metrics.length})</span>
            </div>
            <div class="space-y-3">`;

        for (const met of metrics) {
            const current = ev[met.key];
            const optsHtml = met.options.map((opt, oi) => {
                const sel = current === opt ? ' selected' : '';
                return `<button type="button" class="opt-btn${sel}" onclick="setMetric('${met.key}', this.dataset.v)" data-v="${escapeHtml(opt)}">${escapeHtml(opt)}</button>`;
            }).join('');
            html += `
            <div>
                <div class="flex items-start justify-between gap-2">
                    <label class="text-sm font-semibold text-gray-700">${met.label}
                        <span class="block text-xs font-normal text-gray-400">${escapeHtml(met.what)}</span>
                    </label>
                    ${current ? `<button class="text-[10px] text-gray-400 hover:text-red-500 underline" onclick="clearMetric('${met.key}')">clear</button>` : ''}
                </div>
                <div class="flex flex-wrap gap-1.5 mt-1">${optsHtml}</div>
            </div>`;
        }
        html += `</div></div>`;
    }

    // Premise Coverage (numeric) — separate because it depends on conclusions
    html += `<div class="mt-5 pt-3 border-t border-gray-200">
        <div class="flex items-center gap-2 mb-2">
            <span class="metric-group-title" style="background:${GROUP_COLORS['Conclusion']}">Conclusion &rsaquo; Premise Coverage</span>
            <span class="text-[10px] text-gray-400">total premises: ${totalPremises}</span>
        </div>`;

    if (totalConclusions === 0) {
        html += `<p class="text-sm text-gray-400 italic">No conclusions parsed for this model &mdash; nothing to cover.</p>`;
    } else {
        html += `<div class="space-y-2">`;
        for (let i = 0; i < totalConclusions; i++) {
            const key = `premise_coverage_${i}`;
            const val = ev[key];
            const numeric = (typeof val === 'number') ? val : (val === '' || val == null ? '' : Number(val));
            const pct = (typeof numeric === 'number' && totalPremises > 0 && !isNaN(numeric))
                ? Math.round((Math.min(numeric, totalPremises) / totalPremises) * 100) : null;
            html += `
            <div class="flex items-center gap-3">
                <span class="text-xs text-gray-600 w-48 shrink-0 truncate" title="${escapeHtml(mo.conclusions[i])}">C${i + 1}: ${escapeHtml(mo.conclusions[i]).slice(0, 60)}...</span>
                <input type="number" min="0" max="${totalPremises}" step="1" class="coverage-input"
                    value="${val === '' || val == null ? '' : val}"
                    data-cidx="${i}"
                    oninput="onCoverageInput(this, ${i}, ${totalPremises})"
                    title="How many of the ${totalPremises} premises support conclusion ${i + 1}?" />
                <span class="text-xs text-gray-500 w-16 coverage-pct">${pct === null ? '&mdash;' : pct + '%'}</span>
            </div>`;
        }
        html += `</div>`;
    }
    html += `</div>`;

    cont.innerHTML = html;
}

// ---------------------------------------------------------------------------
// EVENT HANDLERS
// ---------------------------------------------------------------------------
function setMetric(key, value) {
    const ev = getModelEval(currentImageId, currentModel);
    ev[key] = value;
    renderCurrent();
    updateProgress();
    scheduleSave();
}

function clearMetric(key) {
    const ev = getModelEval(currentImageId, currentModel);
    delete ev[key];
    renderCurrent();
    updateProgress();
    scheduleSave();
}

function clearGroup(modelKey) {
    if (!confirm('Clear all metrics for this model-pair?')) return;
    delete evaluations[currentImageId][modelKey];
    renderCurrent();
    updateProgress();
    scheduleSave();
}

function onCoverageInput(input, cidx, totalPremises) {
    const ev = getModelEval(currentImageId, currentModel);
    const key = `premise_coverage_${cidx}`;
    if (input.value === '' || input.value === null) {
        delete ev[key];
        input.parentElement.querySelector('.coverage-pct').innerHTML = '&mdash;';
    } else {
        let num = Number(input.value);
        ev[key] = num;
        const pctSpan = input.parentElement.querySelector('.coverage-pct');
        if (totalPremises > 0 && !isNaN(num)) {
            num = Math.max(0, Math.min(num, totalPremises));
            pctSpan.textContent = Math.round((num / totalPremises) * 100) + '%';
        } else {
            pctSpan.innerHTML = '&mdash;';
        }
    }
    updateProgress();
    scheduleSave();
}

function onCommentChange(text) {
    const ev = getModelEval(currentImageId, currentModel);
    ev.comment = text;
    scheduleSave();
}

function navImage(dir) {
    const idx = DATA.images.findIndex(i => i.id === currentImageId);
    const next = idx + dir;
    if (next < 0 || next >= DATA.images.length) return;
    currentImageId = DATA.images[next].id;
    renderAll();
}

function updateModelDots() {
    const img = DATA.images.find(i => i.id === currentImageId);
    if (!img) return;
    const dots = document.getElementById('model-dots');
    dots.innerHTML = MODEL_KEYS.map(key => {
        const ev = evaluations[img.id] && evaluations[img.id][key];
        const rated = !!ev && Object.keys(ev).length > 0;
        return `<span title="${(DATA.models && DATA.models[key]) || key}" class="inline-block w-3 h-3 rounded-full ${rated ? 'bg-green-500' : 'bg-gray-400'}"></span>`;
    }).join('');
}

// ---------------------------------------------------------------------------
// PROGRESS METRICS
// ---------------------------------------------------------------------------
function isImageRated(imageId) {
    return !!evaluations[imageId] && Object.values(evaluations[imageId]).some(m => m && Object.keys(m).length > 0);
}

function isModelPairComplete(imageId, modelKey) {
    // A pair counts as complete when every non-comment key for the current model's
    // expected metrics is present AND all premise-coverage inputs are filled.
    const img = DATA.images.find(i => i.id === imageId);
    if (!img) return false;
    const ev = evaluations[imageId] && evaluations[imageId][modelKey];
    if (!ev) return false;
    const mo = img.models[modelKey] || {};
    const totalConclusions = (mo.conclusions || []).length;
    let expected = 0, filled = 0;
    for (const group of GROUP_ORDER) {
        for (const m of (DATA.metrics[group] || [])) {
            if (m.key === 'premise_coverage') {
                expected += totalConclusions;
                for (let i = 0; i < totalConclusions; i++) {
                    if (ev[`premise_coverage_${i}`] !== undefined && ev[`premise_coverage_${i}`] !== '') filled++;
                }
            } else {
                expected++;
                if (ev[m.key] !== undefined && ev[m.key] !== '') filled++;
            }
        }
    }
    if (expected === 0) return false;
    return filled >= expected;
}

function updateProgress() {
    if (!DATA) return;
    const totalImages = DATA.images.length;
    const totalPairs = totalImages * MODEL_KEYS.length;

    let ratedImages = 0;
    let completePairs = 0;
    DATA.images.forEach(img => {
        if (isImageRated(img.id)) ratedImages++;
        MODEL_KEYS.forEach(k => { if (isModelPairComplete(img.id, k)) completePairs++; });
    });

    document.getElementById('progress-full-text').textContent = `${ratedImages}/${totalImages}`;
    document.getElementById('progress-model-text').textContent = `${completePairs}/${totalPairs}`;
}

// ---------------------------------------------------------------------------
// INIT
// ---------------------------------------------------------------------------
async function init() {
    try {
        const [dataRes, evalRes] = await Promise.all([
            fetch('/api/data').then(r => r.json()),
            fetch('/api/evaluations').then(r => r.json()),
        ]);
        DATA = dataRes;
        evaluations = (evalRes && evalRes.evaluations) || {};
        currentModel = MODEL_KEYS[0];
        currentImageId = DATA.images.length ? DATA.images[0].id : null;
        renderAll();
        showSaveStatus(`Loaded ${DATA.images.length} images`);
    } catch (e) {
        document.body.innerHTML = `<div class="p-8 font-mono text-red-600">Failed to load data: ${escapeHtml(e.message)}<br><br>Ensure the Flask server is running: <code>python user_interfaces/eval_app.py</code></div>`;
    }
}

document.addEventListener('DOMContentLoaded', init);