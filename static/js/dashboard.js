/* ─────────────────────────────────────────────
   Smart Traffic Dashboard – 4-Lane Frontend Logic
   ───────────────────────────────────────────── */

const DIRECTIONS = ['North', 'East', 'South', 'West'];
const DIR_ICONS = { North: '🔴', East: '🔴', South: '🔴', West: '🔴' };
const DIR_COLORS = {
  North: '#6366f1', East: '#22c55e', South: '#f59e0b', West: '#ec4899'
};

// ── State ──
let jobId = null;
let laneFiles = { North: null, East: null, South: null, West: null };
let laneState = { North: 'idle', East: 'idle', South: 'idle', West: 'idle' };
let fullResult = null;
let logData = [];

// ── Charts ──
let vehicleChart = null;
let densityChart = null;
let typeChart = null;
let timeChart = null;

// ── DOM refs ──
const clockEl = document.getElementById('clock');
const lanesGrid = document.getElementById('lanesGrid');
const signalGrid = document.getElementById('signalGrid');
const logBody = document.getElementById('logBody');
const loader = document.getElementById('loader');
const loaderLabel = document.getElementById('loaderLabel');
const toast = document.getElementById('toast');
const processAllBtn = document.getElementById('processAllBtn');
const exportBtn = document.getElementById('exportBtn');
const outputsSection = document.getElementById('outputsSection');
const outputsGrid = document.getElementById('outputsGrid');


// ─────────────────────────────────────────────
// CLOCK
// ─────────────────────────────────────────────
function updateClock() {
  clockEl.textContent = new Date().toLocaleString('id-ID', {
    weekday: 'short', year: 'numeric', month: 'short',
    day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}
updateClock();
setInterval(updateClock, 1000);


// ─────────────────────────────────────────────
// TOAST
// ─────────────────────────────────────────────
function showToast(msg, type = 'success') {
  toast.textContent = msg;
  toast.className = `toast show ${type}`;
  setTimeout(() => { toast.className = 'toast'; }, 4500);
}


// ─────────────────────────────────────────────
// BUILD LANE SLOTS (on page load)
// ─────────────────────────────────────────────
function buildLaneSlots() {
  lanesGrid.innerHTML = DIRECTIONS.map(d => `
    <div class="lane-slot" id="slot-${d}">
      <div class="lane-slot-header">
        <div class="lane-slot-title" id="lane-title-${d}">🔴 ${d} Lane</div>
        <div class="lane-status-badge" id="badge-${d}">Idle</div>
      </div>

      <div class="lane-upload-area" id="uploadArea-${d}">
        <input type="file" id="fileInput-${d}" accept="video/*" />
        <div class="lane-upload-icon">🎬</div>
        <p>Drag &amp; drop or <span class="browse" id="browseBtn-${d}">browse</span></p>
        <p style="font-size:.7rem;margin-top:4px">MP4, AVI, MOV, etc.</p>
      </div>

      <div class="lane-filename" id="filename-${d}"></div>

      <!-- Original uploaded video preview -->
      <div class="lane-preview-wrap" id="origWrap-${d}">
        <video class="lane-video-preview" id="preview-${d}" controls></video>
        <div class="lane-placeholder" id="placeholder-${d}">
          <span class="ph-icon">📹</span>
          <span>${d} camera video</span>
        </div>
      </div>

      <div class="lane-progress" id="progress-${d}">
        <div class="lane-progress-bar" id="progressBar-${d}"></div>
      </div>

      <button class="lane-process-btn" id="processBtn-${d}" disabled>
        ⚙️ Process ${d}
      </button>

      <!-- Processed output video (hidden until done) -->
      <div class="lane-output-wrap" id="outputWrap-${d}" style="display:none">
        <div class="lane-output-label">🎯 Processed Output — Bounding Box Detection</div>
        <video class="lane-video-preview lane-output-video" id="outputVideo-${d}" controls></video>
        <a class="lane-download-btn" id="downloadBtn-${d}" download>⬇ Download Processed Video</a>
      </div>
    </div>
  `).join('');

  // Wire up events for each lane
  DIRECTIONS.forEach(d => {
    const input = document.getElementById(`fileInput-${d}`);
    const area = document.getElementById(`uploadArea-${d}`);
    const browseBtn = document.getElementById(`browseBtn-${d}`);
    const processBtn = document.getElementById(`processBtn-${d}`);

    browseBtn.addEventListener('click', e => { e.stopPropagation(); input.click(); });
    area.addEventListener('click', () => input.click());

    area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('dragging'); });
    area.addEventListener('dragleave', () => area.classList.remove('dragging'));
    area.addEventListener('drop', e => {
      e.preventDefault(); area.classList.remove('dragging');
      const file = e.dataTransfer.files[0];
      if (file) handleLaneFile(d, file);
    });

    input.addEventListener('change', () => {
      if (input.files[0]) handleLaneFile(d, input.files[0]);
    });

    processBtn.addEventListener('click', () => processLane(d));
  });
}


// ─────────────────────────────────────────────
// HANDLE FILE SELECTION per lane
// ─────────────────────────────────────────────
function handleLaneFile(dir, file) {
  if (!file.type.startsWith('video/')) {
    showToast(`❌ Please select a valid video for ${dir} lane.`, 'error');
    return;
  }

  laneFiles[dir] = file;
  const slot = document.getElementById(`slot-${dir}`);
  slot.classList.add('has-file');

  document.getElementById(`filename-${dir}`).textContent =
    `📁 ${file.name}  (${(file.size / 1024 / 1024).toFixed(1)} MB)`;

  const preview = document.getElementById(`preview-${dir}`);
  const ph = document.getElementById(`placeholder-${dir}`);
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  ph.style.display = 'none';

  document.getElementById(`processBtn-${dir}`).disabled = false;

  // Update "Lane Ready" counter
  updateLanesReadyCard();

  // Enable "Process All" if at least 1 lane has a file
  const anyFile = DIRECTIONS.some(d => laneFiles[d] !== null);
  processAllBtn.disabled = !anyFile;
}


function updateLanesReadyCard() {
  const ready = DIRECTIONS.filter(d => laneFiles[d] !== null).length;
  document.getElementById('lanesReady').textContent = `${ready} / 4`;
}


// ─────────────────────────────────────────────
// GET / CREATE JOB ID
// ─────────────────────────────────────────────
async function ensureJob() {
  if (jobId) return jobId;
  const resp = await fetch('/api/new-job', { method: 'POST' });
  const data = await resp.json();
  jobId = data.jobId;
  return jobId;
}


// ─────────────────────────────────────────────
// PROCESS SINGLE LANE
// ─────────────────────────────────────────────
async function processLane(dir) {
  const file = laneFiles[dir];
  if (!file) return;

  await ensureJob();

  setLaneState(dir, 'processing');

  const fd = new FormData();
  fd.append('video', file);
  fd.append('lane', dir);
  fd.append('jobId', jobId);

  try {
    const resp = await fetch('/api/upload-lane', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok || data.error) throw new Error(data.error || 'Server error');

    setLaneState(dir, 'done');
    showToast(`✅ ${dir} lane processed!`, 'success');

    // ── Show processed video inline in the lane slot ──
    if (data.outputVideo) {
      const outputWrap = document.getElementById(`outputWrap-${dir}`);
      const outputVid = document.getElementById(`outputVideo-${dir}`);
      const dlBtn = document.getElementById(`downloadBtn-${dir}`);
      outputVid.src = data.outputVideo + '?t=' + Date.now(); // cache-bust
      dlBtn.href = data.outputVideo;
      dlBtn.download = `processed_${dir}.mp4`;
      outputWrap.style.display = 'block';
      outputVid.load();
    }

    // Refresh aggregate results
    await fetchAndRenderResults();

  } catch (err) {
    setLaneState(dir, 'error');
    showToast(`❌ ${dir}: ${err.message}`, 'error');
    console.error(err);
  }
}


// ─────────────────────────────────────────────
// PROCESS ALL LANES (sequential)
// ─────────────────────────────────────────────
processAllBtn.addEventListener('click', async () => {
  const lanesToProcess = DIRECTIONS.filter(d =>
    laneFiles[d] !== null && laneState[d] !== 'done'
  );
  if (lanesToProcess.length === 0) {
    showToast('All uploaded lanes already processed.', 'info'); return;
  }

  processAllBtn.disabled = true;

  for (const d of lanesToProcess) {
    await processLane(d);
  }

  processAllBtn.disabled = false;
});


// ─────────────────────────────────────────────
// LANE STATE UI HELPER
// ─────────────────────────────────────────────
function setLaneState(dir, state) {
  laneState[dir] = state;
  const slot = document.getElementById(`slot-${dir}`);
  const badge = document.getElementById(`badge-${dir}`);
  const progress = document.getElementById(`progress-${dir}`);
  const btn = document.getElementById(`processBtn-${dir}`);

  slot.classList.remove('has-file', 'processing', 'done', 'error');
  badge.classList.remove('processing', 'done', 'error');
  progress.classList.remove('active', 'done');

  const labels = { idle: 'Idle', processing: 'Processing…', done: 'Done ✓', error: 'Error' };
  badge.textContent = labels[state] || state;

  if (state === 'processing') {
    slot.classList.add('processing');
    badge.classList.add('processing');
    progress.classList.add('active');
    btn.disabled = true;
    btn.textContent = `⏳ Processing…`;
  } else if (state === 'done') {
    slot.classList.add('done');
    badge.classList.add('done');
    progress.classList.add('done');
    btn.disabled = true;
    btn.textContent = `✅ Done`;
  } else if (state === 'error') {
    slot.classList.add('error');
    badge.classList.add('error');
    btn.disabled = false;
    btn.textContent = `🔄 Retry`;
  } else {
    if (laneFiles[dir]) slot.classList.add('has-file');
    btn.disabled = !laneFiles[dir];
    btn.textContent = `⚙️ Process ${dir}`;
  }
}


// ─────────────────────────────────────────────
// FETCH COMBINED RESULTS + RENDER
// ─────────────────────────────────────────────
async function fetchAndRenderResults() {
  if (!jobId) return;
  try {
    const resp = await fetch(`/api/results/${jobId}`);
    if (resp.status === 202) return;  // still pending
    if (!resp.ok) return;
    fullResult = await resp.json();
    renderDashboard(fullResult);
  } catch (e) {
    console.error('Results fetch error:', e);
  }
}


// ─────────────────────────────────────────────
// RENDER FULL DASHBOARD
// ─────────────────────────────────────────────
function renderDashboard(data) {
  const { counts, density, green, duration, totalVehicles, totalByType, decisionLog, outputVideos } = data;

  // ── Summary cards ──
  document.getElementById('totalVehicles').textContent = totalVehicles;
  document.getElementById('greenLane').textContent = `🟢 ${green}`;
  document.getElementById('greenTime').textContent = `${duration}s`;
  document.getElementById('peakDensity').textContent = density[green];

  // ── Vehicle type breakdown cards ──
  const setTypeCard = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.textContent !== String(value)) {
      el.textContent = value;
      el.classList.remove('pop');
      // Force reflow so the animation restarts cleanly
      void el.offsetWidth;
      el.classList.add('pop');
    }
  };
  if (totalByType) {
    setTypeCard('totalMotorcycle', totalByType.motorcycle ?? 0);
    setTypeCard('totalCar', totalByType.car ?? 0);
    setTypeCard('totalBus', totalByType.bus ?? 0);
    setTypeCard('totalTruck', totalByType.truck ?? 0);
  } else {
    // Fallback: sum from per-lane counts
    const sumType = t => DIRECTIONS.reduce((s, d) => s + (counts[d]?.[t] || 0), 0);
    setTypeCard('totalMotorcycle', sumType('motorcycle'));
    setTypeCard('totalCar', sumType('car'));
    setTypeCard('totalBus', sumType('bus'));
    setTypeCard('totalTruck', sumType('truck'));
  }

  // ── Update lane titles with duration ──
  DIRECTIONS.forEach(d => {
    const isGreen = d === green;
    const icon = isGreen ? '🟢' : '🔴';
    const titleEl = document.getElementById(`lane-title-${d}`);
    if (titleEl) {
      titleEl.textContent = `${icon} ${d} Lane — ${duration}s`;
    }
  });

  // ── Traffic lights ──
  updateTrafficLights(green);

  // ── Densities on intersection map ──
  DIRECTIONS.forEach(d => {
    document.getElementById(`density-${d}`).textContent =
      density[d] > 0 ? `🔢 ${density[d]}` : '—';
  });

  // ── Signal cards ──
  renderSignalCards(counts, density, green, duration);

  // ── Processed videos ──
  if (outputVideos && Object.keys(outputVideos).length > 0) {
    outputsSection.style.display = 'block';
    outputsGrid.innerHTML = DIRECTIONS
      .filter(d => outputVideos[d])
      .map(d => `
        <div class="output-box">
          <h4>${d === green ? '🟢' : '🔴'} ${d} Lane Output</h4>
          <video src="${outputVideos[d]}" controls></video>
        </div>`
      ).join('');
  }

  // ── Charts ──
  renderCharts(counts, density);

  // ── Decision Log ──
  logData = decisionLog || [];
  renderLog(logData);
  exportBtn.disabled = logData.length === 0;
}


// ─────────────────────────────────────────────
// TRAFFIC LIGHTS
// ─────────────────────────────────────────────
function updateTrafficLights(greenDir) {
  DIRECTIONS.forEach(d => {
    const isGreen = d === greenDir;
    document.getElementById(`red-${d}`).classList.toggle('active', !isGreen);
    document.getElementById(`yellow-${d}`).classList.remove('active');
    document.getElementById(`green-${d}`).classList.toggle('active', isGreen);
  });
}

// Init: all red
function initTrafficLights() {
  DIRECTIONS.forEach(d => {
    document.getElementById(`red-${d}`).classList.add('active');
  });
}


// ─────────────────────────────────────────────
// SIGNAL CARDS
// ─────────────────────────────────────────────
function renderSignalCards(counts, density, green, duration) {
  signalGrid.innerHTML = DIRECTIONS.map(d => {
    const c = counts[d];
    const total = c.motor + c.car + c.bus + c.truck;
    const isGreen = d === green;
    const barPct = val => Math.round((val / Math.max(total, 1)) * 100);

    return `
    <div class="signal-card ${isGreen ? 'active-green' : ''}">
      <div class="signal-header">
        <span class="signal-name">${isGreen ? '🟢' : '🔴'} ${d}</span>
        <div class="signal-light ${isGreen ? 'green' : 'red'}">●</div>
      </div>
      <div class="vehicle-rows">
        ${vRow('🛵', 'Motorcycle', c.motorcycle, barPct(c.motorcycle))}
        ${vRow('🚗', 'Car', c.car, barPct(c.car))}
        ${vRow('🚌', 'Bus', c.bus, barPct(c.bus))}
        ${vRow('🚛', 'Truck', c.truck, barPct(c.truck))}
      </div>
      <div class="density-row">
        <div>
          <div class="density-lbl">Density Score</div>
          <div class="density-val">${density[d]}</div>
        </div>
        <div style="text-align:right">
          <div class="density-lbl">Signal</div>
          <div style="font-size:.85rem;font-weight:700;color:${isGreen ? 'var(--green)' : 'var(--red)'}">
            ${isGreen ? `🟢 GREEN · ${duration}s` : `🔴 RED · ${duration}s`}
          </div>
        </div>
      </div>
    </div>`;
  }).join('');
}

function vRow(icon, label, count, pct) {
  return `
  <div class="v-row">
    <span class="v-label">${icon} ${label}</span>
    <div class="v-bar-wrap"><div class="v-bar" style="width:${pct}%"></div></div>
    <span class="v-count">${count}</span>
  </div>`;
}


// ─────────────────────────────────────────────
// CHARTS
// ─────────────────────────────────────────────
const CHART_BASE = {
  responsive: true,
  plugins: {
    legend: { labels: { color: '#7d8590', font: { family: 'Inter', size: 11 } } },
    tooltip: { mode: 'index', intersect: false }
  },
  scales: {
    x: { ticks: { color: '#7d8590' }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#7d8590' }, grid: { color: '#21262d' }, beginAtZero: true }
  }
};

function renderCharts(counts, density) {
  const labels = DIRECTIONS;
  const bgColors = Object.values(DIR_COLORS).map(c => c + 'cc');

  // ── 1. Stacked bar: vehicle count per lane ──
  if (vehicleChart) vehicleChart.destroy();
  vehicleChart = new Chart(document.getElementById('vehicleChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: '🛵 Motorcycle', data: labels.map(d => counts[d].motorcycle), backgroundColor: 'rgba(0,255,255,.7)' },
        { label: '🚗 Car', data: labels.map(d => counts[d].car), backgroundColor: 'rgba(255,200,0,.7)' },
        { label: '🚌 Bus', data: labels.map(d => counts[d].bus), backgroundColor: 'rgba(0,180,255,.7)' },
        { label: '🚛 Truck', data: labels.map(d => counts[d].truck), backgroundColor: 'rgba(255,100,0,.7)' },
      ]
    },
    options: {
      ...CHART_BASE,
      scales: {
        ...CHART_BASE.scales,
        x: { ...CHART_BASE.scales.x, stacked: true },
        y: { ...CHART_BASE.scales.y, stacked: true }
      }
    }
  });

  // ── 2. Doughnut: density distribution ──
  if (densityChart) densityChart.destroy();
  densityChart = new Chart(document.getElementById('densityChart'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        label: 'Density Score', data: labels.map(d => density[d]),
        backgroundColor: bgColors, borderColor: 'transparent'
      }]
    },
    options: {
      responsive: true, plugins: {
        legend: { position: 'right', labels: { color: '#7d8590', font: { family: 'Inter', size: 11 } } }
      }
    }
  });

  // ── 3. Radar: vehicle type breakdown per lane ──
  if (typeChart) typeChart.destroy();
  typeChart = new Chart(document.getElementById('typeChart'), {
    type: 'radar',
    data: {
      labels,
      datasets: [
        { label: 'Motorcycle', data: labels.map(d => counts[d].motorcycle), borderColor: '#00ffff', backgroundColor: 'rgba(0,255,255,.1)', pointBackgroundColor: '#00ffff' },
        { label: 'Car', data: labels.map(d => counts[d].car), borderColor: '#ffc800', backgroundColor: 'rgba(255,200,0,.1)', pointBackgroundColor: '#ffc800' },
        { label: 'Bus', data: labels.map(d => counts[d].bus), borderColor: '#00b4ff', backgroundColor: 'rgba(0,180,255,.1)', pointBackgroundColor: '#00b4ff' },
        { label: 'Truck', data: labels.map(d => counts[d].truck), borderColor: '#ff6400', backgroundColor: 'rgba(255,100,0,.1)', pointBackgroundColor: '#ff6400' },
      ]
    },
    options: {
      responsive: true,
      scales: {
        r: {
          ticks: { color: '#7d8590', backdropColor: 'transparent' }, grid: { color: '#21262d' },
          pointLabels: { color: '#e6edf3', font: { size: 12 } }
        }
      },
      plugins: { legend: { labels: { color: '#7d8590', font: { family: 'Inter', size: 11 } } } }
    }
  });

  // ── 4. Horizontal bar: green time allocation ──
  const durations = labels.map(d => {
    const dens = density[d];
    if (dens >= 40) return 60;
    if (dens >= 25) return 45;
    if (dens >= 12) return 30;
    if (dens > 0) return 20;
    return 10;
  });

  if (timeChart) timeChart.destroy();
  timeChart = new Chart(document.getElementById('timeChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Green Time (s)', data: durations,
        backgroundColor: bgColors, borderColor: 'transparent', borderRadius: 6
      }]
    },
    options: { ...CHART_BASE, indexAxis: 'y' }
  });
}


// ─────────────────────────────────────────────
// DECISION LOG TABLE
// ─────────────────────────────────────────────
function renderLog(log) {
  if (!log || log.length === 0) {
    logBody.innerHTML = `<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:24px">No log data yet.</td></tr>`;
    return;
  }
  logBody.innerHTML = log.map(row => {
    // New format: per-type counts + totalTracked field
    const hasTypes = row.motor !== undefined;
    if (hasTypes) {
      const tracked = row.totalTracked !== undefined
        ? `<td style="color:var(--green);font-weight:700">${row.totalTracked}</td>`
        : '<td>—</td>';
      return `<tr>
        <td style="color:var(--muted)">#${row.frame}</td>
        <td style="color:#00ffff">${row.motor || 0}</td>
        <td style="color:#ffc800">${row.car || 0}</td>
        <td style="color:#00b4ff">${row.bus || 0}</td>
        <td style="color:#ff6400">${row.truck || 0}</td>
        <td style="font-weight:600">${row.total || 0}</td>
        ${tracked}
      </tr>`;
    } else {
      const total = (row.North || 0) + (row.East || 0) + (row.South || 0) + (row.West || 0);
      return `<tr>
        <td style="color:var(--muted)">#${row.frame}</td>
        <td>${row.North || 0}</td>
        <td>${row.East || 0}</td>
        <td>${row.South || 0}</td>
        <td>${row.West || 0}</td>
        <td style="font-weight:600">${total}</td>
        <td>—</td>
      </tr>`;
    }
  }).join('');
}


// ─────────────────────────────────────────────
// CSV EXPORT
// ─────────────────────────────────────────────
exportBtn.addEventListener('click', () => {
  if (!logData.length) return;
  const hasTypes = logData[0].motor !== undefined;
  let header, rows;
  if (hasTypes) {
    header = 'Frame,Motorcycle,Car,Bus,Truck,Active,Tracked\n';
    rows = logData.map(r =>
      `${r.frame},${r.motorcycle || 0},${r.car || 0},${r.bus || 0},${r.truck || 0},${r.total || 0},${r.totalTracked || 0}`
    ).join('\n');
  } else {
    header = 'Frame,North,East,South,West,Total\n';
    rows = logData.map(r => {
      const t = (r.North || 0) + (r.East || 0) + (r.South || 0) + (r.West || 0);
      return `${r.frame},${r.North || 0},${r.East || 0},${r.South || 0},${r.West || 0},${t}`;
    }).join('\n');
  }

  const blob = new Blob([header + rows], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `traffic_log_${Date.now()}.csv`;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('📥 CSV exported!', 'success');
});


// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
(function init() {
  buildLaneSlots();
  initTrafficLights();

  // Placeholder signal cards
  signalGrid.innerHTML = DIRECTIONS.map(d => `
    <div class="signal-card">
      <div class="signal-header">
        <span class="signal-name">🔴 ${d}</span>
        <div class="signal-light red">●</div>
      </div>
      <div class="vehicle-rows">
        ${vRow('🛵', 'Motorcycle', 0, 0)} ${vRow('🚗', 'Car', 0, 0)}
        ${vRow('🚌', 'Bus', 0, 0)}   ${vRow('🚛', 'Truck', 0, 0)}
      </div>
      <div class="density-row">
        <div><div class="density-lbl">Density Score</div><div class="density-val">0</div></div>
        <div style="text-align:right"><div class="density-lbl">Signal</div>
          <div style="font-size:.85rem;font-weight:700;color:var(--muted)">Awaiting…</div>
        </div>
      </div>
    </div>`).join('');
})();
