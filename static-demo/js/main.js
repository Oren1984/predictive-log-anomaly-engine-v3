/* ═══════════════════════════════════════════
   Predictive Log Anomaly Engine V3 — Main JS
   Counter animations · Pipeline stepper
   Live simulation engine
═══════════════════════════════════════════ */

/* ─── Pipeline Step Definitions ──────────── */
const PIPELINE_STEPS = [
  {
    num: '01',
    tag: 'INPUT',
    tagClass: 'tag-input',
    title: 'Raw Log Input',
    description: 'A raw log line enters the system — unstructured text from any service. Timestamps, IPs, block IDs, error codes all mixed together with no schema.',
    detail: 'Source: HDFS namenode  ·  Format: syslog  ·  Volume: millions per hour',
    visual: `
      <div class="log-line-demo">
        <span class="log-ts">2024-03-15 08:47:33</span>&nbsp;
        <span class="log-lvl error">ERROR</span>&nbsp;
        <span class="log-svc">[namenode]</span>&nbsp;
        <span class="log-msg">Connection pool exhausted: 192.168.1.42:5432 after 30s timeout</span>
      </div>`
  },
  {
    num: '02',
    tag: 'PROCESS',
    tagClass: 'tag-process',
    title: 'Regex Normalization (9 steps)',
    description: 'Nine regex patterns replace dynamic values — IPs, timestamps, block IDs, ports, numbers — with canonical placeholders. Identical to training-time normalization.',
    detail: 'Steps: IP → &lt;IP&gt;  ·  timestamp → &lt;DATETIME&gt;  ·  port → &lt;NUM&gt;  ·  block ID → &lt;BLOCK&gt;',
    visual: `
      <div>
        <div class="log-line-demo" style="margin-bottom:12px;">
          <span class="replaced">&lt;DATETIME&gt;</span>&nbsp;
          <span class="log-lvl error">ERROR</span>&nbsp;
          <span class="log-svc">[namenode]</span>&nbsp;
          <span class="log-msg">Connection pool exhausted:&nbsp;</span><span class="replaced">&lt;IP&gt;</span><span class="log-msg">:</span><span class="replaced">&lt;NUM&gt;</span><span class="log-msg"> after&nbsp;</span><span class="replaced">&lt;NUM&gt;</span><span class="log-msg">s timeout</span>
        </div>
        <div class="regex-rules">
          <span class="rule">\\d+\\.\\d+\\.\\d+\\.\\d+ → &lt;IP&gt;</span>
          <span class="rule">\\d{4}-\\d{2}-\\d{2} → &lt;DATETIME&gt;</span>
          <span class="rule">:\\d{1,5} → :&lt;NUM&gt;</span>
          <span class="rule">\\d+s → &lt;NUM&gt;s</span>
        </div>
      </div>`
  },
  {
    num: '03',
    tag: 'PROCESS',
    tagClass: 'tag-process',
    title: 'Template Mining (Drain3)',
    description: 'The normalized log is matched against a vocabulary of known templates. Unknown patterns are mined on-the-fly and added. Template ID is a compact behavioral identifier.',
    detail: 'Templates: ~8,000 patterns  ·  Vocabulary: templates.csv  ·  token_id = template_id + 2',
    visual: `
      <div style="width:100%">
        <div class="template-table">
          <div class="tbl-row header"><span>ID</span><span>Template Pattern</span><span>Count</span></div>
          <div class="tbl-row"><span style="color:var(--text-dim)">46</span><span>Block replicated to &lt;*&gt;</span><span>1,247</span></div>
          <div class="tbl-row matched">
            <span class="highlight">→ 47</span>
            <span>Connection pool exhausted: &lt;*&gt;</span>
            <span>83</span>
          </div>
          <div class="tbl-row"><span style="color:var(--text-dim)">48</span><span>Heartbeat received from &lt;*&gt;</span><span>9,122</span></div>
        </div>
        <div class="token-result">token_id = 49&nbsp;&nbsp;(template_id 47 + offset 2)</div>
      </div>`
  },
  {
    num: '04',
    tag: 'ML',
    tagClass: 'tag-ml',
    title: 'Word2Vec Embedding',
    description: 'Token ID 49 is looked up in the Word2Vec model to produce a 100-dimensional semantic vector. The model was trained on token-ID sequences from HDFS and BGL datasets.',
    detail: 'Model: gensim KeyedVectors  ·  Dims: 100  ·  Lookup: wv[str(49)] → float32[100]',
    visual: `
      <div class="vector-viz">
        <div class="vec-label">token_id[49] → float32[100]</div>
        <div class="vec-bars">
          ${[35,62,18,80,45,29,71,54,38,90,22,67,41,58,75,33,86,47,63,28,
             55,39,72,84,19,61,43,77,31,65,88,26,50,70,37,82,24,57,69,44]
            .map(h => `<div class="vec-bar" style="height:${h}%"></div>`).join('')}
        </div>
        <div class="vec-dim" style="font-family:var(--font-mono);font-size:9px;color:var(--text-dim);margin-top:4px;text-align:center;">dim[0 … 99]</div>
      </div>`
  },
  {
    num: '05',
    tag: 'PROCESS',
    tagClass: 'tag-process',
    title: 'Sequence Buffer (Rolling Window)',
    description: 'Each embedding is appended to a per-stream rolling buffer. When 10 embeddings accumulate, the window is emitted to the behavior model. Separate buffer per service/session.',
    detail: 'window_size = 10  ·  stride = 1  ·  per-stream LRU cache (max 5,000 streams)',
    visual: `
      <div class="seq-buffer">
        <div class="seq-label">Rolling Window (window_size = 10)</div>
        <div class="seq-blocks">
          ${[1,2,3,4,5,6,7,8,9].map((i) => `<div class="seq-block" title="event ${i}">${i}</div>`).join('')}
          <div class="seq-block new" title="new event">10</div>
        </div>
        <div class="seq-emit">↓ Window full — emitting to BiLSTM</div>
      </div>`
  },
  {
    num: '06',
    tag: 'ML',
    tagClass: 'tag-ml',
    title: 'BiLSTM Behavior Model',
    description: 'A 2-layer bidirectional LSTM processes the 10-step embedding sequence and outputs a 128-dimensional behavioral context vector — a compact summary of what the system was doing.',
    detail: 'Input: [1, 10, 100]  ·  Architecture: 2-layer BiLSTM  ·  Output: [1, 128] context vector',
    visual: `
      <div class="tensor-flow">
        <div class="tensor">
          <span class="t-label">INPUT</span>
          <span class="t-shape">[1, 10, 100]</span>
        </div>
        <span class="t-arrow">→</span>
        <div class="t-model">
          <span class="t-name">BiLSTM</span>
          <span class="t-detail">2 layers · hidden=128</span>
          <span class="t-detail">bidirectional</span>
        </div>
        <span class="t-arrow">→</span>
        <div class="tensor">
          <span class="t-label">CONTEXT</span>
          <span class="t-shape">[1, 128]</span>
        </div>
      </div>`
  },
  {
    num: '07',
    tag: 'ML',
    tagClass: 'tag-ml',
    title: 'Denoising Autoencoder (Anomaly Score)',
    description: 'The context vector is fed through an autoencoder trained only on normal behavior. It tries to reconstruct the input. High reconstruction error signals a behavioral anomaly.',
    detail: 'Encoder: 128→64→32  ·  Decoder: 32→64→128  ·  Score: MSE reconstruction error',
    visual: `
      <div>
        <div class="ae-diagram">
          <div class="ae-block">Encoder<br/>128→64→32</div>
          <div style="color:var(--text-dim);font-size:1rem;">→</div>
          <div class="ae-latent">latent<br/>[32]</div>
          <div style="color:var(--text-dim);font-size:1rem;">→</div>
          <div class="ae-block">Decoder<br/>32→64→128</div>
        </div>
        <div class="ae-score" style="margin-top:14px;">
          <span class="score-label">Reconstruction Error: </span>
          <span class="score-value anomaly">0.847</span>
          <span class="score-threshold">(threshold: 0.450)</span>
        </div>
        <div style="text-align:center;margin-top:8px;font-family:var(--font-mono);font-size:11px;color:var(--critical);font-weight:700;">
          ⚠ ANOMALY DETECTED — score exceeds threshold by 1.88×
        </div>
      </div>`
  },
  {
    num: '08',
    tag: 'OUTPUT',
    tagClass: 'tag-output',
    title: 'Severity Classification & Alert',
    description: 'The Severity MLP classifies the anomaly using the latent vector [32] + reconstruction error [1]. The AlertManager applies cooldown deduplication and dispatches to the outbox.',
    detail: 'MLP input: [33]  ·  V2 classifier labels: info / warning / critical  ·  V1 AlertPolicy labels: critical / high / medium / low',
    visual: `
      <div class="alert-card-demo critical">
        <div class="alert-header">
          <span class="severity-badge critical">CRITICAL</span>
          <span class="alert-time">08:47:33</span>
        </div>
        <div class="alert-body">
          <div>Score:&nbsp;<span class="val">0.847</span></div>
          <div>Confidence:&nbsp;<span class="val">94%</span></div>
          <div>Threshold:&nbsp;<span class="val">0.450</span></div>
          <div>Stream:&nbsp;<span class="val mono">namenode:session-12</span></div>
          <div>Output:&nbsp;<span class="val">artifacts/n8n_outbox/&lt;uuid&gt;.json</span></div>
        </div>
      </div>`
  },
  {
    num: '09',
    tag: 'V3 — OPTIONAL',
    tagClass: 'tag-ml',
    title: 'V3 Semantic Enrichment',
    description: 'When SEMANTIC_ENABLED=true, the V3 semantic layer enriches confirmed alerts with a natural-language explanation, evidence tokens, and cosine similarity against historical alerts — all attached to the alert response.',
    detail: 'Model: all-MiniLM-L6-v2 (sentence-transformers, ~90 MB)  ·  CPU by default, GPU-capable  ·  No overhead when disabled',
    visual: `
      <div class="alert-card-demo" style="border-color:rgba(0,212,255,0.4);">
        <div class="alert-header">
          <span class="severity-badge" style="background:rgba(0,212,255,0.18);color:#00d4ff;">V3 ENRICHED</span>
          <span class="alert-time">SEMANTIC_ENABLED=true</span>
        </div>
        <div class="alert-body">
          <div>explanation:&nbsp;<span class="val">"Anomalous pattern: high error density, 4 distinct templates"</span></div>
          <div>evidence_tokens:&nbsp;<span class="val mono">["PacketResponder: Exception", "BLOCK* NameSystem"]</span></div>
          <div>semantic_similarity:&nbsp;<span class="val">0.923</span></div>
          <div>top_similar_events:&nbsp;<span class="val mono">[{score: 0.923, ...}]</span></div>
        </div>
        <div style="margin-top:8px;font-size:10px;color:var(--text-dim);font-style:italic;">Fields are null when SEMANTIC_ENABLED=false (default)</div>
      </div>`
  }
];

/* ─── Simulation Data ─────────────────────── */
const NORMAL_LOGS = [
  { ts: '08:42:01', lvl: 'INFO',  svc: '[namenode]',  msg: 'Block blk_1073745340 replicated to /10.251.43.21:50010' },
  { ts: '08:42:02', lvl: 'INFO',  svc: '[datanode]',  msg: 'Receiving block blk_1073745340 src: /10.251.42.11:54106' },
  { ts: '08:42:03', lvl: 'DEBUG', svc: '[pipeline]',  msg: 'Processing event batch size=128 latency=0.8ms' },
  { ts: '08:42:04', lvl: 'INFO',  svc: '[namenode]',  msg: 'BLOCK* NameSystem.addStoredBlock: blk_1073745340 added' },
  { ts: '08:42:05', lvl: 'INFO',  svc: '[datanode]',  msg: 'PacketResponder 0 for block blk_1073745340 terminating' },
  { ts: '08:42:06', lvl: 'INFO',  svc: '[namenode]',  msg: '10.251.43.21:50010 Starting upload of blk_1073745341' },
  { ts: '08:42:07', lvl: 'INFO',  svc: '[datanode]',  msg: 'Served block blk_1073745341 to /10.251.44.12:50010' },
  { ts: '08:42:08', lvl: 'INFO',  svc: '[namenode]',  msg: 'Replication monitor: 0 blocks over-replicated, 0 under' },
  { ts: '08:42:09', lvl: 'INFO',  svc: '[scheduler]', msg: 'Heartbeat received from node-07 at 10.251.42.07:50020' },
  { ts: '08:42:10', lvl: 'INFO',  svc: '[namenode]',  msg: 'Block report processed from datanode:50010, 8,230 blocks' },
  { ts: '08:42:11', lvl: 'DEBUG', svc: '[pipeline]',  msg: 'Window emitted: stream=namenode:12, latency=0.6ms' },
  { ts: '08:42:12', lvl: 'INFO',  svc: '[namenode]',  msg: 'Lease expiry check: 0 leases expired' },
  { ts: '08:42:13', lvl: 'INFO',  svc: '[datanode]',  msg: 'Block blk_1073745342 verified: checksum OK' },
  { ts: '08:42:14', lvl: 'INFO',  svc: '[namenode]',  msg: 'Safe mode is OFF. 8,312 blocks are processed' },
  { ts: '08:42:15', lvl: 'DEBUG', svc: '[pipeline]',  msg: 'Anomaly score: 0.21 (below threshold 0.45) — nominal' },
  { ts: '08:42:16', lvl: 'INFO',  svc: '[scheduler]', msg: 'Task allocation: 14 map, 3 reduce slots available' },
  { ts: '08:42:17', lvl: 'INFO',  svc: '[namenode]',  msg: 'Block blk_1073745343 delete received from /10.251.43.22' },
  { ts: '08:42:18', lvl: 'DEBUG', svc: '[pipeline]',  msg: 'Anomaly score: 0.18 (below threshold 0.45) — nominal' },
  { ts: '08:42:19', lvl: 'INFO',  svc: '[namenode]',  msg: 'Successfully completed replication of blk_1073745340' },
  { ts: '08:42:20', lvl: 'INFO',  svc: '[datanode]',  msg: 'DataNode blocks total: 8,312, missing: 0, corrupt: 0' },
];

const ANOMALY_LOGS = [
  { ts: '08:47:33', lvl: 'ERROR', svc: '[namenode]',  msg: 'Connection pool exhausted: 192.168.1.42:5432 after 30s' },
  { ts: '08:47:34', lvl: 'ERROR', svc: '[namenode]',  msg: 'Failed to place replication: insufficient nodes (need 3, got 1)' },
  { ts: '08:47:35', lvl: 'WARN',  svc: '[datanode]',  msg: 'Slow disk I/O detected: write latency 4200ms on /data/disk1' },
  { ts: '08:47:36', lvl: 'ERROR', svc: '[pipeline]',  msg: 'Unusual token sequence: deviation score 0.847 — ANOMALY' },
  { ts: '08:47:37', lvl: 'FATAL', svc: '[namenode]',  msg: 'Block replication failed after 3 retries: blk_1073745399' },
  { ts: '08:47:38', lvl: 'ERROR', svc: '[scheduler]', msg: 'Node 10.251.42.07 unresponsive: 5 missed heartbeats' },
  { ts: '08:47:39', lvl: 'WARN',  svc: '[datanode]',  msg: 'Checksum mismatch on blk_1073745400, requesting re-read' },
  { ts: '08:47:40', lvl: 'ERROR', svc: '[namenode]',  msg: 'OutOfMemoryError in BlockManager during replication storm' },
];

const ALERT_TEMPLATES = [
  { severity: 'critical', score: 0.847, confidence: 0.94, stream: 'namenode:session-12', msg: 'Connection pool exhausted' },
  { severity: 'warning',  score: 0.621, confidence: 0.82, stream: 'datanode:session-08',  msg: 'Disk I/O latency spike' },
  { severity: 'critical', score: 0.912, confidence: 0.97, stream: 'namenode:session-12', msg: 'Replication failure storm' },
  { severity: 'warning',  score: 0.558, confidence: 0.76, stream: 'scheduler:session-03', msg: 'Node heartbeat missed' },
  { severity: 'critical', score: 0.783, confidence: 0.91, stream: 'namenode:session-12', msg: 'BlockManager memory error' },
  { severity: 'info',     score: 0.481, confidence: 0.68, stream: 'datanode:session-05',  msg: 'Checksum mismatch detected' },
];

/* ─── Utility ─────────────────────────────── */
function $(id) { return document.getElementById(id); }
function qs(sel, parent) { return (parent || document).querySelector(sel); }
function qsAll(sel, parent) { return (parent || document).querySelectorAll(sel); }

/* ─── Safe module runner ────────────────────
   Isolates each section so a failure in one
   (e.g. pipeline) doesn't kill the others.
─────────────────────────────────────────── */
function _run(name, fn) {
  try {
    fn();
  } catch (e) {
    console.error('[LogAnomalyEngine] Module "' + name + '" failed to initialize.', e);
  }
}

/* ─── Nav: scroll + active section ──────────── */
_run('nav', function() {
  const navbar = $('navbar');
  const links = qsAll('.nav-links a');
  const sections = qsAll('section[id]');

  window.addEventListener('scroll', () => {
    // Solid background after hero
    navbar.style.background = window.scrollY > 60
      ? 'rgba(10, 14, 26, 0.97)'
      : 'rgba(10, 14, 26, 0.85)';

    // Active link highlighting
    let current = '';
    sections.forEach(sec => {
      if (window.scrollY >= sec.offsetTop - 100) current = sec.id;
    });
    links.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  }, { passive: true });
});

/* ─── Animated Counters ─────────────────────── */
_run('counters', function() {
  const counters = qsAll('.counter-val[data-target]');
  const seen = new Set();

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !seen.has(entry.target)) {
        seen.add(entry.target);
        animateCounter(entry.target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => observer.observe(el));

  function animateCounter(el) {
    const target = parseInt(el.dataset.target, 10);
    const duration = 1400;
    const start = performance.now();

    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      el.textContent = Math.round(eased * target);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
});

/* ─── Scroll Fade-in ─────────────────────────── */
_run('fadeIns', function() {
  const els = qsAll('section');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.querySelectorAll('.callout-item, .result-card, .stack-category').forEach((el, i) => {
          setTimeout(() => el.classList.add('visible'), i * 80);
        });
      }
    });
  }, { threshold: 0.1 });
  els.forEach(s => observer.observe(s));
});

/* ─── Pipeline Stepper ───────────────────────── */
_run('pipeline', function() {
  let currentStep = 0;
  let autoplayTimer = null;
  const AUTOPLAY_INTERVAL = 4000;

  const card     = $('pipeline-card');
  const progress = $('pipeline-progress');
  const counter  = $('step-counter');
  const prevBtn  = $('prev-step');
  const nextBtn  = $('next-step');
  const autoBtn  = $('autoplay-btn');

  // Build progress dots
  PIPELINE_STEPS.forEach((_, i) => {
    const dot = document.createElement('div');
    dot.className = 'progress-dot';
    dot.title = PIPELINE_STEPS[i].title;
    dot.addEventListener('click', () => goToStep(i));
    progress.appendChild(dot);
  });

  function renderStep(i) {
    const step = PIPELINE_STEPS[i];
    card.innerHTML = `
      <div class="step-header">
        <span class="step-num">STEP ${step.num}</span>
        <span class="step-tag ${step.tagClass}">${step.tag}</span>
        <h3 class="step-title">${step.title}</h3>
      </div>
      <div class="step-body">
        <div>
          <p class="step-description">${step.description}</p>
          <div class="step-detail">${step.detail}</div>
        </div>
        <div class="step-visual-wrap">${step.visual}</div>
      </div>`;

    // Progress dots
    const dots = qsAll('.progress-dot', progress);
    dots.forEach((dot, j) => {
      dot.classList.remove('active', 'done');
      if (j === i) dot.classList.add('active');
      else if (j < i) dot.classList.add('done');
    });

    // Buttons
    prevBtn.disabled = i === 0;
    nextBtn.disabled = i === PIPELINE_STEPS.length - 1;
    counter.textContent = `Step ${i + 1} of ${PIPELINE_STEPS.length}`;
  }

  function goToStep(i) {
    currentStep = i;
    renderStep(currentStep);
  }

  prevBtn.addEventListener('click', () => {
    if (currentStep > 0) goToStep(currentStep - 1);
  });
  nextBtn.addEventListener('click', () => {
    if (currentStep < PIPELINE_STEPS.length - 1) goToStep(currentStep + 1);
    else if (autoplayTimer) stopAutoplay();
  });

  function startAutoplay() {
    autoBtn.textContent = '⏸ Auto-playing...';
    autoBtn.classList.add('playing');
    autoplayTimer = setInterval(() => {
      if (currentStep < PIPELINE_STEPS.length - 1) {
        goToStep(currentStep + 1);
      } else {
        goToStep(0);
      }
    }, AUTOPLAY_INTERVAL);
  }

  function stopAutoplay() {
    clearInterval(autoplayTimer);
    autoplayTimer = null;
    autoBtn.textContent = '⏵ Auto-play';
    autoBtn.classList.remove('playing');
  }

  autoBtn.addEventListener('click', () => {
    if (autoplayTimer) stopAutoplay();
    else startAutoplay();
  });

  // Init
  renderStep(0);
});

/* ─── Live Simulation ─────────────────────────── */
_run('simulation', function() {
  const logFeed    = $('log-feed');
  const alertFeed  = $('alert-feed');
  const alertEmpty = $('alert-feed-empty');
  const mEvents    = $('m-events');
  const mWindows   = $('m-windows');
  const mAnomalies = $('m-anomalies');
  const mAlerts    = $('m-alerts');
  const btnAttack  = $('btn-attack');
  const btnPause   = $('btn-pause');
  const btnReset   = $('btn-reset');

  const MAX_LOG_ENTRIES   = 30;
  const MAX_ALERT_ENTRIES = 8;
  const LOG_INTERVAL_MS   = 1100;
  const ATTACK_BURST      = 6;

  let state = {
    events: 0, windows: 0, anomalies: 0, alerts: 0,
    logIdx: 0, alertIdx: 0,
    paused: false,
    timer: null,
    alertCards: [],
  };

  // Mixed log sequence: mostly normal, anomaly injected every ~20 logs
  function buildMixedSequence() {
    const seq = [];
    let ai = 0;
    for (let i = 0; i < 200; i++) {
      if (i > 0 && i % 20 === 0 && ai < ANOMALY_LOGS.length) {
        seq.push({ ...ANOMALY_LOGS[ai++], isAnomaly: true });
      } else {
        seq.push({ ...NORMAL_LOGS[i % NORMAL_LOGS.length], isAnomaly: false });
      }
    }
    return seq;
  }

  let logSequence = buildMixedSequence();

  function levelClass(lvl) {
    const map = { INFO: 'lvl-INFO', DEBUG: 'lvl-DEBUG', WARN: 'lvl-WARN', ERROR: 'lvl-ERROR', FATAL: 'lvl-FATAL' };
    return map[lvl] || '';
  }

  function addLogEntry(entry) {
    const div = document.createElement('div');
    div.className = 'log-entry' + (entry.isAnomaly ? ' anomaly' : (entry.lvl === 'WARN' ? ' warning-log' : ''));
    div.innerHTML = `<span class="log-ts">${entry.ts}</span> <span class="log-lvl ${levelClass(entry.lvl)}">${entry.lvl.padEnd(5)}</span> <span class="log-svc-inline">${entry.svc}</span> ${entry.msg}`;
    logFeed.appendChild(div);

    // Trim
    while (logFeed.children.length > MAX_LOG_ENTRIES) {
      logFeed.removeChild(logFeed.firstChild);
    }
    logFeed.scrollTop = logFeed.scrollHeight;
  }

  function addAlertCard(tpl, ts) {
    if (alertEmpty) alertEmpty.style.display = 'none';
    const now = ts || new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const sac = tpl.severity;
    const scoreColor = sac === 'critical' ? 'crit' : sac === 'warning' ? 'warn' : 'info-col';

    const card = document.createElement('div');
    card.className = `sim-alert-card sac-${sac}`;
    card.innerHTML = `
      <div class="sac-header">
        <span class="severity-badge ${sac}">${sac.toUpperCase()}</span>
        <span class="alert-time">${now}</span>
      </div>
      <div class="sac-body">
        <div class="sac-field"><span class="sac-key">Score</span><span class="sac-val ${scoreColor}">${tpl.score.toFixed(3)}</span></div>
        <div class="sac-field"><span class="sac-key">Confidence</span><span class="sac-val">${Math.round(tpl.confidence * 100)}%</span></div>
        <div class="sac-field"><span class="sac-key">Stream</span><span class="sac-val">${tpl.stream}</span></div>
        <div class="sac-field"><span class="sac-key">Message</span><span class="sac-val">${tpl.msg}</span></div>
      </div>`;

    alertFeed.insertBefore(card, alertFeed.firstChild);
    state.alertCards.push(card);

    // Trim alert list
    while (state.alertCards.length > MAX_ALERT_ENTRIES) {
      const old = state.alertCards.shift();
      if (old.parentNode) old.parentNode.removeChild(old);
    }
  }

  function updateMetric(el, val) {
    el.textContent = val;
    el.style.transform = 'scale(1.15)';
    el.style.transition = 'transform 0.15s';
    setTimeout(() => { el.style.transform = ''; }, 150);
  }

  function tick(entry) {
    if (state.paused) return;
    addLogEntry(entry);
    state.events++;
    updateMetric(mEvents, state.events);

    // Emit a window every 10 events
    if (state.events % 10 === 0) {
      state.windows++;
      updateMetric(mWindows, state.windows);
    }

    if (entry.isAnomaly) {
      state.anomalies++;
      updateMetric(mAnomalies, state.anomalies);

      // Fire alert card after short delay
      setTimeout(() => {
        const tpl = ALERT_TEMPLATES[state.alertIdx % ALERT_TEMPLATES.length];
        state.alertIdx++;
        addAlertCard(tpl);
        state.alerts++;
        updateMetric(mAlerts, state.alerts);
      }, 600);
    }
  }

  function startTimer() {
    state.timer = setInterval(() => {
      const entry = logSequence[state.logIdx % logSequence.length];
      state.logIdx++;
      tick(entry);
    }, LOG_INTERVAL_MS);
  }

  function stopTimer() {
    clearInterval(state.timer);
    state.timer = null;
  }

  function resetSim() {
    stopTimer();
    logFeed.innerHTML = '';
    alertFeed.innerHTML = '';
    if (alertEmpty) {
      alertEmpty.style.display = '';
      alertFeed.appendChild(alertEmpty);
    }
    state = { events: 0, windows: 0, anomalies: 0, alerts: 0,
      logIdx: 0, alertIdx: 0, paused: false, timer: null, alertCards: [] };
    mEvents.textContent = '0';
    mWindows.textContent = '0';
    mAnomalies.textContent = '0';
    mAlerts.textContent = '0';
    btnPause.textContent = '⏸ Pause';
    startTimer();
  }

  // Simulate Attack: inject burst of anomalous logs
  btnAttack.addEventListener('click', () => {
    let i = 0;
    const burst = () => {
      if (i >= ATTACK_BURST || state.paused) return;
      const entry = { ...ANOMALY_LOGS[i % ANOMALY_LOGS.length], isAnomaly: true };
      tick(entry);
      i++;
      setTimeout(burst, 380);
    };
    burst();
  });

  btnPause.addEventListener('click', () => {
    state.paused = !state.paused;
    btnPause.textContent = state.paused ? '▶ Resume' : '⏸ Pause';
    if (state.paused) {
      stopTimer();
    } else {
      startTimer();
    }
  });

  btnReset.addEventListener('click', resetSim);

  // Auto-start when section enters viewport
  const simSection = document.querySelector('#simulation');
  if (!simSection) { console.warn('[LogAnomalyEngine] #simulation not found'); return; }
  let started = false;
  const simObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !started) {
      started = true;
      startTimer();
    }
  }, { threshold: 0.2 });
  simObserver.observe(simSection);
});
