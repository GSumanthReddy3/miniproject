'use strict';

/* ─── Utility helpers ───────────────────────────────────────────────────────*/
const $ = id => document.getElementById(id);
const esc = str => { const d = document.createElement('div'); d.textContent = str; return d.innerHTML; };

/* ─── Navbar scroll shadow ───────────────────────────────────────────────────*/
window.addEventListener('scroll', () => {
  const n = $('navbar');
  if (n) n.classList.toggle('scrolled', window.scrollY > 20);
});

/* ═══════════════════════════════════════════════════════════════════════════
   INDEX PAGE  (class="page-index" on <body>)
   ═══════════════════════════════════════════════════════════════════════════*/
if (document.body.classList.contains('page-index')) {

  /* ── Session check ─────────────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', () => {
    checkSession();
    setupFileInput();
    setupFileDrop();
    setupFormBtnSpinner();
  });

  async function checkSession() {
    try {
      const d = await fetch('/session-status').then(r => r.json());
      updateAuthUI(d.logged_in, d.username);
    } catch (_) {}
  }

  function updateAuthUI(loggedIn, username) {
    const status  = $('auth-status');
    const btnAuth = $('btn-auth');
    const btnHist = $('btn-history');
    if (loggedIn) {
      status.innerHTML = `<span>👤</span><span>${esc(username)}</span>`;
      btnAuth.textContent = 'Sign Out';
      btnAuth.onclick = () => { window.location.href = '/logout'; };
      if (btnHist) btnHist.style.display = '';
    } else {
      status.innerHTML = '';
      btnAuth.textContent = 'Sign In';
      btnAuth.onclick = openAuthModal;
      if (btnHist) btnHist.style.display = 'none';
    }
  }

  /* ── Tabs ─────────────────────────────────────────────────────────────── */
  window.switchTab = function(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    $(`tab-${tab}`).classList.add('active');
    $(`content-${tab}`).classList.add('active');
    $('input_type').value = tab;
  };

  /* ── File input display ────────────────────────────────────────────────── */
  function setupFileInput() {
    const fi = $('csv_file');
    if (!fi) return;
    fi.addEventListener('change', () => {
      const d = $('file-name-display');
      if (fi.files.length) {
        d.style.display = 'flex';
        d.innerHTML = `✅ ${esc(fi.files[0].name)} (${fmtBytes(fi.files[0].size)})`;
      } else { d.style.display = 'none'; }
    });
  }

  /* ── Drag & drop ───────────────────────────────────────────────────────── */
  function setupFileDrop() {
    const zone = $('file-drop-zone');
    if (!zone) return;
    ['dragenter','dragover'].forEach(e =>
      zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add('drag-over'); }));
    ['dragleave','drop'].forEach(e =>
      zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove('drag-over'); }));
    zone.addEventListener('drop', ev => {
      const f = ev.dataTransfer.files[0];
      if (f && f.name.endsWith('.csv')) {
        const dt = new DataTransfer(); dt.items.add(f);
        $('csv_file').files = dt.files;
        $('csv_file').dispatchEvent(new Event('change'));
      } else {
        showToast('Please drop a .csv file', 'error');
      }
    });
  }

  /* ── Show spinner on form submit ───────────────────────────────────────── */
  function setupFormBtnSpinner() {
    const form = $('analyze-form');
    if (!form) return;
    form.addEventListener('submit', () => {
      const spinner = $('btn-spinner');
      const btnIcon = document.querySelector('#btn-analyze .btn-icon');
      if (spinner) spinner.style.display = 'inline';
      if (btnIcon) btnIcon.style.display = 'none';
      $('btn-analyze').disabled = true;
    });
  }

  /* ── History ───────────────────────────────────────────────────────────── */
  window.loadHistory = async function() {
    const panel = $('history-panel');
    const list  = $('history-list');
    if (!panel) return;
    panel.style.display = '';
    list.innerHTML = '<p style="color:var(--text-3)">Loading…</p>';
    panel.scrollIntoView({ behavior: 'smooth', block:'start' });
    try {
      const data = await fetch('/history').then(r => r.json());
      if (data.error) { list.innerHTML = `<p style="color:var(--red-l)">${esc(data.error)}</p>`; return; }
      if (!data.length) { list.innerHTML = '<p style="color:var(--text-3)">No history yet.</p>'; return; }
      list.innerHTML = data.map(a => `
        <div class="history-item">
          <div class="history-meta">
            <div style="font-weight:600">${esc(a.input_type.toUpperCase())} Analysis — ${a.total} reviews</div>
            <div class="history-time">${esc(a.created_at)}</div>
            <div class="history-stats">
              <span class="history-tag" style="color:var(--green-l)">✅ ${a.genuine_count} genuine</span>
              <span class="history-tag" style="color:var(--red-l)">🚫 ${a.fake_count} fake</span>
            </div>
          </div>
          <a href="/results/${a.id}" class="history-link btn btn-outline-sm">View →</a>
        </div>`).join('');
    } catch (_) { list.innerHTML = '<p style="color:var(--red-l)">Failed to load history.</p>'; }
  };

  window.closeHistory = () => { $('history-panel').style.display = 'none'; };

  /* ── Auth Modal ────────────────────────────────────────────────────────── */
  window.openAuthModal  = () => { $('auth-modal').style.display = 'flex'; document.body.style.overflow = 'hidden'; };
  window.closeAuthModal = (e) => {
    if (e && e.target !== $('auth-modal')) return;
    $('auth-modal').style.display = 'none'; document.body.style.overflow = '';
  };
  window.switchAuthTab = (tab) => {
    document.querySelectorAll('.mtab').forEach(t => t.classList.remove('active'));
    $(`mtab-${tab}`).classList.add('active');
    $('login-form').style.display    = tab === 'login'    ? '' : 'none';
    $('register-form').style.display = tab === 'register' ? '' : 'none';
    $('login-error').style.display = $('reg-error').style.display = 'none';
  };

  window.handleLogin = async (e) => {
    e.preventDefault();
    $('login-error').style.display = 'none';
    const u = $('login-username').value.trim();
    const p = $('login-password').value;
    try {
      const d = await fetch('/login', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ username: u, password: p })
      }).then(r => r.json());
      if (d.error) { $('login-error').style.display=''; $('login-error').textContent = d.error; return; }
      window.closeAuthModal();
      updateAuthUI(true, d.username);
      showToast(`Welcome back, ${d.username}! 👋`, 'success');
    } catch (_) { $('login-error').style.display=''; $('login-error').textContent = 'Connection error.'; }
  };

  window.handleRegister = async (e) => {
    e.preventDefault();
    $('reg-error').style.display = 'none';
    const u = $('reg-username').value.trim();
    const em = $('reg-email').value.trim();
    const p  = $('reg-password').value;
    try {
      const d = await fetch('/register', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ username: u, email: em, password: p })
      }).then(r => r.json());
      if (d.error) { $('reg-error').style.display=''; $('reg-error').textContent = d.error; return; }
      window.closeAuthModal();
      updateAuthUI(true, d.username);
      showToast(`Account created! Welcome, ${d.username} 🎉`, 'success');
    } catch (_) { $('reg-error').style.display=''; $('reg-error').textContent = 'Connection error.'; }
  };
}


/* ═══════════════════════════════════════════════════════════════════════════
   RESULTS PAGE  (class="page-results" on <body>)
   ═══════════════════════════════════════════════════════════════════════════*/
if (document.body.classList.contains('page-results')) {

  document.addEventListener('DOMContentLoaded', () => {
    initTrendChart();
    animateBars();
  });

  /* ── Trend chart ───────────────────────────────────────────────────────── */
  function initTrendChart() {
    const canvas = $('trend-chart');
    if (!canvas || !RESULT.trend || RESULT.trend.length === 0) return;
    const td = RESULT.trend;

    const labels = td.map(item => item.month);
    const data = td.map(item => item.reviews);

    const datasets = [{
      label: 'Total Reviews',
      data:  data,
      borderColor:     'rgba(124,58,237,1)',
      backgroundColor: 'rgba(124,58,237,0.12)',
      fill: true, tension: 0.4,
      pointBackgroundColor: 'rgba(124,58,237,1)',
      pointRadius: 5, pointHoverRadius: 8,
    }];

    new Chart(canvas, {
      type: 'line',
      data: { labels: labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            labels: { color: '#a0a0c0', font: { family: 'Inter', size: 12 }, padding: 20 }
          },
          tooltip: {
            backgroundColor: 'rgba(13,13,26,.95)',
            borderColor: 'rgba(255,255,255,.1)', borderWidth: 1,
            titleColor: '#f0f0ff', bodyColor: '#a0a0c0',
            titleFont: { family: 'Inter', weight: '700' },
            bodyFont:  { family: 'Inter', size: 12 },
            padding: 14, cornerRadius: 10,
          }
        },
        scales: {
          x: { ticks: { color:'#6060a0', font:{ family:'Inter', size:11 } }, grid: { color:'rgba(255,255,255,.04)' } },
          y: { beginAtZero: true, ticks: { color:'#6060a0', font:{ family:'Inter', size:11 }, stepSize:1 }, grid: { color:'rgba(255,255,255,.04)' } }
        }
      }
    });
  }

  /* ── Animate confidence bars (already set via CSS width, just re-trigger) ── */
  function animateBars() {
    document.querySelectorAll('.conf-bar-fill').forEach(el => {
      const w = el.style.width;
      el.style.width = '0%';
      requestAnimationFrame(() => { setTimeout(() => { el.style.width = w; }, 100); });
    });
  }

  /* ── Table filter ──────────────────────────────────────────────────────── */
  window.filterTable = function(filter, btn) {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('#review-tbody tr').forEach(row => {
      const pred = row.dataset.prediction;
      row.classList.toggle('row-hidden', filter !== 'all' && pred !== filter);
    });
  };

  /* ── Export JSON ───────────────────────────────────────────────────────── */
  window.exportJSON = function() {
    const blob = new Blob([JSON.stringify(RESULT, null, 2)], { type: 'application/json' });
    dl(blob, `reviewguard-${ANALYSIS_ID}.json`);
    showToast('JSON exported ✓', 'success');
  };

  /* ── Export CSV ────────────────────────────────────────────────────────── */
  window.exportCSV = function() {
    const hdr  = ['#','Review Text','Prediction','Confidence %'];
    const rows = RESULT.reviews.map((r, i) =>
      [i+1, `"${r.text.replace(/"/g,'""')}"`, r.prediction, r.confidence]);
    const csv  = [hdr, ...rows].map(r => r.join(',')).join('\n');
    dl(new Blob([csv], { type:'text/csv' }), `reviewguard-reviews-${ANALYSIS_ID}.csv`);
    showToast('CSV exported ✓', 'success');
  };

  function dl(blob, name) {
    const a  = document.createElement('a');
    a.href   = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(a.href);
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   SHARED UTILITIES
   ═══════════════════════════════════════════════════════════════════════════*/
function showToast(msg, type = 'success') {
  const t = $('toast');
  if (!t) return;
  t.textContent   = msg;
  t.className     = `toast ${type}`;
  t.style.display = '';
  clearTimeout(t._to);
  t._to = setTimeout(() => { t.style.display = 'none'; }, 3500);
}

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}
