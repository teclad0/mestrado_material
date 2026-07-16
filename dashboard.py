#!/usr/bin/env python3
"""
Live experiment dashboard. Run in a separate terminal while experiments execute.
Usage: python dashboard.py [--port 8050] [--dir experiment_results]
"""

import http.server
import json
import os
import argparse
from pathlib import Path

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { color: #64ffda; margin-bottom: 10px; font-size: 1.4em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
.card { background: #16213e; border-radius: 8px; padding: 16px; border: 1px solid #0f3460; }
.card h2 { font-size: 0.9em; color: #8892b0; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
.big-number { font-size: 2.2em; font-weight: bold; color: #64ffda; }
.sub { font-size: 0.85em; color: #8892b0; margin-top: 4px; }
.progress-bar { width: 100%; height: 24px; background: #0f3460; border-radius: 12px; overflow: hidden; margin: 12px 0; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #64ffda, #00b4d8); transition: width 0.5s; display: flex; align-items: center; justify-content: center; font-size: 0.75em; font-weight: bold; color: #1a1a2e; }
.full-width { grid-column: 1 / -1; }
table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
th { text-align: left; padding: 6px 8px; color: #64ffda; border-bottom: 1px solid #0f3460; }
td { padding: 5px 8px; border-bottom: 1px solid #0f3460; }
tr:hover td { background: #0f3460; }
.status-success { color: #64ffda; }
.status-error { color: #ff6b6b; }
.particles-bar { display: flex; gap: 4px; align-items: flex-end; height: 60px; margin-top: 8px; }
.p-col { flex: 1; background: #0f3460; border-radius: 3px 3px 0 0; position: relative; transition: height 0.5s; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; }
.p-col .label { font-size: 0.65em; color: #8892b0; margin-top: 4px; position: absolute; bottom: -16px; }
.p-col .value { font-size: 0.7em; color: #64ffda; position: absolute; top: -14px; }
.meta { font-size: 0.75em; color: #555; text-align: right; margin-top: 12px; }
.disconnected { color: #ff6b6b; font-weight: bold; }
</style>
</head>
<body>
<h1>PULearningPC Experiment Dashboard</h1>
<div id="status-line" class="sub">Connecting...</div>

<div class="progress-bar"><div class="progress-fill" id="progress" style="width:0%">0%</div></div>

<div class="grid">
  <div class="card">
    <h2>Progress</h2>
    <div class="big-number" id="completed">-</div>
    <div class="sub"><span id="successful">0</span> successful / <span id="failed">0</span> failed</div>
  </div>
  <div class="card">
    <h2>Time</h2>
    <div class="big-number" id="eta">--:--</div>
    <div class="sub">ETA remaining | Elapsed: <span id="elapsed">--:--</span></div>
  </div>
  <div class="card">
    <h2>Task Rate</h2>
    <div class="big-number" id="rate">-</div>
    <div class="sub">avg seconds per task | <span id="in-flight">0</span> in-flight</div>
  </div>
  <div class="card">
    <h2>Dataset</h2>
    <div class="big-number" id="dataset">-</div>
    <div class="sub" id="dataset-sub"></div>
  </div>

  <div class="card full-width">
    <h2>Progress by Particle Count</h2>
    <div class="particles-bar" id="particles-bar"></div>
    <div style="margin-top:20px;">
      <h2 style="margin-top:8px;">Best F1 by Particle Count</h2>
      <div class="particles-bar" id="f1-bar"></div>
    </div>
  </div>

  <div class="card full-width">
    <h2>Recent Tasks</h2>
    <table>
      <thead><tr><th>#</th><th>Parameters</th><th>Time</th><th>Iters</th><th>Coverage</th><th>F1</th><th>Status</th></tr></thead>
      <tbody id="recent-tasks"></tbody>
    </table>
  </div>
</div>

<div class="meta">Auto-refreshes every 2s | <span id="last-update">-</span></div>

<script>
function formatTime(seconds) {
    if (!seconds || seconds < 0) return '--:--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return h + 'h ' + m.toString().padStart(2,'0') + 'm';
    return m + 'm ' + s.toString().padStart(2,'0') + 's';
}

function renderParticlesBar(containerId, data, maxVal, colorFn) {
    const container = document.getElementById(containerId);
    if (!data || Object.keys(data).length === 0) { container.innerHTML = '<div class="sub">No data yet</div>'; return; }
    const keys = Object.keys(data).sort((a,b) => parseInt(a) - parseInt(b));
    const effectiveMax = maxVal || Math.max(...Object.values(data), 1);
    container.innerHTML = keys.map(k => {
        const pct = Math.max(5, (data[k] / effectiveMax) * 100);
        const color = colorFn ? colorFn(data[k]) : '#64ffda';
        return '<div class="p-col" style="height:' + pct + '%;background:' + color + '"><span class="value">' + (typeof data[k] === 'number' && data[k] % 1 !== 0 ? data[k].toFixed(3) : data[k]) + '</span><span class="label">p=' + k + '</span></div>';
    }).join('');
}

async function refresh() {
    try {
        const resp = await fetch('/status.json?' + Date.now());
        if (!resp.ok) { document.getElementById('status-line').innerHTML = '<span class="disconnected">No experiment running (status.json not found)</span>'; return; }
        const d = await resp.json();

        const pct = d.total_tasks ? Math.round(d.completed / d.total_tasks * 100) : 0;
        document.getElementById('progress').style.width = pct + '%';
        document.getElementById('progress').textContent = pct + '%';
        document.getElementById('completed').textContent = d.completed + ' / ' + d.total_tasks;
        document.getElementById('successful').textContent = d.successful;
        document.getElementById('failed').textContent = d.failed;
        document.getElementById('eta').textContent = formatTime(d.eta_seconds);
        document.getElementById('elapsed').textContent = formatTime(d.elapsed_seconds);
        document.getElementById('rate').textContent = d.avg_task_time ? d.avg_task_time.toFixed(1) + 's' : '-';
        document.getElementById('in-flight').textContent = d.in_flight;
        document.getElementById('dataset').textContent = d.dataset || '-';
        document.getElementById('dataset-sub').textContent = d.percent_positive ? 'percent_positive = ' + (d.percent_positive * 100).toFixed(0) + '%' : '';
        document.getElementById('last-update').textContent = d.last_updated || '-';
        const pctPosLabel = d.percent_positive ? ' | pct_pos=' + (d.percent_positive * 100).toFixed(0) + '%' : '';
        document.getElementById('status-line').textContent = 'Live | ' + d.dataset + pctPosLabel + ' | ' + pct + '% complete';

        renderParticlesBar('particles-bar', d.completed_by_particles, d.total_tasks / Object.keys(d.completed_by_particles || {}).length || 100, () => '#00b4d8');
        renderParticlesBar('f1-bar', d.best_f1_by_particles, 1.0, v => v > 0.9 ? '#64ffda' : v > 0.5 ? '#ffd93d' : '#ff6b6b');

        const tbody = document.getElementById('recent-tasks');
        const tasks = (d.recent_tasks || []).slice().reverse();
        tbody.innerHTML = tasks.map(t =>
            '<tr><td>' + t.task_idx + '</td><td>' + t.params + '</td><td>' + t.time + 's</td><td>' + (t.iters||'-') + '</td><td>' + (t.coverage ? (t.coverage*100).toFixed(1)+'%' : '-') + '</td><td>' + (t.f1||0).toFixed(3) + '</td><td class="status-' + (t.status === 'success' ? 'success' : 'error') + '">' + t.status + '</td></tr>'
        ).join('');
    } catch(e) {
        document.getElementById('status-line').innerHTML = '<span class="disconnected">Cannot reach dashboard server</span>';
    }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, results_dir='experiment_results', **kwargs):
        self.results_dir = results_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path.startswith('/status.json'):
            status_path = os.path.join(self.results_dir, 'live_status.json')
            try:
                with open(status_path, 'r') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(data.encode())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description='Live experiment dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port (default: 8050)')
    parser.add_argument('--dir', default='experiment_results', help='Results directory')
    args = parser.parse_args()

    handler = lambda *a, **kw: DashboardHandler(*a, results_dir=args.dir, **kw)
    server = http.server.HTTPServer(('0.0.0.0', args.port), handler)
    print(f"Dashboard running at http://localhost:{args.port}")
    print(f"Watching: {args.dir}/live_status.json")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == '__main__':
    main()
