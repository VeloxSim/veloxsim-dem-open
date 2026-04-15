"""
Generate an interactive HTML viewer for hopper discharge simulations.

Features:
- Three.js 3D particle animation with hopper geometry
- Three colour modes: Solid, Velocity, Layers (coloured-sand technique)
- Clipping plane through the hopper centre (Y=0) to see flow patterns
- Cinema mode for full-screen video recording
- Playback speed control, particle count HUD, colour bar

The Layers mode uses nearest-neighbour tracking between consecutive frames
to propagate layer IDs from the initial z-position of each particle — the
classic coloured-sand technique used in bulk materials handling research.

Usage:
    python hopper_viewer.py --results hopper_results.json \\
                            --output hopper_animation.html

Built by VeloxSim Tech Pty Ltd and Sam Wong.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Optional

import numpy as np


def generate_hopper_html(
    results_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    title: str = "VeloxSim-DEM Hopper Discharge",
    max_anim_frames: int = 200,
    max_particles_per_frame: int = 30_000,
) -> pathlib.Path:
    """Generate a self-contained HTML file with a 3D hopper animation.

    Parameters
    ----------
    results_path
        JSON file produced by ``demo_hopper.py`` with ``config``, ``frames``
        and optional ``stl``.
    output_path
        Where to write the HTML file.
    title
        Page title.
    max_anim_frames
        Max number of animation frames to embed.
    max_particles_per_frame
        Max particles per frame (subsampled if exceeded).

    Returns
    -------
    pathlib.Path
        The output HTML path.
    """
    results_path = pathlib.Path(results_path)
    output_path = pathlib.Path(output_path)

    # ------------------------------------------------------------------
    # Load simulation results
    # ------------------------------------------------------------------
    print(f"Loading {results_path.name}...", flush=True)
    with open(results_path) as f:
        data = json.load(f)

    config = data["config"]
    frames = data["frames"]
    stl_data = data.get("stl", {})
    MAX_PARTICLES = config.get("n_particles", 5000)

    # ------------------------------------------------------------------
    # Subsample frames + propagate layer IDs via nearest-neighbour
    # tracking.  Each particle keeps the layer colour it had in the
    # first frame, so the layer deformation reveals the flow pattern
    # (the classic coloured-sand technique).
    # ------------------------------------------------------------------
    PARTICLE_SUBSAMPLE = max(1, MAX_PARTICLES // max_particles_per_frame)
    step = max(1, len(frames) // max_anim_frames)
    anim_frames = []
    max_n = 0

    try:
        from scipy.spatial import cKDTree
        _has_scipy = True
    except ImportError:
        _has_scipy = False
        print("  WARNING: scipy not available — layer colors will be slower")

    N_LAYERS = 8
    first_fr = frames[0]
    first_pos_full = np.array(
        first_fr.get("pos", first_fr.get("p", [])), dtype=np.float32
    )
    if len(first_pos_full) > 0:
        live_mask_0 = first_pos_full[:, 2] < 500.0
        first_live = first_pos_full[live_mask_0]
        if len(first_live) > 0:
            z_min_init = float(first_live[:, 2].min())
            z_max_init = float(first_live[:, 2].max())
        else:
            z_min_init, z_max_init = 0.0, 1.0
    else:
        z_min_init, z_max_init = 0.0, 1.0
    z_range_init = max(z_max_init - z_min_init, 1e-6)

    def _assign_initial_layers(positions):
        layers = np.zeros(len(positions), dtype=np.int8)
        for i in range(len(positions)):
            z = positions[i][2]
            if z >= 500.0:
                layers[i] = 0
            else:
                lid = int((z - z_min_init) / z_range_init * N_LAYERS)
                layers[i] = max(0, min(N_LAYERS - 1, lid))
        return layers

    print(f"  Tracking particle layers through {len(frames)} frames...", flush=True)
    full_layers = [None] * len(frames)
    prev_pos = np.array(
        frames[0].get("pos", frames[0].get("p", [])), dtype=np.float32
    )
    full_layers[0] = _assign_initial_layers(prev_pos)

    for fi in range(1, len(frames)):
        cur_pos = np.array(
            frames[fi].get("pos", frames[fi].get("p", [])), dtype=np.float32
        )
        if len(cur_pos) == 0 or len(prev_pos) == 0:
            full_layers[fi] = np.zeros(len(cur_pos), dtype=np.int8)
            prev_pos = cur_pos
            continue
        if _has_scipy:
            tree = cKDTree(prev_pos)
            _, nn_idx = tree.query(cur_pos, k=1)
            cur_layers = full_layers[fi - 1][nn_idx]
        else:
            cur_layers = np.zeros(len(cur_pos), dtype=np.int8)
            for ci in range(len(cur_pos)):
                d2 = np.sum((prev_pos - cur_pos[ci]) ** 2, axis=1)
                nn = int(d2.argmin())
                cur_layers[ci] = full_layers[fi - 1][nn]
        full_layers[fi] = cur_layers
        prev_pos = cur_pos

    for fi in range(0, len(frames), step):
        fr = frames[fi]
        frame_layers = full_layers[fi]
        n_fr = fr.get("n", len(fr.get("pos", [])))
        indices = list(range(0, n_fr, PARTICLE_SUBSAMPLE))
        n_sub = len(indices)
        max_n = max(max_n, n_sub)

        compact = {"t": round(fr["t"], 3), "n": n_sub, "p": [], "s": [], "l": []}
        pos_data = fr.get("pos", fr.get("p", []))
        vel_data = fr.get("vel", [])
        speed_data = fr.get("s", [])

        for i in indices:
            px, py, pz = pos_data[i]
            compact["p"].append([round(px, 3), round(py, 3), round(pz, 3)])
            if speed_data:
                compact["s"].append(round(speed_data[i], 2))
            elif vel_data:
                vx, vy, vz = vel_data[i]
                compact["s"].append(round(math.sqrt(vx * vx + vy * vy + vz * vz), 2))
            else:
                compact["s"].append(0.0)
            compact["l"].append(int(frame_layers[i]) if i < len(frame_layers) else 0)
        anim_frames.append(compact)

    print(f"  {len(anim_frames)} frames, max {max_n} particles/frame", flush=True)

    # ------------------------------------------------------------------
    # Build payload
    # ------------------------------------------------------------------
    payload = json.dumps(
        {
            "config": {**config, "n_particles": max_n},
            "stl": stl_data,
            "frames": anim_frames,
        },
        separators=(",", ":"),
    )
    print(f"  Payload: {len(payload)/1024/1024:.1f} MB", flush=True)

    # ------------------------------------------------------------------
    # Generate HTML
    # ------------------------------------------------------------------
    html = _HTML_TEMPLATE.replace("__PAYLOAD__", payload).replace("__TITLE__", title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Written: {output_path}", flush=True)
    return output_path


# ======================================================================
# HTML Template
# ======================================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>__TITLE__</title>
<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js" } }
</script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0f172a; overflow:hidden; font-family:'Segoe UI','Inter',sans-serif; }

  #viewport { position:absolute; top:0; left:0; right:0; bottom:0; }

  #controls {
    position:absolute; bottom:0; left:0; right:0; z-index:10;
    background:rgba(15,23,42,0.92); border-top:1px solid #334155;
    padding:10px 16px; display:flex; align-items:center; gap:12px;
    font-size:12px; color:#94a3b8;
  }
  #controls .label { color:#64748b; margin-right:4px; }
  #controls .btn-group { display:flex; gap:4px; align-items:center; }
  #controls button {
    background:#334155; border:none; color:#e2e8f0; padding:5px 12px;
    border-radius:4px; cursor:pointer; font-size:11px;
  }
  #controls button:hover { background:#475569; }
  #controls button.active { background:#3b82f6; color:#fff; }
  #time-label, #particle-count { color:#e2e8f0; font-variant-numeric:tabular-nums; }

  #scrubber-container {
    flex:1; height:8px; background:#334155; border-radius:4px;
    position:relative; cursor:pointer;
  }
  #scrubber-fill {
    position:absolute; left:0; top:0; bottom:0;
    background:#3b82f6; border-radius:4px; width:0%;
  }

  #info {
    position:absolute; top:16px; left:16px; color:#e2e8f0;
    background:rgba(15,23,42,0.85); padding:12px 16px; border-radius:8px;
    font:13px/1.6 'Segoe UI',sans-serif; border:1px solid #334155; z-index:10;
    max-width:320px;
  }
  #info h1 { font-size:15px; margin-bottom:6px; color:#f1f5f9; }
  #info .meta { color:#94a3b8; font-size:11px; }

  #colorbar {
    position:absolute; top:16px; right:16px; z-index:10; display:none;
    background:rgba(15,23,42,0.9); border:1px solid #334155;
    border-radius:8px; padding:8px 12px; color:#e2e8f0; font-size:11px;
  }
  #colorbar canvas { display:block; margin-top:4px; border-radius:2px; }
  #cb-labels { display:flex; justify-content:space-between; margin-top:4px; color:#94a3b8; }
  #cb-controls { display:flex; gap:6px; margin-top:6px; align-items:center; }
  #cb-controls input[type=number] {
    width:50px; background:#1e293b; border:1px solid #475569; color:#e2e8f0;
    border-radius:3px; padding:2px 4px; font-size:11px;
  }
  #cb-controls button {
    background:#334155; border:none; color:#e2e8f0; padding:3px 8px;
    border-radius:3px; cursor:pointer; font-size:11px;
  }

  /* Cinema mode */
  body.cinema #controls { background:rgba(0,0,0,0.6); border-top:none; }
  body.cinema #info { display:none; }
  #cinema-hud {
    display:none; position:absolute; top:16px; left:50%;
    transform:translateX(-50%); z-index:15; pointer-events:none;
    background:rgba(0,0,0,0.7); padding:10px 24px; border-radius:8px;
    text-align:center;
  }
  #cinema-hud .ch-time {
    font-size:28px; color:#e2e8f0; font-weight:600;
    font-variant-numeric:tabular-nums;
  }
  #cinema-hud .ch-count { font-size:13px; color:#94a3b8; margin-top:2px; }
  body.cinema #cinema-hud { display:block; }
</style>
</head>
<body>

<div id="viewport"></div>

<div id="info">
  <h1 id="sim-title">__TITLE__</h1>
  <div class="meta" id="sim-meta"></div>
</div>

<div id="cinema-hud">
  <div class="ch-time">t = 0.000 s</div>
  <div class="ch-count"></div>
</div>

<div id="colorbar">
  <div>Speed (m/s)</div>
  <canvas id="cb-canvas" width="140" height="12"></canvas>
  <div id="cb-labels">
    <span id="cb-min-label">0.0</span>
    <span id="cb-mid">0.5</span>
    <span id="cb-max">1.0</span>
  </div>
  <div id="cb-controls">
    <span>min</span>
    <input type="number" id="vel-min" step="0.1" value="0.0"/>
    <span>max</span>
    <input type="number" id="vel-max" step="0.1" value="1.0"/>
    <button id="btn-vel-reset">Auto</button>
  </div>
</div>

<div id="controls">
  <button id="btn-play">▶</button>
  <span id="time-label">t = 0.000 s</span>
  <div id="scrubber-container"><div id="scrubber-fill"></div></div>
  <span id="particle-count"></span>
  <div class="btn-group">
    <span class="label">Color:</span>
    <button id="btn-solid" class="active">Solid</button>
    <button id="btn-vel">Velocity</button>
    <button id="btn-layer">Layers</button>
  </div>
  <span class="label">Speed:</span>
  <input type="range" id="pb-speed" min="1" max="50" value="10" style="max-width:80px"/>
  <span class="label">&times;<span id="pb-val">1.0</span></span>
  <button id="btn-clip" title="Toggle clipping plane — cross-section through hopper centre">Clip</button>
  <button id="btn-cinema" title="Cinema mode — full-screen for video recording">Cinema</button>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js';

const SIM = __PAYLOAD__;
const FRAMES = SIM.frames;
const STL_DATA = SIM.stl || {};
const CONFIG = SIM.config;
const R = CONFIG.radius;
const N_MAX = CONFIG.n_particles;

// Metadata display
document.getElementById("sim-meta").textContent =
  `${FRAMES.length} frames  \u00b7  ${N_MAX.toLocaleString()} particles  \u00b7  ` +
  `R = ${(R*1000).toFixed(1)} mm`;

// ==================================================================
// Three.js scene
// ==================================================================
const viewport = document.getElementById("viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f172a);

const camera = new THREE.PerspectiveCamera(
  50, viewport.clientWidth / viewport.clientHeight, 0.01, 100
);
// Z-up convention (DEM simulation uses Z as vertical axis)
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.localClippingEnabled = true;
renderer.autoClear = false;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0x94a3b8, 2.0));
const dl1 = new THREE.DirectionalLight(0xffffff, 2.5);
dl1.position.set(5, -10, 8); dl1.castShadow = true; scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0xffffff, 1.2);
dl2.position.set(-5, 10, 3); scene.add(dl2);

// ==================================================================
// Axis gizmo (small indicator in bottom-left showing X/Y/Z orientation)
// ==================================================================
const gizmoScene = new THREE.Scene();
const gizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
gizmoCamera.up.set(0, 0, 1);

function makeArrow(dir, color) {
  const arrow = new THREE.ArrowHelper(
    dir, new THREE.Vector3(0, 0, 0), 1.0, color, 0.3, 0.2
  );
  return arrow;
}
gizmoScene.add(makeArrow(new THREE.Vector3(1, 0, 0), 0xff4444));  // X red
gizmoScene.add(makeArrow(new THREE.Vector3(0, 1, 0), 0x44ff44));  // Y green
gizmoScene.add(makeArrow(new THREE.Vector3(0, 0, 1), 0x4488ff));  // Z blue

// Axis labels using CanvasTexture sprites
function makeLabel(text, color) {
  const cvs = document.createElement('canvas');
  cvs.width = 64; cvs.height = 64;
  const ctx = cvs.getContext('2d');
  ctx.font = 'bold 48px sans-serif';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 32);
  const tex = new THREE.CanvasTexture(cvs);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.5, 0.5, 0.5);
  return spr;
}
const labelX = makeLabel('X', '#ff4444'); labelX.position.set(1.35, 0, 0); gizmoScene.add(labelX);
const labelY = makeLabel('Y', '#44ff44'); labelY.position.set(0, 1.35, 0); gizmoScene.add(labelY);
const labelZ = makeLabel('Z', '#4488ff'); labelZ.position.set(0, 0, 1.35); gizmoScene.add(labelZ);

// Clipping plane through Y=0 (centre cross-section)
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let clipEnabled = false;

// STL meshes
for (const [name, mesh] of Object.entries(STL_DATA)) {
  const geo = new THREE.BufferGeometry();
  const verts = new Float32Array(mesh.v.flat());
  const faces = new Uint32Array(mesh.f);
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6, metalness: 0.2, roughness: 0.5,
    side: THREE.DoubleSide, transparent: true, opacity: 0.5,
    clippingPlanes: [],
  });
  const m = new THREE.Mesh(geo, mat);
  scene.add(m);
}

// Particles
const pGeo = new THREE.SphereGeometry(R, 12, 8);
const pMat = new THREE.MeshStandardMaterial({
  roughness: 0.35, metalness: 0.15,
  emissive: 0xdc2626, emissiveIntensity: 0.25,
});
pMat.clippingPlanes = [];
const inst = new THREE.InstancedMesh(pGeo, pMat, N_MAX);
inst.frustumCulled = false;
inst.count = 0;
scene.add(inst);

let colorMode = 'solid';
const solidColor = new THREE.Color(0xf87171);

// Alternating earth-tone layers — classic coloured-sand experiment
const LAYER_COLORS = [
  new THREE.Color(0x8B4513),  // saddlebrown
  new THREE.Color(0xDAA520),  // goldenrod
  new THREE.Color(0xA0522D),  // sienna
  new THREE.Color(0xF4A460),  // sandybrown
  new THREE.Color(0x654321),  // dark brown
  new THREE.Color(0xFFD700),  // gold
  new THREE.Color(0x5C4033),  // very dark brown
  new THREE.Color(0xDEB887),  // burlywood
];

// Velocity colormap: deep blue -> cyan -> green -> yellow -> red
const CMAP_STOPS = [
  [0.00, 0.00, 0.00, 0.60],
  [0.25, 0.00, 0.70, 1.00],
  [0.50, 0.00, 1.00, 0.00],
  [0.75, 1.00, 1.00, 0.00],
  [1.00, 1.00, 0.00, 0.00],
];
function turboColor(t) {
  t = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < CMAP_STOPS.length - 2 && CMAP_STOPS[i + 1][0] < t) i++;
  const a = CMAP_STOPS[i], b = CMAP_STOPS[i + 1];
  const f = (t - a[0]) / (b[0] - a[0]);
  return new THREE.Color(
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
    a[3] + (b[3] - a[3]) * f
  );
}

// Velocity range (auto-computed from first non-empty frame)
let autoMaxSpeed = 0;
for (const fr of FRAMES) {
  for (const s of fr.s) { if (s > autoMaxSpeed) autoMaxSpeed = s; }
}
let velMin = 0, velMax = Math.max(autoMaxSpeed, 1.0);

function updateColorBar() {
  const cb = document.getElementById('cb-canvas'), ctx = cb.getContext('2d');
  for (let i = 0; i < 140; i++) {
    const c = turboColor(i / 139);
    ctx.fillStyle = `rgb(${(c.r*255)|0},${(c.g*255)|0},${(c.b*255)|0})`;
    ctx.fillRect(i, 0, 1, 12);
  }
  document.getElementById('cb-min-label').textContent = velMin.toFixed(1);
  document.getElementById('cb-mid').textContent = ((velMin + velMax) / 2).toFixed(1);
  document.getElementById('cb-max').textContent = velMax.toFixed(1);
  document.getElementById('vel-min').value = velMin.toFixed(1);
  document.getElementById('vel-max').value = velMax.toFixed(1);
}
updateColorBar();

// ==================================================================
// Frame rendering
// ==================================================================
const tc = new THREE.Color();
const d = new THREE.Object3D();

function setFrame(idx) {
  const fr = FRAMES[Math.min(idx, FRAMES.length - 1)];
  const activeN = fr.n || fr.p.length;
  inst.count = activeN;
  for (let p = 0; p < activeN; p++) {
    const pos = fr.p[p];
    if (colorMode === 'velocity') {
      const range = velMax - velMin || 1;
      tc.copy(turboColor(Math.max(0, Math.min(1, (fr.s[p] - velMin) / range))));
    } else if (colorMode === 'layer') {
      const lid = (fr.l && p < fr.l.length) ? fr.l[p] : 0;
      tc.copy(LAYER_COLORS[lid % LAYER_COLORS.length]);
    } else {
      tc.copy(solidColor);
    }
    inst.setColorAt(p, tc);
    d.position.set(pos[0], pos[1], pos[2]);
    d.updateMatrix();
    inst.setMatrixAt(p, d.matrix);
  }
  inst.instanceMatrix.needsUpdate = true;
  if (inst.instanceColor) inst.instanceColor.needsUpdate = true;

  // Update HUD
  document.getElementById("time-label").textContent = `t = ${fr.t.toFixed(3)} s`;
  document.getElementById("particle-count").textContent =
    `${activeN.toLocaleString()} particles`;
  document.getElementById("scrubber-fill").style.width =
    `${(idx / Math.max(1, FRAMES.length - 1)) * 100}%`;
  document.querySelector("#cinema-hud .ch-time").textContent = `t = ${fr.t.toFixed(3)} s`;
  document.querySelector("#cinema-hud .ch-count").textContent =
    `${activeN.toLocaleString()} particles`;
}

// ==================================================================
// Camera auto-framing (first non-empty frame)
// ==================================================================
let targetFrame = FRAMES[0];
for (const fr of FRAMES) { if (fr.p.length > 0) { targetFrame = fr; break; } }
const pts = targetFrame.p;
let cx = 0, cy = 0, cz = 0;
let xmin = Infinity, xmax = -Infinity;
let ymin = Infinity, ymax = -Infinity;
let zmin = Infinity, zmax = -Infinity;
for (const p of pts) {
  cx += p[0]; cy += p[1]; cz += p[2];
  if (p[0] < xmin) xmin = p[0]; if (p[0] > xmax) xmax = p[0];
  if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1];
  if (p[2] < zmin) zmin = p[2]; if (p[2] > zmax) zmax = p[2];
}
const n = pts.length || 1;
cx /= n; cy /= n; cz /= n;
const bedSize = Math.max(xmax - xmin, ymax - ymin, zmax - zmin, 1);
const bedCentre = new THREE.Vector3(cx, cy, cz);
camera.position.set(cx + bedSize * 1.5, cy + bedSize * 1.5, cz + bedSize * 0.8);
controls.target.copy(bedCentre);
controls.update();

// ==================================================================
// Playback
// ==================================================================
let frameIdx = 0, playing = false, lastT = 0;
let playbackSpeed = 1.0;
const tMin = FRAMES[0].t, tMax = FRAMES[FRAMES.length - 1].t;
const totalDur = Math.max(tMax - tMin, 0.001);

function sliderToSpeed(val) {
  if (val <= 10) return 0.2 + (val - 1) * (0.8 / 9);  // 0.2 .. 1.0
  return 1.0 + (val - 10) * (4.0 / 40);                // 1.0 .. 5.0
}

function animate(now) {
  requestAnimationFrame(animate);
  controls.update();
  if (playing) {
    const dt = (now - lastT) / 1000;
    lastT = now;
    const simAdvance = dt * playbackSpeed;
    const frameStep = simAdvance / totalDur * FRAMES.length;
    frameIdx = (frameIdx + frameStep) % FRAMES.length;
    setFrame(Math.floor(frameIdx));
  }
  // Main scene
  renderer.setViewport(0, 0, viewport.clientWidth, viewport.clientHeight);
  renderer.setScissor(0, 0, viewport.clientWidth, viewport.clientHeight);
  renderer.setScissorTest(true);
  renderer.clear();
  renderer.render(scene, camera);

  // Axis gizmo overlay in bottom-left corner (100x100 px)
  const gs = 110;
  renderer.setViewport(10, 10, gs, gs);
  renderer.setScissor(10, 10, gs, gs);
  renderer.clearDepth();
  // Mirror the main camera's orientation for the gizmo camera
  const offset = new THREE.Vector3();
  offset.subVectors(camera.position, controls.target).normalize().multiplyScalar(4);
  gizmoCamera.position.copy(offset);
  gizmoCamera.up.copy(camera.up);
  gizmoCamera.lookAt(0, 0, 0);
  renderer.render(gizmoScene, gizmoCamera);
  renderer.setScissorTest(false);
}
setFrame(0);
animate(performance.now());

// ==================================================================
// UI handlers
// ==================================================================
document.getElementById("btn-play").addEventListener("click", (e) => {
  playing = !playing;
  e.target.textContent = playing ? "⏸" : "▶";
  lastT = performance.now();
});

document.getElementById("scrubber-container").addEventListener("click", (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  frameIdx = Math.floor(frac * FRAMES.length);
  setFrame(Math.floor(frameIdx));
});

const pbs = document.getElementById("pb-speed");
pbs.addEventListener("input", () => {
  playbackSpeed = sliderToSpeed(parseInt(pbs.value));
  document.getElementById("pb-val").textContent = playbackSpeed.toFixed(1);
});

document.getElementById("btn-solid").addEventListener("click", () => {
  colorMode = 'solid';
  pMat.emissive.setHex(0xdc2626); pMat.emissiveIntensity = 0.25;
  document.getElementById("btn-solid").classList.add("active");
  document.getElementById("btn-vel").classList.remove("active");
  document.getElementById("btn-layer").classList.remove("active");
  document.getElementById("colorbar").style.display = "none";
  setFrame(Math.floor(frameIdx));
});
document.getElementById("btn-vel").addEventListener("click", () => {
  colorMode = 'velocity';
  pMat.emissive.setHex(0x000000); pMat.emissiveIntensity = 0.0;
  document.getElementById("btn-vel").classList.add("active");
  document.getElementById("btn-solid").classList.remove("active");
  document.getElementById("btn-layer").classList.remove("active");
  document.getElementById("colorbar").style.display = "block";
  setFrame(Math.floor(frameIdx));
});
document.getElementById("btn-layer").addEventListener("click", () => {
  colorMode = 'layer';
  pMat.emissive.setHex(0x000000); pMat.emissiveIntensity = 0.0;
  document.getElementById("btn-layer").classList.add("active");
  document.getElementById("btn-solid").classList.remove("active");
  document.getElementById("btn-vel").classList.remove("active");
  document.getElementById("colorbar").style.display = "none";
  setFrame(Math.floor(frameIdx));
});

document.getElementById("vel-min").addEventListener("change", (e) => {
  velMin = parseFloat(e.target.value) || 0;
  updateColorBar(); setFrame(Math.floor(frameIdx));
});
document.getElementById("vel-max").addEventListener("change", (e) => {
  velMax = parseFloat(e.target.value) || 1;
  updateColorBar(); setFrame(Math.floor(frameIdx));
});
document.getElementById("btn-vel-reset").addEventListener("click", () => {
  velMin = 0; velMax = Math.max(autoMaxSpeed, 1.0);
  updateColorBar(); setFrame(Math.floor(frameIdx));
});

document.getElementById("btn-clip").addEventListener("click", (e) => {
  clipEnabled = !clipEnabled;
  e.target.classList.toggle("active", clipEnabled);
  pMat.clippingPlanes = clipEnabled ? [clipPlane] : [];
  // Apply clipping to STL materials too
  scene.traverse((obj) => {
    if (obj.isMesh && obj.material && obj !== inst) {
      obj.material.clippingPlanes = clipEnabled ? [clipPlane] : [];
      obj.material.needsUpdate = true;
    }
  });
  pMat.needsUpdate = true;
  // Snap the camera to look straight at the clipped cross-section.
  // The clip plane normal is (0,1,0), so the +Y half is kept and
  // the -Y half is clipped away.  Position the camera on the -Y
  // side looking toward +Y so it sees into the exposed cavity.
  if (clipEnabled) {
    camera.position.set(
      bedCentre.x,
      bedCentre.y - bedSize * 2.0,
      bedCentre.z,
    );
    controls.target.copy(bedCentre);
    controls.update();
  }
});

document.getElementById("btn-cinema").addEventListener("click", (e) => {
  document.body.classList.toggle("cinema");
  e.target.classList.toggle("active");
  if (document.body.classList.contains("cinema")) {
    playing = true;
    document.getElementById("btn-play").textContent = "⏸";
    lastT = performance.now();
  }
});

// Resize handler
window.addEventListener("resize", () => {
  camera.aspect = viewport.clientWidth / viewport.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
});
</script>
</body></html>
"""


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML viewer for hopper discharge results"
    )
    parser.add_argument("--results", required=True,
                        help="Path to hopper_results.json")
    parser.add_argument("--output", required=True,
                        help="Output HTML file path")
    parser.add_argument("--title", default="VeloxSim-DEM Hopper Discharge",
                        help="Page title")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Max animation frames to embed")
    parser.add_argument("--max-particles", type=int, default=30_000,
                        help="Max particles per frame (subsampled if exceeded)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    generate_hopper_html(
        results_path=args.results,
        output_path=args.output,
        title=args.title,
        max_anim_frames=args.max_frames,
        max_particles_per_frame=args.max_particles,
    )
    print(f"Done in {time.perf_counter() - t0:.1f}s")
