"""
Build viewer_repose_3d.html with embedded simulation data.

Built by VeloxSim Tech Pty Ltd and Sam Wong.
"""
import json

with open('repose_data.json') as f:
    data = json.load(f)

sim_json = json.dumps(data, separators=(',', ':'))

# Support both old (no_cohesion/cohesion) and new (25deg/40deg) case labels
cases = data['cases']
case_keys = list(cases.keys())
case_a_key = case_keys[0]
case_b_key = case_keys[1]
ang_nc = cases[case_a_key]['repose_angle']
ang_coh = cases[case_b_key]['repose_angle']
gamma = cases[case_b_key]['cohesion_energy']
n = data['n']
d_mm = data['radius'] * 2 * 1000
label_a = case_a_key.replace('_', ' ').title()
label_b = case_b_key.replace('_', ' ').title()

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VeloxSim-DEM - Angle of Repose 3D Viewer</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%;
               background: #0d1117; color: #e6edf3;
               font-family: 'Courier New', monospace; overflow: hidden; }}
  #canvas-wrap {{ position: fixed; inset: 0; }}
  #title-card {{
    position: fixed; top: 14px; left: 14px; z-index: 20;
    background: rgba(13,17,23,0.88); border: 1px solid #30363d;
    border-radius: 8px; padding: 9px 14px; pointer-events: none;
  }}
  #title-card .main {{ font-size: 14px; font-weight: bold; color: #58a6ff; }}
  #title-card .sub  {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
  #title-card .hint {{ font-size: 10px; color: #484f58; margin-top: 5px; line-height: 1.6; }}
  #angle-hud {{
    position: fixed; top: 14px; right: 14px; z-index: 20;
    background: rgba(13,17,23,0.88); border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 18px; min-width: 200px;
  }}
  #angle-hud h3 {{ margin: 0 0 10px; font-size: 11px; color: #8b949e;
                  text-transform: uppercase; letter-spacing: .06em; }}
  .case-row {{ margin: 6px 0; font-size: 13px; }}
  .case-row .dot {{ display: inline-block; width: 10px; height: 10px;
                   border-radius: 50%; margin-right: 7px; vertical-align: middle; }}
  .case-row .lbl {{ color: #8b949e; font-size: 11px; }}
  .case-row .ang {{ font-weight: bold; font-size: 15px; }}
  .delta-row {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid #21262d;
               font-size: 12px; color: #8b949e; }}
  .delta-row span {{ font-size: 15px; font-weight: bold; color: #f0a500; }}
  #legend {{
    position: fixed; bottom: 16px; right: 14px; z-index: 20;
    background: rgba(13,17,23,0.88); border: 1px solid #30363d;
    border-radius: 8px; padding: 9px 14px; font-size: 11px; color: #8b949e;
  }}
  #legend b {{ color: #e6edf3; }}
  #colorbar {{
    position: fixed; bottom: 16px; left: 14px; z-index: 20;
    background: rgba(13,17,23,0.88); border: 1px solid #30363d;
    border-radius: 8px; padding: 9px 14px;
  }}
  #colorbar h3 {{ margin: 0 0 6px; font-size: 11px; color: #8b949e;
                 text-transform: uppercase; letter-spacing: .06em; }}
  #cb-canvas {{ display: block; border-radius: 3px; }}
  .cb-labels {{ display: flex; justify-content: space-between;
               font-size: 10px; color: #6e7681; margin-top: 3px; width: 120px; }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div id="title-card">
  <div class="main">VeloxSim-DEM</div>
  <div class="sub">Angle of Repose &mdash; JKR Cohesion ({n} particles, d={d_mm:.0f}mm)</div>
  <div class="hint">
    Left-drag &nbsp;&#183;&nbsp; orbit<br>
    Right-drag &nbsp;&#183;&nbsp; pan<br>
    Scroll &nbsp;&#183;&nbsp; zoom
  </div>
</div>
<div id="angle-hud">
  <h3>Measured Repose Angle</h3>
  <div class="case-row">
    <span class="dot" id="dot-nc"></span>
    <span class="lbl">No cohesion &nbsp;</span>
    <span class="ang" id="ang-nc" style="color:#4e79a7">--</span>
  </div>
  <div class="case-row">
    <span class="dot" id="dot-coh"></span>
    <span class="lbl">JKR cohesion &nbsp;</span>
    <span class="ang" id="ang-coh" style="color:#e15759">--</span>
  </div>
  <div class="delta-row">Cohesion effect: <span id="ang-delta">--</span></div>
</div>
<div id="legend">
  <b>Left pile</b>: {label_a}<br>
  <b>Right pile</b>: {label_b} (&#947; = {gamma:.0f} J/m&#178;)<br>
  <span style="color:#484f58">{n} spheres (d={d_mm:.0f}mm) coloured by height</span>
</div>
<div id="colorbar">
  <h3>Height (mm)</h3>
  <canvas id="cb-canvas" width="120" height="12"></canvas>
  <div class="cb-labels"><span id="cb-lo">0</span><span id="cb-hi">--</span></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.137.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.137.0/examples/js/controls/OrbitControls.js"></script>

<script>
var SIM_DATA = {sim_json};

var RADIUS = SIM_DATA.radius;
var caseKeys = Object.keys(SIM_DATA.cases);
var cNC  = SIM_DATA.cases[caseKeys[0]];
var cCOH = SIM_DATA.cases[caseKeys[1]];

// Sim Z-up -> Three.js Y-up
var POS_NC  = cNC.positions.map(function(p) {{ return new THREE.Vector3(p[0], p[2], p[1]); }});
var POS_COH = cCOH.positions.map(function(p) {{ return new THREE.Vector3(p[0], p[2], p[1]); }});

var ANG_NC  = cNC.repose_angle.toFixed(1);
var ANG_COH = cCOH.repose_angle.toFixed(1);
var DELTA   = (cCOH.repose_angle - cNC.repose_angle).toFixed(1);

document.getElementById('dot-nc').style.background  = '#4e79a7';
document.getElementById('dot-coh').style.background = '#e15759';
document.getElementById('ang-nc').textContent  = ANG_NC  + '\\u00b0';
document.getElementById('ang-coh').textContent = ANG_COH + '\\u00b0';
var dv = parseFloat(DELTA);
document.getElementById('ang-delta').textContent = (dv >= 0 ? '+' : '') + DELTA + '\\u00b0';

// Viridis-like colormap
function heightColor(t) {{
  var cp = [[0.267,0.004,0.329],[0.128,0.566,0.551],[0.993,0.906,0.144]];
  t = Math.max(0, Math.min(1, t));
  var r, g, b;
  if (t < 0.5) {{
    var s = t * 2;
    r = cp[0][0]+s*(cp[1][0]-cp[0][0]); g = cp[0][1]+s*(cp[1][1]-cp[0][1]); b = cp[0][2]+s*(cp[1][2]-cp[0][2]);
  }} else {{
    var s2 = (t-0.5)*2;
    r = cp[1][0]+s2*(cp[2][0]-cp[1][0]); g = cp[1][1]+s2*(cp[2][1]-cp[1][1]); b = cp[1][2]+s2*(cp[2][2]-cp[1][2]);
  }}
  return new THREE.Color(r, g, b);
}}

var allY = POS_NC.concat(POS_COH).map(function(p) {{ return p.y; }});
var Z_MIN = Math.min.apply(null, allY);
var Z_MAX = Math.max.apply(null, allY);

// Colorbar
var cb = document.getElementById('cb-canvas');
var cbCtx = cb.getContext('2d');
for (var i = 0; i < 120; i++) {{
  var c = heightColor(i/119);
  cbCtx.fillStyle = 'rgb('+((c.r*255)|0)+','+((c.g*255)|0)+','+((c.b*255)|0)+')';
  cbCtx.fillRect(i, 0, 1, 12);
}}
document.getElementById('cb-lo').textContent = '0';
document.getElementById('cb-hi').textContent = ((Z_MAX - RADIUS) * 1000).toFixed(0) + ' mm';

// Renderer
var renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('canvas-wrap').appendChild(renderer.domElement);

var scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

// Camera
var camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 1.2, 3.5);
var TARGET = new THREE.Vector3(0, 0.15, 0);
camera.lookAt(TARGET);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(TARGET);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.1;
controls.maxDistance = 20.0;
controls.update();

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
var sun = new THREE.DirectionalLight(0xfff5e0, 1.8);
sun.position.set(2, 4, 3);
sun.castShadow = true;
sun.shadow.mapSize.width = 2048; sun.shadow.mapSize.height = 2048;
sun.shadow.camera.left = -3; sun.shadow.camera.right = 3;
sun.shadow.camera.top = 3; sun.shadow.camera.bottom = -3;
sun.shadow.camera.near = 0.1; sun.shadow.camera.far = 20;
scene.add(sun);
var fill = new THREE.DirectionalLight(0x4466cc, 0.4);
fill.position.set(-2, 1.5, -2);
scene.add(fill);

// Pile offsets
var OFFSET_NC  = new THREE.Vector3(-1.2, 0, 0);
var OFFSET_COH = new THREE.Vector3( 1.2, 0, 0);

// Ground plates
var plateMat = new THREE.MeshStandardMaterial({{ color: 0x21262d, roughness: 0.85, metalness: 0.1 }});
function makePlate(offsetX) {{
  var m = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.01, 2.0), plateMat.clone());
  m.position.set(offsetX, -0.005, 0);
  m.receiveShadow = true;
  scene.add(m);
  var g = new THREE.GridHelper(1.8, 12, 0x30363d, 0x30363d);
  g.position.set(offsetX, 0.003, 0);
  scene.add(g);
}}
makePlate(OFFSET_NC.x);
makePlate(OFFSET_COH.x);

// Divider
var divMesh = new THREE.Mesh(
  new THREE.BoxGeometry(0.01, 1.0, 0.01),
  new THREE.MeshStandardMaterial({{ color: 0x30363d, metalness: 0.5 }})
);
divMesh.position.set(0, 0.5, 0);
scene.add(divMesh);

// Particle spheres
var sphGeo = new THREE.SphereGeometry(RADIUS * 0.92, 16, 12);

function buildPile(positions, offset) {{
  var group = new THREE.Group();
  for (var i = 0; i < positions.length; i++) {{
    var p = positions[i];
    var t = (Z_MAX > Z_MIN) ? Math.max(0, (p.y - Z_MIN) / (Z_MAX - Z_MIN)) : 0;
    var col = heightColor(t);
    var mat = new THREE.MeshStandardMaterial({{
      color: col, roughness: 0.30, metalness: 0.08,
      emissive: col.clone().multiplyScalar(0.25)
    }});
    var mesh = new THREE.Mesh(sphGeo, mat);
    mesh.position.set(p.x + offset.x, p.y + offset.y, p.z + offset.z);
    mesh.castShadow = true; mesh.receiveShadow = true;
    group.add(mesh);
  }}
  scene.add(group);
}}
buildPile(POS_NC, OFFSET_NC);
buildPile(POS_COH, OFFSET_COH);

// Angle measurement lines — rendered on top using tube geometry for visibility
function drawAngleMeasurement(positions, offset, angle, color) {{
  var maxY = -Infinity;
  for (var i = 0; i < positions.length; i++) {{
    if (positions[i].y > maxY) maxY = positions[i].y;
  }}
  var apex = maxY - RADIUS;

  var radii = [];
  for (var i = 0; i < positions.length; i++) {{
    var dx = positions[i].x;
    var dz = positions[i].z;
    radii.push(Math.sqrt(dx*dx + dz*dz));
  }}
  radii.sort(function(a,b) {{ return a-b; }});
  var baseR = radii[Math.floor(radii.length * 0.9)] + RADIUS;

  var ox = offset.x;
  var baseY = RADIUS;
  var fz = baseR + RADIUS * 3;  // push lines in front of pile

  // Tube material — always on top, bright, thick
  var tubeMat = new THREE.MeshBasicMaterial({{ color: color, depthTest: false, transparent: true, opacity: 0.95 }});
  var whiteMat = new THREE.MeshBasicMaterial({{ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.7 }});
  var dashMat = new THREE.MeshBasicMaterial({{ color: 0x8b949e, depthTest: false, transparent: true, opacity: 0.5 }});
  var tubeR = RADIUS * 0.12;  // tube thickness

  function makeTube(pts, mat) {{
    var path = new THREE.CatmullRomCurve3(pts);
    var geo = new THREE.TubeGeometry(path, 1, tubeR, 6, false);
    var m = new THREE.Mesh(geo, mat);
    m.renderOrder = 999;
    scene.add(m);
  }}

  // Horizontal base line
  makeTube([
    new THREE.Vector3(ox - baseR * 1.15, baseY, fz),
    new THREE.Vector3(ox + baseR * 1.15, baseY, fz)
  ], whiteMat);

  // Slope line from base edge to apex
  makeTube([
    new THREE.Vector3(ox + baseR, baseY, fz),
    new THREE.Vector3(ox, apex, fz)
  ], tubeMat);

  // Vertical dashed line (use thinner tube)
  var dashR = tubeR * 0.6;
  var nDashes = 12;
  var dashH = (apex - baseY) / (nDashes * 2);
  for (var i = 0; i < nDashes; i++) {{
    var y0 = baseY + i * dashH * 2;
    var y1 = y0 + dashH;
    var dpath = new THREE.CatmullRomCurve3([
      new THREE.Vector3(ox, y0, fz),
      new THREE.Vector3(ox, y1, fz)
    ]);
    var dgeo = new THREE.TubeGeometry(dpath, 1, dashR, 4, false);
    var dm = new THREE.Mesh(dgeo, dashMat);
    dm.renderOrder = 999;
    scene.add(dm);
  }}

  // Angle arc
  var arcRadius = baseR * 0.4;
  var angleRad = angle * Math.PI / 180;
  var arcPts = [];
  for (var i = 0; i <= 32; i++) {{
    var a = i / 32 * angleRad;
    arcPts.push(new THREE.Vector3(
      ox + baseR - arcRadius * Math.cos(a),
      baseY + arcRadius * Math.sin(a),
      fz
    ));
  }}
  var arcPath = new THREE.CatmullRomCurve3(arcPts);
  var arcGeo = new THREE.TubeGeometry(arcPath, 32, tubeR * 0.8, 6, false);
  var arcMesh = new THREE.Mesh(arcGeo, tubeMat);
  arcMesh.renderOrder = 999;
  scene.add(arcMesh);

  // Angle text label — small, tucked next to the arc end
  var lblW = 160, lblH = 48;
  var cnv = document.createElement('canvas');
  cnv.width = lblW; cnv.height = lblH;
  var ctx = cnv.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 24px Courier New';
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.fillText(angle.toFixed(1) + '\\u00b0', 4, lblH/2);
  var tex = new THREE.CanvasTexture(cnv);
  var spMat = new THREE.SpriteMaterial({{ map: tex, transparent: true, depthTest: false, depthWrite: false }});
  var spr = new THREE.Sprite(spMat);
  spr.scale.set(0.28, 0.09, 1);
  spr.renderOrder = 1000;
  // Position just outside the arc tip
  var labelAngle = angleRad * 0.5;
  spr.position.set(
    ox + baseR - arcRadius * 1.35 * Math.cos(labelAngle),
    baseY + arcRadius * 1.35 * Math.sin(labelAngle),
    fz
  );
  scene.add(spr);
}}

drawAngleMeasurement(POS_NC, OFFSET_NC, parseFloat(ANG_NC), 0x4e79a7);
drawAngleMeasurement(POS_COH, OFFSET_COH, parseFloat(ANG_COH), 0xe15759);

// Labels
function makeLabel(line1, line2, borderColor) {{
  var w = 320, h = 76;
  var cnv = document.createElement('canvas');
  cnv.width = w; cnv.height = h;
  var ctx = cnv.getContext('2d');
  ctx.fillStyle = 'rgba(13,17,23,0.85)';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = borderColor; ctx.lineWidth = 3;
  ctx.strokeRect(1.5, 1.5, w-3, h-3);
  ctx.fillStyle = borderColor;
  ctx.font = 'bold 24px Courier New'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(line1, w/2, 26);
  ctx.fillStyle = '#8b949e'; ctx.font = '18px Courier New';
  ctx.fillText(line2, w/2, 54);
  var tex = new THREE.CanvasTexture(cnv);
  var spriteMat = new THREE.SpriteMaterial({{ map: tex, transparent: true, depthWrite: false }});
  var spr = new THREE.Sprite(spriteMat);
  spr.scale.set(0.9, 0.22, 1);
  return spr;
}}

var lblNC = makeLabel('{label_a}', '\\u03b8 = ' + ANG_NC + '\\u00b0', '#4e79a7');
lblNC.position.set(OFFSET_NC.x, 0.85, 0);
scene.add(lblNC);

var lblCOH = makeLabel('{label_b}', '\\u03b8 = ' + ANG_COH + '\\u00b0', '#e15759');
lblCOH.position.set(OFFSET_COH.x, 0.85, 0);
scene.add(lblCOH);

// Render loop
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', function() {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>'''

with open('viewer_repose_3d.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Written viewer_repose_3d.html ({len(html)} bytes)')
