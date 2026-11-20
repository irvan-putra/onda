"use strict";
const $ = id => document.getElementById(id), S = 640;
const TABS = [["sensors", "Sensor Lab"], ["detect", "Object Detection"], ["pose", "Pose Estimation"], ["classify", "Image Classification"]];
let tab = "sensors", COCO = [], IMN = [];
const SKEL = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const nav = $("nav");
for (const [id, name] of TABS) { const b = document.createElement("button"); b.textContent = name; b.dataset.t = id; b.onclick = () => show(id); nav.appendChild(b); }
function show(id) {
  tab = id;
  for (const [t] of TABS) $(t).classList.toggle("hide", t !== id);
  $("vision").classList.toggle("hide", id === "sensors");
  if (id !== "sensors") $("vision").parentNode.insertBefore($("vision"), $(id).nextSibling);
  nav.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.t === id));
  $("vst").textContent = "Pick a photo or start the camera.";
}
show("sensors");

/* ---------- Sensor Lab (Generic Sensor API) ---------- */
const SENSORS = [["Accelerometer", "Accelerometer", ["x","y","z"], "m/s2"], ["LinearAccelerationSensor", "Linear acceleration", ["x","y","z"], "m/s2"], ["GravitySensor", "Gravity", ["x","y","z"], "m/s2"], ["Gyroscope", "Gyroscope", ["x","y","z"], "rad/s"], ["Magnetometer", "Magnetometer", ["x","y","z"], "uT"], ["AmbientLightSensor", "Ambient light", ["illuminance"], "lx"], ["AbsoluteOrientationSensor", "Absolute orientation", ["quaternion"], ""], ["RelativeOrientationSensor", "Relative orientation", ["quaternion"], ""]];
let live = [];
$("sstart").onclick = async () => {
  if (live.length) { live.forEach(s => { try { s.stop(); } catch (e) {} }); live = []; $("sstart").textContent = "Start sensors"; $("sst").textContent = "Stopped."; return; }
  const rows = [];
  $("stb").replaceChildren();
  for (const [cls, label, props, unit] of SENSORS) {
    const tr = document.createElement("tr"), a = document.createElement("td"), b = document.createElement("td");
    a.textContent = label; tr.append(a, b); $("stb").appendChild(tr);
    if (!(cls in window)) { b.textContent = "not exposed by this browser"; continue; }
    try {
      const s = new window[cls]({frequency: 10});
      s.addEventListener("reading", () => { b.textContent = props.map(p => { const v = s[p]; return v == null ? "-" : (Array.isArray(v) || ArrayBuffer.isView(v)) ? Array.from(v).map(n => n.toFixed(2)).join(", ") : v.toFixed(2); }).join("  ") + " " + unit; });
      s.addEventListener("error", e => { b.textContent = "unavailable: " + e.error.name; });
      s.start(); live.push(s); b.textContent = "waiting for a reading...";
    } catch (e) { b.textContent = "unavailable: " + e.name; }
  }
  $("sstart").textContent = "Stop sensors"; $("sst").textContent = "Listing what the browser exposes.";
};

/* ---------- Vision ---------- */
const sessions = {};
ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = true; ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
const MODEL = {detect: "models/yolov8n.onnx", pose: "models/yolov8n-pose.onnx", classify: "models/mobilenetv2-7.onnx"};
async function getSession(t) { if (!sessions[t]) { $("vst").textContent = "Loading model " + MODEL[t].split("/")[1] + "..."; sessions[t] = await ort.InferenceSession.create(MODEL[t], {executionProviders: ["wasm"]}); } return sessions[t]; }
let src = null, stream = null; // src: canvas with current frame
const cv = $("cv");
function setSrc(c) { src = c; cv.width = c.width; cv.height = c.height; cv.getContext("2d").drawImage(c, 0, 0); $("run").disabled = false; }
$("pf").onchange = async e => { const f = e.target.files[0]; e.target.value = ""; if (!f) return; stopCam(); const im = await createImageBitmap(f); const c = document.createElement("canvas"); const k = Math.min(1, 1280 / Math.max(im.width, im.height)); c.width = Math.round(im.width * k); c.height = Math.round(im.height * k); c.getContext("2d").drawImage(im, 0, 0, c.width, c.height); setSrc(c); $("vst").textContent = "Photo loaded. Press Run."; };
function stopCam() { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; $("v").srcObject = null; $("cam").textContent = "Start camera"; }
$("cam").onclick = async () => {
  if (stream) { const v = $("v"), c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0); stopCam(); setSrc(c); $("vst").textContent = "Frame captured. Press Run."; return; }
  try { stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: {ideal: "environment"}, width: {ideal: 1280}}, audio: false}); } catch (e) { $("vst").textContent = "Camera unavailable (" + e.name + "). Pick a photo instead."; return; }
  const v = $("v"); v.srcObject = stream; await v.play(); $("cam").textContent = "Capture frame"; $("vst").textContent = "Camera on. Press Capture frame, then Run."; };
document.addEventListener("visibilitychange", () => { if (document.hidden) stopCam(); });
function letterbox(c) { const k = Math.min(S / c.width, S / c.height), nw = Math.round(c.width * k), nh = Math.round(c.height * k), px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  const w = document.createElement("canvas"); w.width = w.height = S; const g = w.getContext("2d", {willReadFrequently: true}); g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, S, S); g.drawImage(c, px, py, nw, nh);
  const d = g.getImageData(0, 0, S, S).data, inp = new Float32Array(3 * S * S), pl = S * S; for (let i = 0; i < pl; i++) { inp[i] = d[i*4] / 255; inp[pl+i] = d[i*4+1] / 255; inp[2*pl+i] = d[i*4+2] / 255; }
  return {inp, k, px, py}; }
function nms(list, th) { list.sort((a, b) => b.score - a.score); const kept = []; for (const a of list) { if (kept.every(b => { const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)), ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1)), i = iw * ih; return i / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - i + 1e-6) <= th; })) kept.push(a); } return kept; }
function box(o, n, i, lb) { const cx = o[i], cy = o[n+i], w = o[2*n+i], h = o[3*n+i]; return {x1: (cx-w/2-lb.px)/lb.k, y1: (cy-h/2-lb.py)/lb.k, x2: (cx+w/2-lb.px)/lb.k, y2: (cy+h/2-lb.py)/lb.k}; }
async function yolo(t, c) {
  const lb = letterbox(c), s = await getSession(t), t0 = performance.now();
  const res = await s.run({[s.inputNames[0]]: new ort.Tensor("float32", lb.inp, [1, 3, S, S])}), out = res[s.outputNames[0]], o = out.data, n = out.dims[2], dets = [];
  for (let i = 0; i < n; i++) {
    if (t === "detect") { let best = 0, bc = -1; for (let k = 0; k < 80; k++) { const v = o[(4+k)*n+i]; if (v > best) { best = v; bc = k; } } if (best < .25) continue; dets.push({...box(o, n, i, lb), score: best, cls: bc}); }
    else { const cf = o[4*n+i]; if (cf < .4) continue; const d = {...box(o, n, i, lb), score: cf, kp: []}; for (let k = 0; k < 17; k++) d.kp.push([(o[(5+k*3)*n+i]-lb.px)/lb.k, (o[(6+k*3)*n+i]-lb.py)/lb.k, o[(7+k*3)*n+i]]); dets.push(d); }
  }
  return {dets: nms(dets, .45), ms: Math.round(performance.now() - t0)};
}
function drawBoxes(list, t) { const g = cv.getContext("2d"); g.drawImage(src, 0, 0); g.lineWidth = Math.max(2, cv.width / 250); g.font = Math.max(14, cv.width / 40) + "px sans-serif";
  for (const d of list) { g.strokeStyle = "#ffd400"; g.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    if (t === "detect") { const txt = `${COCO[d.cls] || d.cls} ${Math.round(d.score*100)}%`; g.fillStyle = "#ffd400"; g.fillText(txt, d.x1 + 2, Math.max(14, d.y1 - 4)); }
    else { g.strokeStyle = "#4cd0a8"; for (const [a, b] of SKEL) if (d.kp[a][2] > .5 && d.kp[b][2] > .5) { g.beginPath(); g.moveTo(d.kp[a][0], d.kp[a][1]); g.lineTo(d.kp[b][0], d.kp[b][1]); g.stroke(); } g.fillStyle = "#ff4a55"; for (const p of d.kp) if (p[2] > .5) { g.beginPath(); g.arc(p[0], p[1], Math.max(3, cv.width / 200), 0, 7); g.fill(); } } } }
async function classify(c) {
  const s = await getSession("classify"), side = Math.min(c.width, c.height), w = document.createElement("canvas"); w.width = w.height = 224;
  const g = w.getContext("2d", {willReadFrequently: true}); g.drawImage(c, (c.width - side) / 2, (c.height - side) / 2, side, side, 0, 0, 224, 224);
  const d = g.getImageData(0, 0, 224, 224).data, pl = 224 * 224, inp = new Float32Array(3 * pl), mean = [.485, .456, .406], std = [.229, .224, .225];
  for (let i = 0; i < pl; i++) for (let k = 0; k < 3; k++) inp[k*pl+i] = (d[i*4+k] / 255 - mean[k]) / std[k];
  const t0 = performance.now(), r = await s.run({[s.inputNames[0]]: new ort.Tensor("float32", inp, [1, 3, 224, 224])}), lg = Array.from(r[s.outputNames[0]].data), ms = Math.round(performance.now() - t0);
  const mx = Math.max(...lg), ex = lg.map(v => Math.exp(v - mx)), sum = ex.reduce((a, b) => a + b, 0), top = ex.map((v, i) => [v / sum, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  return {txt: top.map(([p, i]) => `${(IMN[i] || "class " + i).split(",")[0].trim()} ${Math.round(p*100)}%`).join(" · ") + ` · ${ms} ms`, ms};
}
$("run").onclick = async () => {
  if (!src) return; $("run").disabled = true;
  try {
    if (!COCO.length) { COCO = (await (await fetch("data/coco.txt")).text()).trim().split("\n"); IMN = (await (await fetch("data/imagenet_synset.txt")).text()).trim().split("\n").map(l => l.slice(l.indexOf(" ") + 1)); }
    if (tab === "classify") { const r = await classify(src); $("vst").textContent = r.txt; }
    else { const r = await yolo(tab, src); drawBoxes(r.dets, tab);
      $("vst").textContent = tab === "detect" ? `${r.dets.length} objects: ` + Object.entries(r.dets.reduce((a, d) => (a[COCO[d.cls]] = (a[COCO[d.cls]] || 0) + 1, a), {})).map(([k, v]) => `${k} ${v}`).join(", ") + ` · ${r.ms} ms` : `${r.dets.length} people with pose · ${r.ms} ms`; }
  } catch (e) { $("vst").textContent = "Failed: " + e.message; }
  $("run").disabled = false;
};
