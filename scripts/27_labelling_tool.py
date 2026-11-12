"""Build a local labelling page for the 350 stratified validation points (open it in a browser).

    python scripts/27_labelling_tool.py      -> tools/label_points.html

Points are shown on Esri World Imagery. Keys 1-7 set the class, U = unsure, arrows move between
points; progress is saved in the browser; 'Export CSV' downloads id, lon, lat, map class, reference
class. Score it with scripts/28_score_validation.py. The map class is hidden while labelling.
"""
import json

import geopandas as gpd

from kampala.config import OUTPUTS, ROOT

pts = gpd.read_file(OUTPUTS / "tables" / "m1_validation_points_2025.gpkg").to_crs("EPSG:4326")
pts = pts.sample(frac=1, random_state=3).reset_index(drop=True)
data = [dict(id=int(i), lon=round(g.x, 6), lat=round(g.y, 6), map=int(m)) for i, (g, m) in enumerate(zip(pts.geometry, pts.map_class))]
html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Land-cover validation · Kampala 2025</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body{margin:0;font:14px/1.4 -apple-system,Helvetica,Arial,sans-serif;display:grid;grid-template-columns:1fr 320px;height:100vh}
#map{height:100vh} aside{padding:16px;display:grid;gap:10px;align-content:start;border-left:1px solid #ddd;background:#fafaf8}
button{font:inherit;padding:8px 10px;border:1px solid #ccc;background:#fff;border-radius:6px;cursor:pointer;text-align:left}
button.on{background:#0b0b0b;color:#fff;border-color:#0b0b0b} .k{display:inline-block;width:18px;font-weight:700}
.row{display:flex;gap:8px} .row button{flex:1;text-align:center} h1{font-size:16px;margin:0} .muted{color:#777;font-size:12px}
</style></head><body><div id="map"></div><aside>
<h1>Kampala land cover · 2025 check</h1><div id="prog" class="muted"></div>
<div>What covers the point (red ring; yellow box = the 10 m pixel) in 2025?</div>
<div id="btns" style="display:grid;gap:6px"></div>
<div class="row"><button id="prev">← Prev</button><button id="next">Next →</button></div>
<button id="exp">Export CSV</button>
<p class="muted">Built dense: roofs cover most of the pixel and surroundings. Built sparse: buildings with gardens or yards between.
Bare soil: cleared plots, murram, fallow without cover. Wetland: papyrus or flooded vegetation. Imagery dates vary; if the spot has
clearly changed, use your best judgement or U.</p></aside>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const PTS = __DATA__;
const CL = ["built dense","built sparse","tree cover","grass / cropland","bare soil","wetland","water"];
let lab = {}; try { lab = JSON.parse(localStorage.getItem("kla_labels") || "{}"); } catch (e) {}
let i = 0; while (i < PTS.length - 1 && lab[PTS[i].id] !== undefined) i++;
const map = L.map("map", {maxZoom: 20});
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {maxZoom: 20, maxNativeZoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics"}).addTo(map);
let ring = null, box = null; const btns = document.getElementById("btns");
CL.forEach((c, k) => { const b = document.createElement("button"); b.innerHTML = `<span class="k">${k + 1}</span>${c}`; b.onclick = () => set(k + 1); btns.appendChild(b); });
const ub = document.createElement("button"); ub.innerHTML = '<span class="k">U</span>unsure'; ub.onclick = () => set(0); btns.appendChild(ub);
function show() {
  const p = PTS[i]; map.setView([p.lat, p.lon], 19);
  if (ring) map.removeLayer(ring); if (box) map.removeLayer(box);
  ring = L.circle([p.lat, p.lon], {radius: 12, color: "#e00", weight: 2, fill: false}).addTo(map);
  const d = 0.000045; box = L.rectangle([[p.lat - d, p.lon - d], [p.lat + d, p.lon + d]], {color: "#ff0", weight: 1, fill: false}).addTo(map);
  document.getElementById("prog").textContent = `Point ${i + 1} of ${PTS.length} · labelled ${Object.keys(lab).length}`;
  [...btns.children].forEach((b, k) => b.classList.toggle("on", lab[p.id] === (k < 7 ? k + 1 : 0)));
}
function set(v) { lab[PTS[i].id] = v; try { localStorage.setItem("kla_labels", JSON.stringify(lab)); } catch (e) {} if (i < PTS.length - 1) i++; show(); }
document.getElementById("prev").onclick = () => { if (i > 0) i--; show(); };
document.getElementById("next").onclick = () => { if (i < PTS.length - 1) i++; show(); };
document.addEventListener("keydown", e => { if (e.key >= "1" && e.key <= "7") set(+e.key); else if (e.key.toLowerCase() === "u") set(0);
  else if (e.key === "ArrowRight") document.getElementById("next").click(); else if (e.key === "ArrowLeft") document.getElementById("prev").click(); });
document.getElementById("exp").onclick = () => {
  const rows = ["id,lon,lat,map_class,ref_class"].concat(PTS.map(p => [p.id, p.lon, p.lat, p.map, lab[p.id] ?? ""].join(",")));
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.join("\\n")], {type: "text/csv"}));
  a.download = "kampala_validation_2025.csv"; a.click();
};
show();
</script></body></html>"""
(ROOT / "tools").mkdir(exist_ok=True)
(ROOT / "tools" / "label_points.html").write_text(html.replace("__DATA__", json.dumps(data)))
print("wrote tools/label_points.html with", len(data), "points")
