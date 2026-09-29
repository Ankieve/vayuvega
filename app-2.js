/* =========================================================
   PRISM-TC FRONTEND
   SIH26070

   Talks to the backend in backend/server.py:
     GET  /api/health    GET /api/samples    GET /api/demo
     POST /api/predict
   REAL      : category, confidence, class probabilities (AI model)
   SIMULATED : track, wind outlook, pressure, risk, rapid flag
   ========================================================= */

/* When the page is opened with VS Code Live Server (port 5500) or as a
   file, talk to the backend on port 8000. Override with ?api=URL */
const DEFAULT_API =
    (location.protocol === "file:" || location.port === "5500")
        ? "http://localhost:8000/api"
        : "/api";

const API_BASE =
    new URLSearchParams(location.search).get("api") || DEFAULT_API;

const API_TIMEOUT_MS = 60000;   /* first model call can be slow */

const CLASS_ORDER = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe/Super Cyclone"
];

const CLASS_COLORS = [
    "#4C9F70", "#E0B040", "#E8873A", "#D6493B", "#8B1E3F"
];

let map;
let hasLeaflet = false;
let trackLayers = [];
let windChart;
let samples = [];

/* the image chosen by the user: null | {kind, name, sample?, dataUrl?, trueClass?} */
let selectedImage = null;
let toastTimer;


/* =========================================================
   SMALL HELPERS
   ========================================================= */

function el(id) {
    return document.getElementById(id);
}

function showError(message) {
    showToast(message, "error");
}

function showToast(message, kind) {
    const box = el("toast");
    box.textContent = message;
    box.className = "toast " + (kind || "info");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        box.classList.add("hidden");
    }, 7000);
}

function formatNumber(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "—";
    }
    return number.toFixed(1);
}

function setTag(id, text, kind) {
    const tag = el(id);
    if (!text) {
        tag.classList.add("hidden");
        return;
    }
    tag.textContent = text;
    tag.className = "tag " + kind;
}


/* =========================================================
   INITIALIZE MAP
   ========================================================= */

function initializeMap() {

    hasLeaflet = typeof L !== "undefined";

    if (!hasLeaflet) {
        /* offline / CDN blocked: we draw a simple SVG plot instead */
        el("map").innerHTML =
            '<div class="map-fallback">Map library not loaded ' +
            "(no internet?). A simple track plot will be shown.</div>";
        return;
    }

    map = L.map("map").setView(
        [15.5, 85],
        5
    );

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution:
                "&copy; OpenStreetMap contributors"
        }
    ).addTo(map);

}


/* =========================================================
   READ FORM VALUES
   ========================================================= */

function getInputs() {

    return {

        latitude:
            Number(el("latitude").value),

        longitude:
            Number(el("longitude").value),

        sst:
            Number(el("sst").value),

        wind_shear:
            Number(el("shear").value),

        pressure:
            Number(el("pressureInput").value),

        wind:
            Number(el("windInput").value),

        humidity:
            Number(el("humidity").value),

        vorticity:
            Number(el("vorticity").value)
    };

}


/* =========================================================
   API HELPER
   ========================================================= */

async function apiRequest(endpoint, options = {}) {

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    let response;

    try {

        response = await fetch(
            `${API_BASE}${endpoint}`,
            {
                headers: {
                    "Content-Type": "application/json"
                },
                signal: controller.signal,
                ...options
            }
        );

    } catch (error) {

        if (error.name === "AbortError") {
            throw new Error("The server took too long to answer.");
        }
        throw new Error(
            "Cannot reach the backend. Start it with: python backend/server.py"
        );

    } finally {

        clearTimeout(timer);

    }

    let payload = null;

    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }

    if (!response.ok) {
        throw new Error(
            (payload && payload.error) ||
            `API error: ${response.status}`
        );
    }

    return payload;

}


/* =========================================================
   CHECK HEALTH
   ========================================================= */

async function checkHealth() {

    const status = el("aiStatus");
    const dot = el("statusDot");

    try {

        const data = await apiRequest("/health");

        if (data.model_loaded) {

            status.textContent = "ONLINE · model loaded";
            status.style.color = "#36d399";
            el("sysTitle").textContent = "System Online";
            el("sysSub").textContent = "AI model ready";
            dot.className = "status-dot";

        } else {

            status.textContent = "DEMO MODE · no model file";
            status.style.color = "#fbbf24";
            el("sysTitle").textContent = "Demo Mode";
            el("sysSub").textContent = "Placeholder predictions";
            dot.className = "status-dot warn";

        }

    } catch (error) {

        status.textContent = "OFFLINE";
        status.style.color = "#fb7185";
        el("sysTitle").textContent = "Backend Offline";
        el("sysSub").textContent = "Start backend/server.py";
        dot.className = "status-dot bad";

        console.error(error);

    }

}


/* =========================================================
   SAMPLE IMAGES + IMAGE SELECTION
   ========================================================= */

async function loadSamples() {

    try {

        samples = await apiRequest("/samples");

        const select = el("sampleSelect");

        samples.forEach((sample, index) => {

            const option = document.createElement("option");
            option.value = sample.filename;
            option.textContent =
                `Sample ${index + 1} · storm ${sample.storm_id}`;
            select.appendChild(option);

        });

    } catch (error) {

        console.error(error);

    }

}

function showPreview(src, name, info) {

    el("preview").src = src;
    el("previewName").textContent = name;
    el("previewInfo").textContent = info || "";
    el("previewBox").classList.remove("hidden");

}

function clearImage() {

    selectedImage = null;
    el("previewBox").classList.add("hidden");
    el("preview").removeAttribute("src");
    el("sampleSelect").value = "";
    el("imageInput").value = "";

}

function chooseSample(filename) {

    if (!filename) {
        clearImage();
        return;
    }

    const sample = samples.find(s => s.filename === filename);

    if (!sample) {
        return;
    }

    selectedImage = {
        kind: "sample",
        name: sample.filename,
        sample: sample.filename,
        trueClass: sample.true_class
    };

    el("imageInput").value = "";
    el("latitude").value = sample.lat;
    el("longitude").value = sample.lon;

    showPreview(
        sample.url,
        sample.filename,
        `Sample from the test set · storm ${sample.storm_id}`
    );

}

function onFileChosen(event) {

    const file = event.target.files && event.target.files[0];

    if (!file) {
        return;
    }

    if (!["image/png", "image/jpeg"].includes(file.type)) {
        showError("Please choose a PNG or JPG image.");
        event.target.value = "";
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image is larger than 10 MB. Please choose a smaller one.");
        event.target.value = "";
        return;
    }

    const reader = new FileReader();

    reader.onload = () => {

        selectedImage = {
            kind: "upload",
            name: file.name,
            dataUrl: reader.result
        };

        el("sampleSelect").value = "";

        showPreview(
            reader.result,
            file.name,
            `${(file.size / 1024).toFixed(0)} KB · uploaded`
        );

    };

    reader.onerror = () => {
        showError("Could not read that file.");
    };

    reader.readAsDataURL(file);

}


/* =========================================================
   RUN DEMO
   ========================================================= */

async function runDemo() {

    setLoading(true);

    try {

        const data = await apiRequest("/demo");

        console.log(
            "Demo response:",
            data
        );

        /* show which sample the backend used */
        if (data.meta && data.meta.image) {
            el("sampleSelect").value = data.meta.image;
            chooseSample(data.meta.image);
            if (data.meta.position) {
                el("latitude").value = data.meta.position[0];
                el("longitude").value = data.meta.position[1];
            }
        }

        updateDashboard(data);

        el("modeStatus").textContent = "Demo";

    } catch (error) {

        showError(error.message);

        console.error(error);

    } finally {

        setLoading(false);

    }

}


/* =========================================================
   GENERATE PREDICTION
   ========================================================= */

async function generatePrediction() {

    const inputs = getInputs();

    const body = { ...inputs };

    if (selectedImage && selectedImage.kind === "sample") {
        body.sample = selectedImage.sample;
    } else if (selectedImage && selectedImage.kind === "upload") {
        body.image = selectedImage.dataUrl;
    }

    setLoading(true);

    try {

        const data = await apiRequest(
            "/predict",
            {
                method: "POST",
                body: JSON.stringify(body)
            }
        );

        console.log(
            "Prediction:",
            data
        );

        updateDashboard(data);

        el("modeStatus").textContent =
            selectedImage ? "Live image" : "Rule-based (no image)";

    } catch (error) {

        showError(error.message);

        console.error(error);

    } finally {

        setLoading(false);

    }

}


/* =========================================================
   UPDATE DASHBOARD
   ========================================================= */

function updateDashboard(data) {

    console.log(
        "Dashboard data:",
        data
    );

    const prediction = data.prediction || data;
    const sources = data.sources || {};
    const meta = data.meta || {};


    /* CATEGORY */

    const category =
        prediction.category ||
        prediction.class_name ||
        prediction.cyclone_category ||
        "Unknown";

    el("category").textContent = category;

    const catSource = sources.category;

    if (catSource === "model") {
        setTag("tagCategory", "MODEL", "real");
    } else if (catSource === "placeholder") {
        setTag("tagCategory", "PLACEHOLDER", "sim");
    } else if (catSource === "rule") {
        setTag("tagCategory", "RULE", "sim");
    } else {
        setTag("tagCategory", "", "");
    }


    /* WIND: class range when it came from an image */

    const wind = prediction.wind ?? prediction.predicted_wind ??
        prediction.wind_speed ?? data.wind;

    el("wind").textContent =
        prediction.wind_display || formatNumber(wind);

    el("windUnit").textContent =
        meta.mode === "image" ? "knots (class range)" : "knots";

    setTag(
        "tagWind",
        sources.wind === "estimate" ? "EST." :
        sources.wind === "input" ? "INPUT" : "",
        sources.wind === "input" ? "real" : "plan"
    );


    /* PRESSURE */

    const pressure = prediction.pressure ?? data.pressure;

    el("pressure").textContent = formatNumber(pressure);

    setTag(
        "tagPressure",
        sources.pressure === "simulated" ? "SIM" :
        sources.pressure === "input" ? "INPUT" : "",
        sources.pressure === "input" ? "real" : "sim"
    );


    /* RISK */

    const risk =
        prediction.risk_index ??
        prediction.risk ??
        data.risk_index ??
        0;

    el("risk").textContent = formatNumber(risk);

    setTag("tagRisk", sources.risk ? "SIM" : "", "sim");


    /* CONFIDENCE (null when no AI prediction) */

    const confidence = prediction.confidence ?? data.confidence;

    el("confidence").textContent = formatNumber(confidence);

    setTag(
        "tagConf",
        confidence === null || confidence === undefined ? "" :
        (sources.confidence === "placeholder" ? "PLACEHOLDER" : "MODEL"),
        sources.confidence === "placeholder" ? "sim" : "real"
    );


    /* PROBABILITIES */

    renderProbabilities(data.probabilities, category);


    /* PROVENANCE LINE */

    renderProvenance(category, meta, sources);


    /* RAPID INTENSIFICATION */

    const rapid =
        prediction.rapid_intensification ??
        data.rapid_intensification ??
        false;

    el("rapidText").textContent =
        rapid
            ? "Simulated outlook shows a possible wind rise of 30 kt or more within 24 h."
            : "No rapid intensification in the simulated 24 h outlook.";


    /* RISK BADGE */

    updateRiskBadge(Number(risk));


    /* TRACK */

    const track =
        data.track ||
        data.forecast_track ||
        prediction.track ||
        [];

    drawTrack(data.past_track || [], track);


    /* CHART */

    const forecast =
        data.forecast ||
        data.wind_forecast ||
        prediction.forecast ||
        [];

    if (
        Array.isArray(forecast) &&
        forecast.length
    ) {

        updateChart(
            forecast
        );

    }

}


function renderProvenance(category, meta, sources) {

    const line = el("provenance");
    line.className = "provenance";
    const parts = [];

    if (sources.category === "model") {

        parts.push(
            `AI prediction from ${meta.image || "the image"}.`
        );

        if (selectedImage && selectedImage.trueClass) {
            const match = selectedImage.trueClass === category;
            parts.push(
                `Dataset label: ${selectedImage.trueClass} — ` +
                (match ? "match." : "not a match.")
            );
        }

    } else if (sources.category === "placeholder") {

        parts.push(
            "DEMO MODE: model file not found, so this result is a placeholder."
        );
        line.classList.add("warn");

    } else if (sources.category === "rule") {

        parts.push(
            "Category derived from the entered wind speed (IMD rule), not from the AI model."
        );

    }

    if (meta.analog_storm) {
        parts.push(
            `Simulated outlook copies storm ${meta.analog_storm}.`
        );
    }

    /* only extra warnings (e.g. low confidence); the two cases above already explain themselves */
    (meta.warnings || []).forEach(w => {
        if (!w.startsWith("No image provided") && !w.startsWith("DEMO MODE")) {
            parts.push(w);
        }
    });

    line.textContent = parts.join(" ");

}


function renderProbabilities(probs, predicted) {

    const box = el("probBars");
    box.innerHTML = "";

    if (!probs) {
        const p = document.createElement("p");
        p.className = "hint";
        p.textContent = "No AI probabilities for a rule-based result.";
        box.appendChild(p);
        return;
    }

    CLASS_ORDER.forEach((name, i) => {

        const value = Number(probs[name] ?? 0);

        const row = document.createElement("div");
        row.className = "prob-row" + (name === predicted ? " top" : "");

        const label = document.createElement("span");
        label.className = "prob-name";
        label.textContent = name;

        const track = document.createElement("div");
        track.className = "prob-track";

        const fill = document.createElement("div");
        fill.className = "prob-fill";
        fill.style.width = Math.max(0, Math.min(100, value)) + "%";
        fill.style.background = CLASS_COLORS[i];
        track.appendChild(fill);

        const num = document.createElement("b");
        num.textContent = value.toFixed(1) + "%";

        row.append(label, track, num);
        box.appendChild(row);

    });

}


/* =========================================================
   RISK BADGE
   ========================================================= */

function updateRiskBadge(risk) {

    const badge = el("riskBadge");

    badge.className = "badge";

    if (risk >= 75) {

        badge.classList.add("danger");
        badge.textContent = "HIGH RISK";

    } else if (risk >= 45) {

        badge.classList.add("warning");
        badge.textContent = "MODERATE RISK";

    } else {

        badge.classList.add("safe");
        badge.textContent = "LOW RISK";

    }

}


/* =========================================================
   DRAW CYCLONE TRACK
   ========================================================= */

function toCoordinates(track) {

    return track
        .map(point => {

            if (Array.isArray(point)) {
                return [Number(point[0]), Number(point[1])];
            }

            return [
                Number(point.latitude ?? point.lat),
                Number(point.longitude ?? point.lon ?? point.lng)
            ];

        })
        .filter(
            point =>
                Number.isFinite(point[0]) &&
                Number.isFinite(point[1])
        );

}

function drawTrack(pastTrack, futureTrack) {

    const past = toCoordinates(Array.isArray(pastTrack) ? pastTrack : []);
    const future = toCoordinates(Array.isArray(futureTrack) ? futureTrack : []);

    if (!past.length && !future.length) {
        return;
    }

    if (!hasLeaflet) {
        drawTrackFallback(past, future);
        return;
    }

    trackLayers.forEach(layer => map.removeLayer(layer));
    trackLayers = [];

    const bounds = [];

    if (past.length > 1) {
        trackLayers.push(
            L.polyline(past, { color: "#19b5fe", weight: 4 })
                .addTo(map)
                .bindTooltip("Observed track (IBTrACS)")
        );
    }

    if (future.length > 1) {
        trackLayers.push(
            L.polyline(
                future,
                { color: "#ff9f43", weight: 4, dashArray: "8 8" }
            )
            .addTo(map)
            .bindTooltip("Simulated 24 h outlook")
        );
    }

    const now = future[0] || past[past.length - 1];

    trackLayers.push(
        L.circleMarker(now, {
            radius: 9, color: "#ffffff", weight: 2,
            fillColor: "#fb7185", fillOpacity: 1
        })
        .addTo(map)
        .bindPopup("<strong>Current position</strong>")
        .openPopup()
    );

    if (future.length > 1) {
        trackLayers.push(
            L.circleMarker(future[future.length - 1], {
                radius: 6, color: "#ffffff", weight: 2,
                fillColor: "#ff9f43", fillOpacity: 1
            })
            .addTo(map)
            .bindTooltip("Simulated position in 24 h")
        );
    }

    past.concat(future).forEach(p => bounds.push(p));

    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 7 });

}

/* Simple SVG track plot used when Leaflet cannot load */
function drawTrackFallback(past, future) {

    const all = past.concat(future);
    const lats = all.map(p => p[0]);
    const lons = all.map(p => p[1]);
    const minLat = Math.min(...lats) - 1;
    const maxLat = Math.max(...lats) + 1;
    const minLon = Math.min(...lons) - 1;
    const maxLon = Math.max(...lons) + 1;

    const W = 700, H = 460, pad = 30;
    const sx = v => pad + (v - minLon) / (maxLon - minLon || 1) * (W - 2 * pad);
    const sy = v => H - pad - (v - minLat) / (maxLat - minLat || 1) * (H - 2 * pad);
    const pts = list => list.map(p => `${sx(p[1]).toFixed(1)},${sy(p[0]).toFixed(1)}`).join(" ");

    const now = future[0] || past[past.length - 1];
    const end = future[future.length - 1];

    el("map").innerHTML =
        `<svg viewBox="0 0 ${W} ${H}" class="track-svg" role="img" aria-label="Cyclone track">` +
        `<text x="${pad}" y="20" class="svg-note">Simple track plot (map tiles unavailable)</text>` +
        (past.length > 1 ? `<polyline points="${pts(past)}" fill="none" stroke="#19b5fe" stroke-width="4"/>` : "") +
        (future.length > 1 ? `<polyline points="${pts(future)}" fill="none" stroke="#ff9f43" stroke-width="4" stroke-dasharray="8 8"/>` : "") +
        `<circle cx="${sx(now[1])}" cy="${sy(now[0])}" r="9" fill="#fb7185" stroke="#fff" stroke-width="2"/>` +
        (future.length > 1 ? `<circle cx="${sx(end[1])}" cy="${sy(end[0])}" r="6" fill="#ff9f43" stroke="#fff" stroke-width="2"/>` : "") +
        `<text x="${W - pad}" y="${H - 8}" text-anchor="end" class="svg-note">lat ${minLat.toFixed(0)}–${maxLat.toFixed(0)}, lon ${minLon.toFixed(0)}–${maxLon.toFixed(0)}</text>` +
        `</svg>`;

}


/* =========================================================
   WIND CHART
   ========================================================= */

function updateChart(forecast) {

    const labels = [];
    const values = [];

    forecast.forEach(
        (item, index) => {

            if (typeof item === "number") {
                labels.push(`T+${index * 6}h`);
                values.push(item);
                return;
            }

            labels.push(
                item.time ||
                item.hour ||
                `T+${index * 6}h`
            );

            values.push(
                Number(
                    item.wind ??
                    item.predicted_wind ??
                    item.value ??
                    0
                )
            );

        }
    );

    renderChart(labels, values);

}

function renderChart(labels, values) {

    const canvas = el("windChart");

    if (typeof Chart === "undefined") {
        drawChartFallback(canvas, labels, values);
        return;
    }

    const ctx = canvas.getContext("2d");

    if (windChart) {
        windChart.destroy();
    }

    windChart = new Chart(
        ctx,
        {

            type: "line",

            data: {

                labels,

                datasets: [

                    {

                        label: "Wind (knots) - simulated",

                        data: values,

                        borderColor: "#19b5fe",

                        backgroundColor: "rgba(25,181,254,0.12)",

                        borderWidth: 3,

                        tension: 0.35,

                        pointRadius: 4,

                        fill: true

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        labels: { color: "#8fa5bf" }
                    }

                },

                scales: {

                    x: {
                        ticks: { color: "#8fa5bf" },
                        grid: { color: "rgba(255,255,255,0.05)" }
                    },

                    y: {
                        ticks: { color: "#8fa5bf" },
                        grid: { color: "rgba(255,255,255,0.05)" }
                    }

                }

            }

        }
    );

}

/* Plain-canvas line chart used when Chart.js cannot load */
function drawChartFallback(canvas, labels, values) {

    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600;
    const h = canvas.clientHeight || 260;

    canvas.width = w * ratio;
    canvas.height = h * ratio;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const padL = 44, padR = 14, padT = 16, padB = 30;
    const lo = Math.floor(Math.min(...values) / 10) * 10 - 10;
    const hi = Math.ceil(Math.max(...values) / 10) * 10 + 10;
    const x = i => padL + i / Math.max(values.length - 1, 1) * (w - padL - padR);
    const y = v => padT + (1 - (v - lo) / (hi - lo || 1)) * (h - padT - padB);

    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#8fa5bf";
    ctx.strokeStyle = "rgba(255,255,255,0.06)";

    for (let g = 0; g <= 4; g++) {
        const v = lo + (hi - lo) * g / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y(v));
        ctx.lineTo(w - padR, y(v));
        ctx.stroke();
        ctx.fillText(v.toFixed(0), 8, y(v) + 4);
    }

    labels.forEach((label, i) => {
        if (i % 2 === 0) {
            ctx.fillText(label, x(i) - 12, h - 10);
        }
    });

    ctx.beginPath();
    values.forEach((v, i) => {
        if (i === 0) { ctx.moveTo(x(i), y(v)); } else { ctx.lineTo(x(i), y(v)); }
    });
    ctx.strokeStyle = "#19b5fe";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = "#19b5fe";
    values.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(x(i), y(v), 4, 0, Math.PI * 2);
        ctx.fill();
    });

}


/* =========================================================
   LOADING STATE
   ========================================================= */

function setLoading(loading) {

    const predictButton = el("predictBtn");
    const demoButton = el("demoBtn");

    predictButton.disabled = loading;
    demoButton.disabled = loading;

    predictButton.textContent =
        loading
            ? "⏳ Processing..."
            : "🔮 Generate AI Prediction";

}


/* =========================================================
   LIVE / NEAR-REAL-TIME SATELLITE

   Polls the backend's /api/satellite/* routes (see backend/satellite/).
   Everything here is defensive: if the satellite feature is unavailable
   (503), disabled, or the fetch itself times out, this panel shows a
   clear status instead of breaking the rest of the page. It never talks
   to a satellite provider directly - only to our own backend.
   ========================================================= */

const SAT_POLL_MS = 60000;         /* refresh the panel every 60s */
const SAT_BADGE_KIND = {
    LIVE: "safe",
    STALE: "warning",
    UPDATING: "neutral",
    ERROR: "danger",
    NO_DATA: "neutral"
};
let satPollTimer = null;

function satFormatAge(isoTime) {
    if (!isoTime) return "—";
    const then = new Date(isoTime).getTime();
    if (Number.isNaN(then)) return "—";
    const minutes = Math.max(0, Math.round((Date.now() - then) / 60000));
    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours} h ${minutes % 60} min ago`;
}

function satRenderState(state) {

    const status = state.status || "NO_DATA";
    const badge = el("satBadge");
    badge.textContent = status.replace("_", " ");
    badge.className = "badge " + (SAT_BADGE_KIND[status] || "neutral");

    el("satSource").textContent = state.source || "—";
    el("satSatellite").textContent = state.satellite || "—";
    el("satProduct").textContent = state.product || "—";
    el("satObsTime").textContent = state.observation_time_utc || "—";
    el("satAge").textContent = satFormatAge(state.observation_time_utc);
    el("satLastCheck").textContent = satFormatAge(state.last_check_utc) === "—"
        ? "—" : `${satFormatAge(state.last_check_utc)} (${state.last_check_utc || ""})`;
    el("satLastSuccess").textContent = state.last_success_utc
        ? satFormatAge(state.last_success_utc) : "—";

    if (state.region && state.region.coverage_fraction != null) {
        const pct = Math.round(state.region.coverage_fraction * 100);
        el("satCoverage").textContent = state.region.fully_visible
            ? "Full requested region visible"
            : `~${pct}% of the North Indian Ocean box visible from this satellite`;
    } else if (state.region && state.region.description) {
        el("satCoverage").textContent = state.region.description;
    } else {
        el("satCoverage").textContent = "—";
    }

    const img = el("satImage");
    const placeholder = el("satPlaceholder");
    if (status === "LIVE" || status === "STALE") {
        img.src = `${API_BASE}/satellite/image?ts=${Date.now()}`;
        img.classList.remove("hidden");
        placeholder.classList.add("hidden");
    } else {
        img.classList.add("hidden");
        placeholder.classList.remove("hidden");
        placeholder.textContent = status === "UPDATING"
            ? "Checking for a new satellite image..."
            : "No satellite image captured yet.";
    }

    const errorBox = el("satErrorBox");
    if (state.error) {
        errorBox.textContent = `${state.error_stage ? `[${state.error_stage}] ` : ""}${state.error}`;
        errorBox.classList.remove("hidden");
    } else {
        errorBox.classList.add("hidden");
    }
}

async function refreshSatellite() {
    try {
        const state = await apiRequest("/satellite/status");
        satRenderState(state);
    } catch (error) {
        // The satellite feature failing must never look like the whole
        // app is broken - show a scoped message in this panel only.
        el("satBadge").textContent = "UNAVAILABLE";
        el("satBadge").className = "badge neutral";
        el("satErrorBox").textContent =
            "Could not reach the satellite status endpoint: " + error.message;
        el("satErrorBox").classList.remove("hidden");
    }
}

function startSatellitePolling() {
    refreshSatellite();
    if (satPollTimer) clearInterval(satPollTimer);
    satPollTimer = setInterval(refreshSatellite, SAT_POLL_MS);
}


/* =========================================================
   EVENT LISTENERS
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initializeMap();

        checkHealth();
        loadSamples();
        startSatellitePolling();

        el("demoBtn").addEventListener("click", runDemo);
        el("predictBtn").addEventListener("click", generatePrediction);
        el("healthBtn").addEventListener("click", checkHealth);
        el("satRefreshBtn").addEventListener("click", refreshSatellite);
        el("imageInput").addEventListener("change", onFileChosen);
        el("clearImage").addEventListener("click", clearImage);
        el("sampleSelect").addEventListener(
            "change",
            event => chooseSample(event.target.value)
        );

        /* INITIAL MAP MARKER */

        if (hasLeaflet) {

            const lat = Number(el("latitude").value);
            const lon = Number(el("longitude").value);

            trackLayers.push(
                L.circleMarker([lat, lon], {
                    radius: 8, color: "#ffffff", weight: 2,
                    fillColor: "#19b5fe", fillOpacity: 1
                })
                .addTo(map)
                .bindPopup("Initial cyclone position")
            );

        }

    }
);
