/* =========================================================
   PRISM-TC FRONTEND
   SIH26070
   ========================================================= */

const API_BASE = "/api";

let map;
let cycloneMarker;
let trackLine;
let windChart;


/* =========================================================
   INITIALIZE MAP
   ========================================================= */

function initializeMap() {

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
            Number(
                document.getElementById("latitude").value
            ),

        longitude:
            Number(
                document.getElementById("longitude").value
            ),

        sst:
            Number(
                document.getElementById("sst").value
            ),

        wind_shear:
            Number(
                document.getElementById("shear").value
            ),

        pressure:
            Number(
                document.getElementById("pressureInput").value
            ),

        wind:
            Number(
                document.getElementById("windInput").value
            ),

        humidity:
            Number(
                document.getElementById("humidity").value
            ),

        vorticity:
            Number(
                document.getElementById("vorticity").value
            )
    };

}


/* =========================================================
   API HELPER
   ========================================================= */

async function apiRequest(
    endpoint,
    options = {}
) {

    const response = await fetch(
        `${API_BASE}${endpoint}`,
        {
            headers: {
                "Content-Type":
                    "application/json"
            },

            ...options
        }
    );


    if (!response.ok) {

        throw new Error(
            `API error: ${response.status}`
        );

    }


    return response.json();

}


/* =========================================================
   CHECK HEALTH
   ========================================================= */

async function checkHealth() {

    const status =
        document.getElementById("aiStatus");

    try {

        const data =
            await apiRequest("/health");

        status.textContent =
            data.status || "ONLINE";

        status.style.color =
            "#36d399";

    } catch (error) {

        status.textContent =
            "OFFLINE";

        status.style.color =
            "#fb7185";

        console.error(error);

    }

}


/* =========================================================
   RUN DEMO
   ========================================================= */

async function runDemo() {

    setLoading(true);

    try {

        const data =
            await apiRequest("/demo");

        console.log(
            "Demo response:",
            data
        );

        updateDashboard(data);

        document.getElementById(
            "modeStatus"
        ).textContent = "Demo";

    } catch (error) {

        showError(
            "Demo failed. Make sure the Python and Node servers are running."
        );

        console.error(error);

    } finally {

        setLoading(false);

    }

}


/* =========================================================
   GENERATE PREDICTION
   ========================================================= */

async function generatePrediction() {

    const inputs =
        getInputs();

    setLoading(true);

    try {

        const data =
            await apiRequest(
                "/predict",
                {
                    method: "POST",

                    body:
                        JSON.stringify(inputs)
                }
            );

        console.log(
            "Prediction:",
            data
        );

        updateDashboard(data);

        document.getElementById(
            "modeStatus"
        ).textContent = "Live Input";

    } catch (error) {

        showError(
            "Prediction failed. Check that the backend is running."
        );

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


    const prediction =
        data.prediction ||
        data;


    /* CATEGORY */

    const category =
        prediction.category ||
        prediction.class_name ||
        prediction.cyclone_category ||
        "Unknown";

    document.getElementById(
        "category"
    ).textContent = category;


    /* WIND */

    const wind =
        prediction.wind ||
        prediction.predicted_wind ||
        prediction.wind_speed ||
        data.wind ||
        0;

    document.getElementById(
        "wind"
    ).textContent =
        formatNumber(wind);


    /* PRESSURE */

    const pressure =
        prediction.pressure ||
        data.pressure ||
        0;

    document.getElementById(
        "pressure"
    ).textContent =
        formatNumber(pressure);


    /* RISK */

    const risk =
        prediction.risk_index ||
        prediction.risk ||
        data.risk_index ||
        0;

    document.getElementById(
        "risk"
    ).textContent =
        formatNumber(risk);


    /* CONFIDENCE */

    const confidence =
        prediction.confidence ||
        data.confidence ||
        0;

    document.getElementById(
        "confidence"
    ).textContent =
        formatNumber(
            confidence
        );


    /* RAPID INTENSIFICATION */

    const rapid =
        prediction.rapid_intensification ||
        data.rapid_intensification ||
        false;

    document.getElementById(
        "rapidText"
    ).textContent =
        rapid
            ? "Potential rapid intensification detected."
            : "No rapid intensification detected.";


    /* RISK BADGE */

    updateRiskBadge(
        risk
    );


    /* TRACK */

    const track =
        data.track ||
        data.forecast_track ||
        prediction.track ||
        [];

    if (
        Array.isArray(track) &&
        track.length
    ) {

        drawTrack(
            track
        );

    }


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


/* =========================================================
   RISK BADGE
   ========================================================= */

function updateRiskBadge(
    risk
) {

    const badge =
        document.getElementById(
            "riskBadge"
        );


    badge.className =
        "badge";


    if (risk >= 75) {

        badge.classList.add(
            "danger"
        );

        badge.textContent =
            "HIGH RISK";

    } else if (risk >= 45) {

        badge.classList.add(
            "warning"
        );

        badge.textContent =
            "MODERATE RISK";

    } else {

        badge.classList.add(
            "safe"
        );

        badge.textContent =
            "LOW RISK";

    }

}


/* =========================================================
   DRAW CYCLONE TRACK
   ========================================================= */

function drawTrack(
    track
) {

    const coordinates =
        track
            .map(point => {

                if (
                    Array.isArray(point)
                ) {

                    return [
                        Number(point[0]),
                        Number(point[1])
                    ];

                }

                return [
                    Number(
                        point.latitude ??
                        point.lat
                    ),

                    Number(
                        point.longitude ??
                        point.lon ??
                        point.lng
                    )
                ];

            })
            .filter(
                point =>
                    Number.isFinite(
                        point[0]
                    ) &&
                    Number.isFinite(
                        point[1]
                    )
            );


    if (!coordinates.length)
        return;


    if (trackLine) {

        map.removeLayer(
            trackLine
        );

    }


    if (cycloneMarker) {

        map.removeLayer(
            cycloneMarker
        );

    }


    trackLine =
        L.polyline(
            coordinates,
            {
                weight: 4
            }
        ).addTo(map);


    const latest =
        coordinates[
            coordinates.length - 1
        ];


    cycloneMarker =
        L.marker(
            latest
        )
        .addTo(map)
        .bindPopup(
            "<strong>AI Cyclone Forecast</strong>"
        )
        .openPopup();


    map.fitBounds(
        trackLine.getBounds(),
        {
            padding: [30, 30]
        }
    );

}


/* =========================================================
   WIND CHART
   ========================================================= */

function updateChart(
    forecast
) {

    const labels = [];
    const values = [];


    forecast.forEach(
        (item, index) => {

            if (
                typeof item ===
                "number"
            ) {

                labels.push(
                    `T+${index * 6}h`
                );

                values.push(
                    item
                );

                return;
            }


            labels.push(
                item.time ||
                item.hour ||
                `T+${index * 6}h`
            );


            values.push(
                Number(
                    item.wind ||
                    item.predicted_wind ||
                    item.value ||
                    0
                )
            );

        }
    );


    renderChart(
        labels,
        values
    );

}


/* =========================================================
   CREATE CHART
   ========================================================= */

function renderChart(
    labels,
    values
) {

    const ctx =
        document
            .getElementById(
                "windChart"
            )
            .getContext("2d");


    if (windChart) {

        windChart.destroy();

    }


    windChart =
        new Chart(
            ctx,
            {

                type: "line",

                data: {

                    labels,

                    datasets: [

                        {

                            label:
                                "Wind (knots)",

                            data:
                                values,

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

                            labels: {

                                color:
                                    "#8fa5bf"

                            }

                        }

                    },

                    scales: {

                        x: {

                            ticks: {

                                color:
                                    "#8fa5bf"

                            },

                            grid: {

                                color:
                                    "rgba(255,255,255,0.05)"

                            }

                        },

                        y: {

                            ticks: {

                                color:
                                    "#8fa5bf"

                            },

                            grid: {

                                color:
                                    "rgba(255,255,255,0.05)"

                            }

                        }

                    }

                }

            }
        );

}


/* =========================================================
   LOADING STATE
   ========================================================= */

function setLoading(
    loading
) {

    const predictButton =
        document.getElementById(
            "predictBtn"
        );

    const demoButton =
        document.getElementById(
            "demoBtn"
        );


    if (loading) {

        predictButton.disabled =
            true;

        demoButton.disabled =
            true;

        predictButton.textContent =
            "⏳ Processing...";

    } else {

        predictButton.disabled =
            false;

        demoButton.disabled =
            false;

        predictButton.textContent =
            "🔮 Generate AI Prediction";

    }

}


/* =========================================================
   ERROR MESSAGE
   ========================================================= */

function showError(
    message
) {

    alert(
        message
    );

}


/* =========================================================
   NUMBER FORMAT
   ========================================================= */

function formatNumber(
    value
) {

    const number =
        Number(value);

    if (
        !Number.isFinite(number)
    ) {

        return "—";

    }

    return number.toFixed(
        1
    );

}


/* =========================================================
   EVENT LISTENERS
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initializeMap();

        checkHealth();


        document
            .getElementById(
                "demoBtn"
            )
            .addEventListener(
                "click",
                runDemo
            );


        document
            .getElementById(
                "predictBtn"
            )
            .addEventListener(
                "click",
                generatePrediction
            );


        document
            .getElementById(
                "healthBtn"
            )
            .addEventListener(
                "click",
                checkHealth
            );


        /* INITIAL MAP LOCATION */

        const lat =
            Number(
                document.getElementById(
                    "latitude"
                ).value
            );

        const lon =
            Number(
                document.getElementById(
                    "longitude"
                ).value
            );


        L.marker(
            [lat, lon]
        )
        .addTo(map)
        .bindPopup(
            "Initial cyclone position"
        );

    }
);