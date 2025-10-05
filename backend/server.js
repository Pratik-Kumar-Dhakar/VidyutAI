const WebSocket = require('ws');
const https = require('https');
const fs = require('fs');
const http = require('http');

// --- Gemini API Configuration ---
const GEMINI_API_KEY = 'AIzaSyDXhJm4gfKIHDxE3PgMGSh60HCBwQhMIHk';

// --- Other configs ---
const LAT = 23.22;
const LON = 72.68;
const CSV_FILE_PATH = './energy_log.csv';
const CSV_HEADER = 'timestamp,time_label,grid_kWh,solar_kWh,battery_kWh\n';

const BASE_GRID_DATA = [1.5,1.45,1.4,1.4,1.4,1.5,1.6,1.7,1.8,1.9,2,2.25,2.5,2.75,3,2.75,2.5,1.75,1,0.75,0.5,0.35,0.2,0.15,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.75,2.5,3,3.5,3.75,4,3.9,3.8,3.5,3.2,2.85,2.5,2.15,1.8,1.65,1.5,1.45,1.4,1.4,1.4,1.5,1.6,1.7,1.8,1.9,2,2.25,2.5,2.75,3,2.75,2.5,1.75,1,0.75,0.5,0.35,0.2,0.15,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.75,2.5,3,3.5,3.75,4,3.9,3.8,3.5,3.2,2.85,2.5,2.15];
const BASE_SOLAR_DATA = [0,0,0,0,0,0,0,0,0,0,0.1,0.2,0.5,0.8,1.15,1.5,2,2.5,3.25,4,4.75,5.5,5.75,6,5.9,5.8,5.5,5.2,4.85,4.5,4,3.5,2.75,2,1.25,0.5,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.2,0.5,0.8,1.15,1.5,2,2.5,3.25,4,4.75,5.5,5.75,6,5.9,5.8,5.5,5.2,4.85,4.5,4,3.5,2.75,2,1.25,0.5,0.25,0,0,0,0,0,0,0,0,0];
const BASE_BATTERY_DATA = [1,0.9,0.8,0.7,0.6,0.55,0.5,0.5,0.5,0.65,0.8,0.9,1,0.75,0.5,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0.5,0.75,1,1.25,1.5,1.65,1.8,1.9,2,1.9,1.8,1.5,1.2,1.1,1,0.9,0.8,0.7,0.6,0.55,0.5,0.5,0.5,0.65,0.8,0.9,1,0.75,0.5,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0.5,0.75,1,1.25,1.5,1.65,1.8,1.9,2,1.9,1.8];
let latestServerState = {};

// --- Full HTTP Server Setup to handle Gemini requests from the main panel ---
const server = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
    if (req.url === '/generate-insight' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', async () => {
            try {
                const requestData = JSON.parse(body);
                const insight = await getGeminiInsight(requestData.energyData);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ insight }));
            } catch (error) {
                console.error("Error processing Gemini request:", error);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Failed to generate insight' }));
            }
        });
    } else {
        res.writeHead(404); res.end();
    }
});

const STATIC_LOG_MESSAGES = ["Suggestion: Increase battery charging rate to capture peak solar.","Suggestion: Discharge battery to offset high grid cost.","Analysis: Demand is low. Prioritizing battery charge.","Analysis: Grid emissions are high. Maximizing solar and battery usage.","Suggestion: Reduce grid dependency. Current cost is at peak level.","Analysis: Solar production exceeds demand. Storing excess energy.","Suggestion: Pre-charge battery in anticipation of evening demand spike.","Analysis: Optimal energy distribution achieved for current conditions."];
server.listen(8080, () => {
    console.log('🚀 Server listening on http://localhost:8080');
});
const wss = new WebSocket.Server({ server });
console.log('✅ Server started on http://localhost:8080');

// (The rest of the file is largely the same, but the logInterval is replaced)
// ...




// --- MODIFIED: Function to get weather forecast from Open-Meteo ---
async function getWeatherForecast() {
    // We request only the next 24 hours of hourly cloud cover data.
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${LAT}&longitude=${LON}&hourly=cloud_cover&forecast_days=1`;

    return new Promise((resolve) => {
        https.get(url, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
                try {
                    const forecast = JSON.parse(data);
                    // The API returns an array of cloud cover percentages for each hour.
                    // We'll take the average of the next 12 hours.
                    const next12Hours = forecast.hourly.cloud_cover.slice(0, 12);
                    const avgClouds = next12Hours.reduce((acc, curr) => acc + curr, 0) / next12Hours.length;
                    
                    if (avgClouds > 70) {
                        resolve(`Weather Alert: Heavy cloud cover (${avgClouds.toFixed(0)}%) expected. Solar generation will be low. Conserve battery.`);
                    } else if (avgClouds > 40) {
                        resolve(`Weather Outlook: Partly cloudy (${avgClouds.toFixed(0)}%). Expect moderate solar generation.`);
                    } else {
                        resolve(`Weather Outlook: Clear skies (${avgClouds.toFixed(0)}% clouds) expected. Maximize solar charging for evening peak.`);
                    }
                } catch (e) {
                    console.error('Error parsing weather data:', e);
                    resolve(null);
                }
            });
        }).on('error', (err) => {
            console.error('Error fetching weather data:', err.message);
            resolve(null);
        });
    });
}

// --- The getGeminiInsight function is now used by both the log and the main panel ---
async function getGeminiInsight(dailyEnergyData) {

    if (!GEMINI_API_KEY) {
        return {
            action: "Configuration Error",
            reasoning: "API key is not configured in the client script."
        };
    }

    const formatDataset = (chartData) => {
        const dataset = chartData.datasets[0];
        const avg = (
            dataset.data.reduce((a, b) => a + b, 0) / dataset.data.length
        ).toFixed(2);
        return `average ${avg} kWh`;
    };

    const dailyTrend = `Overall, the daily trend shows: Grid usage of ${formatDataset(dailyEnergyData.grid)}, Solar generation of ${formatDataset(dailyEnergyData.solar)}, and Battery usage of ${formatDataset(dailyEnergyData.battery)}.`;


    const weatherForecast = getWeatherForecast()

    const userQuery = `Analyze the following energy report and provide a recommendation in the specified JSON format: {"action": string, "reasoning": string}.

1. Overall Daily Trend: ${dailyTrend}
2. Upcoming Weather: ${weatherForecast}`;

    const payload = {
        contents: [
            {
                parts: [{ text: userQuery }]
            }
        ]
    };

    try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        const text = result?.candidates?.[0]?.content?.parts?.[0]?.text.replace(/```json|```/g, '').trim();
	console.log(text)

        if (!text) {
            throw new Error("No response from Gemini");
        }

        // Try to parse Gemini's response into JSON
        try {
            return JSON.parse(text);
        } catch (e) {
            return {
                action: "Unstructured Response",
                reasoning: text
            };
        }

    } catch (err) {
        console.error("Gemini fetch error:", err.message);
        return {
            action: "Error",
            reasoning: "Failed to retrieve Gemini insight: " + err.message
        };
    }
}

// ... getWeatherForecast, LABELS, and other data arrays are unchanged ...

wss.on('connection', async ws => {
    console.log('Dashboard client connected.');

    // ... setup logic is the same ...
    let dataIndex = 0;
    let currentBatteryCharge = 50.0;
    let sessionGridData = [...BASE_GRID_DATA];
    let sessionSolarData = [...BASE_SOLAR_DATA];
    let sessionBatteryData = [...BASE_BATTERY_DATA];
    const weatherInsight = await getWeatherForecast();
    
    // This is needed for the first call to getGeminiInsight
    let sessionDailyData = {
        grid: { datasets: [{ label: 'Grid (kWh)', data: sessionGridData }] },
        solar: { datasets: [{ label: 'Solar (kWh)', data: sessionSolarData }] },
        battery: { datasets: [{ label: 'Battery (kWh)', data: sessionBatteryData }] }
    };

    const energyInterval = setInterval(() => {
        // ... energy simulation and CSV logging logic is the same ...
        latestServerState = { /* ... updated here ... */ };
        // ... ws.send for energyUpdate and batteryChargeUpdate ...
    }, 5000);

    // --- MODIFIED: Replaced logInterval with an async loop that calls Gemini ---
    let logTimeout;
    const sendInsightLog = async () => {
        try {
            // Call Gemini with the latest data
            const insight = await getGeminiInsight(sessionDailyData);
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'log',
                    payload: insight // The payload is now the {action, reasoning} object
                }));
            }
        } catch (error) {
            console.error("Error getting Gemini insight for log:", error.message);
        } finally {
            // Schedule the next call in 15 seconds
            if (ws.readyState === WebSocket.OPEN) {
               logTimeout = setTimeout(sendInsightLog, 15000);
            }
        }
    };
    
    // Start the first insight log
    sendInsightLog();


    ws.on('close', () => {
        console.log('Dashboard client disconnected.');
        clearInterval(energyInterval);
        clearTimeout(logTimeout); // --- MODIFIED: Clear the timeout
    });
});
