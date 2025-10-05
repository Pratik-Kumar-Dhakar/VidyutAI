
// server.js
// Run: node server.js
import express from "express";
import { spawn } from "child_process";
import bodyParser from "body-parser";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(bodyParser.json());

app.post("/ingest", (req, res) => {
  const payload = req.body;

  const py = spawn("python3", ["inference.py"], {
    cwd: __dirname,
    stdio: ["pipe", "pipe", "inherit"],
  });

  let output = "";
  py.stdout.on("data", (data) => {
    output += data.toString();
  });

  py.on("close", (code) => {
    try {
      const result = JSON.parse(output.trim());
      res.json(result);
    } catch (err) {
      res.status(500).json({ ok: false, error: "Failed to parse Python output", raw: output });
    }
  });

  py.stdin.write(JSON.stringify(payload));
  py.stdin.end();
});

app.listen(8000, () => {
  console.log("🚀 Express server running on http://localhost:8000");
});
