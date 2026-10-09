// PM2 process file for a plain Ubuntu server (no Docker).
//
//   cd <repo>/speech-judge
//   pm2 start deploy/pm2/ecosystem.config.js --only judge-api                  # Layout A: one process, models inside the API
//   pm2 start deploy/pm2/ecosystem.config.js --only judge-api,judge-worker     # Layout B: API gateway + queue worker (Redis)
//
// All application settings come from speech-judge/.env (the app reads it itself); this file only says
// *how to run* the processes. Thread budgets (SJ_WHISPER_CPU_THREADS, SJ_PHONEME_THREADS, SJ_OLLAMA_NUM_THREAD)
// are set in that .env file too.
const path = require("path");

const fs = require("fs");

const APP_DIR = path.resolve(__dirname, "..", "..");            // .../speech-judge
const VENV_BIN = path.join(APP_DIR, ".venv", "bin");

// SJ_GPU_PRESENT (shell variable first, then speech-judge/.env) picks the worker's default concurrency:
// GPU = 2 sections at once on the shared, batched models; CPU = 3 job slots behind the per-model locks.
function readSetting(name) {
  if (process.env[name] !== undefined) return process.env[name];
  try {
    const text = fs.readFileSync(path.join(APP_DIR, ".env"), "utf8");
    const m = text.match(new RegExp("^\\s*" + name + "\\s*=\\s*([^#\\r\\n]*)", "m"));
    if (m) return m[1].trim().replace(/^['"]|['"]$/g, "");
  } catch (e) { /* no .env: use defaults */ }
  return undefined;
}
const GPU_PRESENT = ["1", "true", "yes", "on"].includes(String(readSetting("SJ_GPU_PRESENT") || "").toLowerCase());
const WORKER_CONCURRENCY = process.env.JUDGE_WORKER_CONCURRENCY || (GPU_PRESENT ? 2 : 3);

const common = {
  cwd: APP_DIR,
  interpreter: "none",                  // run the venv binaries directly, not through node
  autorestart: true,
  min_uptime: "60s",                    // model loading takes a while; a crash before 60 s counts as a failed start
  exp_backoff_restart_delay: 2000,      // 2 s, growing: gives Ollama / Redis time to come up after a boot
  max_restarts: 20,
  kill_timeout: 30000,                  // let an in-flight request finish on reload/stop
  max_memory_restart: "16G",            // safety net against a leak; 32 GiB machine, Ollama is separate
  time: true,                           // timestamp every log line
  merge_logs: true,
  env: {
    PYTHONUNBUFFERED: "1",
    HF_HOME: process.env.HF_HOME || "/opt/speechjudge/hf-cache",   // model cache on the EBS volume
    TOKENIZERS_PARALLELISM: "false",
  },
};

module.exports = {
  apps: [
    {
      ...common,
      name: "judge-api",
      script: path.join(VENV_BIN, "python"),
      args: "-m app",                   // uvicorn on SJ_HOST:SJ_PORT (default 127.0.0.1:8000)
    },
    {
      ...common,
      name: "judge-worker",
      script: path.join(VENV_BIN, "celery"),
      // threads pool: ONE copy of the models shared by the job slots. The per-model locks keep "one Whisper,
      // one pronunciation, one Qwen call at a time", while different jobs can be in different stages.
      // Concurrency: 3 on CPU, 2 on GPU (set SJ_GPU_PRESENT in .env); override with JUDGE_WORKER_CONCURRENCY.
      args: `-A app.workers.celery_app worker --pool=threads --concurrency=${WORKER_CONCURRENCY} --prefetch-multiplier=1 --loglevel=INFO`,
    },
  ],
};
