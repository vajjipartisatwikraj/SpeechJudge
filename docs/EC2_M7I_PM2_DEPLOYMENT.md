# Deploying Speech Judge on AWS EC2 `m7i.2xlarge` (CPU only) with PM2: end to end

A second, independent deployment guide. The first one (`AWS_DEPLOYMENT.md`) targets a GPU server with Docker.
This one targets a **CPU-only EC2 `m7i.2xlarge` (8 vCPU, 32 GiB)**, runs the backend **directly on Ubuntu under
PM2** (no Docker), switches the server on only for assessments, and automates releases and start/stop with
**GitHub Actions**. The frontend stays on your laptop and talks to the server through an SSH tunnel.

> **Honesty note.** The backend, the tests and all speed numbers below come from your Windows laptop (CPU).
> Nothing here has been run on EC2 itself: the commands are standard, but expect to debug the first boot.
> Speeds on the server are **estimates**; `tests/load/` lets you measure the real ones (section 11).

---

## 1. Decisions in one page

| Topic | Choice | Why |
|---|---|---|
| Instance | **`m7i.2xlarge`**: 8 vCPU, 32 GiB RAM, Intel Sapphire Rapids, about **$0.40/h** (us-east-1) | 32 GiB is far more than the ~9 GiB the models need, so RAM is never the limit; CPU is. |
| OS | **Ubuntu Server 24.04 LTS** (ships Python 3.12, which the project targets) | No Docker, no GPU drivers to manage. |
| Process manager | **PM2** (with `pm2 startup` so it comes back after every stop/start of the instance) | What you asked for; restarts crashes, keeps logs, zero-downtime reload. |
| LLM runtime | **Ollama** as a systemd service, model `qwen3:4b` (4-bit) | Already proven on your laptop; the API talks to `127.0.0.1:11434`. |
| Queue | **Redis** on the same box (Layout B) | Durable jobs; API stays responsive while the CPUs are busy. |
| Releases | GitHub Actions → AWS SSM → `deploy.sh` (`git pull`, `pm2 reload`) | No open ports, no SSH keys in GitHub, no Docker registry. |
| Access | API bound to `127.0.0.1`, reached through an **SSH tunnel** | Nothing exposed to the internet. |

### 1.1 Two PM2 layouts, one `.env`

| | **Layout A: single process** | **Layout B: queue (recommended for exams)** |
|---|---|---|
| PM2 processes | `judge-api` | `judge-api` (gateway) + `judge-worker` |
| Redis | not needed | needed |
| Models live in | the API process | the worker process |
| Endpoints | sync `/evaluate` **and** async | **async only** (`/evaluate` returns 503 by design) |
| Jobs survive a restart | no | yes (24 h in Redis) |
| API stays responsive under load | shares the CPU with inference | yes, the gateway only enqueues |
| Use it for | first bring-up, your test console (its default is sync) | the real assessment, the Main Application |

You switch between them by changing two lines in `.env` and starting different PM2 processes (section 8.9).

### 1.2 Thread budget (your proposal and the version I recommend)

Your table (Whisper 3, Pronunciation 3, Qwen 2 threads = 8) is the right idea: give every model its own share
so they don't fight over the same cores. I added the knobs that make it real and benchmarked how each model
scales with threads (laptop, `tests/load/bench_threads.py`):

| Model | 2 threads | 3 threads | 4 threads | 6 threads |
|---|---|---|---|---|
| Whisper `small.en`, s per 10 s clip | 9.6 | 8.5 | 8.0 | – |
| Pronunciation model, s per 10 s clip | 5.2 | 4.9 | 4.4 | – |
| Qwen3-4B, one Q&A answer, s | **72.8** | – | 57.1 | 50.2 |
| Qwen3-4B, **compact scores-only output**, s | – | – | **29.4** | – |

Whisper and the pronunciation model gain only ~20 % from 2 to 4 threads; Qwen gains the most from threads.
With a 24-question Q&A exam, Qwen is about 80 % of all the work, so I recommend giving it the larger share:

| Process | Your proposal | **Recommended** | `.env` setting | Max concurrent |
|---|---|---|---|---|
| Whisper | 3 | **2** | `SJ_WHISPER_CPU_THREADS` | 1 |
| Pronunciation model | 3 | **2** | `SJ_PHONEME_THREADS` | 1 |
| Qwen (Ollama) | 2 | **4** | `SJ_OLLAMA_NUM_THREAD` | 1 |
| Acoustic analysis | shared, light | shared, light (about 20 ms per clip) | – | concurrent |
| FastAPI, Redis, PM2 | shared | shared | – | – |
| **Total** | 8 | **8** | | |

"Max concurrent 1" per model is already how the code works: each model sits behind a lock, so exactly one
Whisper call, one pronunciation call and one Qwen call run at a time. In the worker, three job slots
(`--concurrency=3`) let different jobs be in different stages at once. The `.env` default is the recommended split; to
use your 3/3/2 split change the three numbers.

### 1.3 The biggest speed-up is not a thread setting

`SJ_LLM_JUSTIFICATIONS=false` tells Qwen to return bare scores instead of a sentence of comment per dimension:
**29 s instead of 57 s** per answer in the benchmark (about 2x), with the same relevance score on that sample. The cost is that the
test console shows empty "justification" cells. I have not compared score quality with and without comments on a
larger set, so try it on a few of your recordings before relying on it. It is off by default.

---

## 2. Server specification

| Item | Value |
|---|---|
| Instance type | `m7i.2xlarge` (8 vCPU = 4 physical cores with hyper-threading; 32 GiB; up to 12.5 Gbps network) |
| Root volume | **100 GiB gp3**, encrypted (OS 10 GB + Python env ~8 GB + models ~6 GB + logs; roomy) |
| OS image | Ubuntu Server 24.04 LTS, x86_64 (AMI from the launch wizard's Quick Start) |
| Public IP | Auto-assigned. It changes on every start, which is fine with SSM + tunnel; add an Elastic IP only if you want a fixed address |
| Memory in use (estimate) | Qwen in Ollama ~3.5 GB, Whisper ~1 GB, pronunciation model ~2 GB, Python/API/Redis ~1.5 GB, OS ~1 GB: **~9 GiB** |

Alternatives worth knowing:

| Alternative | When |
|---|---|
| `c7i.2xlarge` (8 vCPU, 16 GiB, a bit cheaper) | If `free -h` after start shows the box stays under ~12 GiB. Same CPU speed, half the RAM. |
| `m7i.4xlarge` (16 vCPU) | Not recommended first: Qwen gains little from more threads. Several 2xlarge servers scale better. |
| **Several `m7i.2xlarge` in parallel** | Each one is an independent copy of this guide handling a share of the students. Throughput scales linearly; cost per student stays the same. |
| GPU (`g5.2xlarge`, see `AWS_DEPLOYMENT.md`) | For cohorts of ~50+ students with many open answers: roughly 5x faster per server at ~3x the hourly price. |

Because the server is switched off between sessions you pay compute only while it runs; the stopped server costs
about **$8 per month** for the 100 GiB disk. Prices are us-east-1 list prices; check your region.

---

## 3. Capacity (estimates)

Assumptions: speeds equal your laptop's (measured), recommended thread split, exam pattern of **8 Reading, 16
Repeat, 10 Jumbled (voice), 24 Q&A, 2 Storytelling = 60 items per student**, one instance.

| Work per student | Full Qwen output | Compact (`SJ_LLM_JUSTIFICATIONS=false`) |
|---|---|---|
| 34 short items (reading/repeat/jumbled) on Whisper + pronunciation threads | ~2.5 min | ~2.5 min |
| 26 open answers on Whisper threads | ~4.5 min (runs in parallel with Qwen) | same |
| 26 Qwen calls (24 Q&A ~57 s, 2 stories ~70 s) | **~25 min** | **~13 min** |
| **Wall time per student (Qwen is the bottleneck)** | **~25 min** | **~13 min** |
| Students per hour per server | **~2.4** | **~4.7** |
| Compute cost per student (at $0.40/h) | ~$0.17 | ~$0.09 |

What that means:

- One server finishes about **10 to 20 students in a 4 to 8 hour window** (compact to full). That is a pilot or a
  small class, not 100 students.
- About **10 servers** would finish 100 students in roughly 2 hours (compact) to 4 hours (full output); one GPU server
  needs about 6 hours for the same 100 (see `AWS_DEPLOYMENT.md`). Total cost is similar, the number of machines differs.
- Students answering in real time: a server keeps up with about **2 to 3 students working at once** (each sending
  about 1 item per minute); more than that queue, and in Layout B results simply arrive later.
- These numbers assume Ollama scales like on your laptop. Measure (section 11) before planning an exam.

---

## 4. Prerequisites

### 4.1 Accounts and tools

- AWS account with billing, and a user who can create IAM roles, EC2 instances and SSM commands.
- A GitHub repository containing this project with `speech-judge/` and `frontend/` in it (repo root =
  `CommunicationJudge/`). If your repo root is `speech-judge/`, drop the `speech-judge/` prefix in the paths below.
- On your laptop: PowerShell, the built-in **OpenSSH client**, and **AWS CLI v2** (`aws --version`).
- Quota: `m7i` is a *Standard* instance family; the default **Running On-Demand Standard instances** quota is normally
  enough for 8 vCPUs. If the launch fails with `VcpuLimitExceeded`, request a quota increase in **Service Quotas**.

### 4.2 Generate the API credential once

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep it in a password manager. It goes only into the server's `.env` and (temporarily) into your laptop's terminal
when you connect the frontend. The service refuses to start in production mode if it is shorter than 24 characters.

### 4.3 Repository checklist

- `.env` is git-ignored: never commit it.
- Make sure these files are in the repo (they are part of this project): `speech-judge/deploy/pm2/ecosystem.config.js`,
  `speech-judge/deploy/scripts/deploy.sh`, `speech-judge/deploy/scripts/wait_ready.sh`,
  `speech-judge/deploy/env/m7i-cpu.env.example`.

---

## 5. One-time AWS setup

### 5.1 Key pair

EC2 → **Key Pairs → Create key pair** → `speechjudge-key`, RSA, `.pem`. Save as `C:\keys\speechjudge.pem`, then
restrict it (Windows insists):

```powershell
icacls "C:\keys\speechjudge.pem" /inheritance:r
icacls "C:\keys\speechjudge.pem" /grant:r "$($env:USERNAME):R"
```

### 5.2 Security group and ports

EC2 → **Security Groups → Create**: `speechjudge-sg`, default VPC.

| Direction | Port | Source | Purpose |
|---|---|---|---|
| Inbound | **22/tcp** | **your public IP /32** (https://checkip.amazonaws.com) | SSH and the SSH tunnel. Optional if you use SSM only. |
| Inbound | nothing else | | The API listens on `127.0.0.1:8000` only; Ollama on `127.0.0.1:11434`; Redis on `127.0.0.1:6379`. |
| Outbound | all | 0.0.0.0/0 | package installs, model downloads, GitHub, SSM |

Your home IP changes: update the SSH rule when it does.

### 5.3 IAM role for the server

IAM → **Roles → Create role** → *AWS service → EC2* → attach **`AmazonSSMManagedInstanceCore`** → name
`speechjudge-ec2-role`. (SSM lets GitHub run deploy commands and lets you open a shell with no SSH. No ECR
permission is needed in this guide.)

### 5.3b IAM role for GitHub Actions (OIDC, no stored keys)

1. IAM → **Identity providers → Add provider** → *OpenID Connect*, URL `https://token.actions.githubusercontent.com`,
   audience `sts.amazonaws.com`.
2. IAM → **Roles → Create role** → *Web identity* → that provider → name `speechjudge-github`. Edit its **trust policy**
   so only your repo's `main` branch can use it:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike":   { "token.actions.githubusercontent.com:sub": "repo:<OWNER>/<REPO>:ref:refs/heads/main" }
    }
  }]
}
```

3. Attach this inline policy (fill in region, account and the instance ID from section 6):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["ec2:StartInstances","ec2:StopInstances"],
      "Resource": "arn:aws:ec2:<REGION>:<ACCOUNT_ID>:instance/<INSTANCE_ID>" },
    { "Effect": "Allow", "Action": ["ec2:DescribeInstances","ec2:DescribeInstanceStatus"], "Resource": "*" },
    { "Effect": "Allow", "Action": "ssm:SendCommand",
      "Resource": ["arn:aws:ssm:<REGION>::document/AWS-RunShellScript",
                   "arn:aws:ec2:<REGION>:<ACCOUNT_ID>:instance/<INSTANCE_ID>"] },
    { "Effect": "Allow",
      "Action": ["ssm:GetCommandInvocation","ssm:ListCommandInvocations","ssm:DescribeInstanceInformation"],
      "Resource": "*" }
  ]
}
```

Note the role's ARN for section 9.

---

## 6. Launch the instance

EC2 → **Launch instance**:

| Field | Value |
|---|---|
| Name | `speechjudge-cpu` |
| AMI | **Ubuntu Server 24.04 LTS (HVM), SSD**, 64-bit (x86) |
| Instance type | **`m7i.2xlarge`** |
| Key pair | `speechjudge-key` |
| Security group | existing `speechjudge-sg` |
| Storage | 1 × **100 GiB gp3**, Encrypted |
| Advanced → IAM instance profile | `speechjudge-ec2-role` |
| Advanced → Metadata version | **V2 only** |
| Advanced → Shutdown behavior | **Stop** (default) |

Launch, wait for "2/2 checks passed", and note the **Instance ID** and **Public IPv4 address**. Put the instance ID
into the GitHub role policy (section 5.3b).

---

## 7. Connect to Ubuntu

**SSH** (PowerShell):

```powershell
ssh -i "C:\keys\speechjudge.pem" ubuntu@<PUBLIC_IP>
```

**Or SSM** (no open port; needs the instance role to have been attached):

```powershell
aws ssm start-session --target <INSTANCE_ID> --region <REGION>
```

You are in when the prompt reads `ubuntu@ip-...:~$`. Everything in section 8 runs there as the `ubuntu` user.

---

## 8. Prepare the server (one time)

### 8.1 System packages

```bash
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git curl jq ffmpeg libsndfile1 build-essential python3-venv python3-pip redis-server
python3 --version            # expect 3.12.x
```

Optional safety net against the out-of-memory killer during model loading (a 4 GiB swap file):

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 8.2 Node.js and PM2

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2
pm2 --version
```

### 8.3 Redis (needed only for Layout B, harmless otherwise)

It is used for temporary job state, so turn persistence off and keep it on localhost (the default):

```bash
echo 'save ""' | sudo tee -a /etc/redis/redis.conf
sudo sed -i 's/^appendonly yes/appendonly no/' /etc/redis/redis.conf
sudo systemctl enable --now redis-server && sudo systemctl restart redis-server
redis-cli ping               # PONG
```

### 8.4 Ollama (runs Qwen)

```bash
curl -fsSL https://ollama.com/install.sh | sh          # installs and enables the 'ollama' systemd service

# one model at a time, kept in memory; threads are set per request by the backend (SJ_OLLAMA_NUM_THREAD)
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=1"\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\nEnvironment="OLLAMA_KEEP_ALIVE=-1"\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama

ollama pull qwen3:4b                                   # about 2.5 GB
curl -s http://127.0.0.1:11434/api/version
```

### 8.5 Application folder and code

The server pulls code from GitHub with a **read-only deploy key**:

```bash
sudo mkdir -p /opt/speechjudge/hf-cache && sudo chown -R ubuntu:ubuntu /opt/speechjudge

ssh-keygen -t ed25519 -f ~/.ssh/speechjudge_deploy -N "" -C "speechjudge-ec2"
cat ~/.ssh/speechjudge_deploy.pub
```

Copy that public key into GitHub → your repo → **Settings → Deploy keys → Add deploy key** (leave *Allow write
access* **off**). Then:

```bash
printf 'Host github.com\n  IdentityFile ~/.ssh/speechjudge_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config
ssh -T git@github.com                                   # type 'yes'; expect "Hi <repo>! You've successfully authenticated"

git clone git@github.com:<OWNER>/<REPO>.git /opt/speechjudge/repo
cd /opt/speechjudge/repo/speech-judge
```

### 8.6 Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel

# CPU-only PyTorch (skips ~3 GB of CUDA libraries). If this exact version is not on that index, use the
# newest CPU wheel there or skip this line and let the next command install the default build.
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

.venv/bin/pip install -r requirements.txt -r requirements-ml.txt
.venv/bin/python -c "import torch, faster_whisper, transformers; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
```

### 8.7 Environment file

```bash
cp deploy/env/m7i-cpu.env.example .env
nano .env                       # set SJ_SERVICE_CREDENTIALS (section 4.2); choose Layout A or B (section 8.9)
chmod 600 .env
```

The file is documented inline. The important lines for this server:

```ini
SJ_ENVIRONMENT=production         # enforces: strong credential, mock off, docs off, both model switches on
SJ_GPU_PRESENT=false              # CPU profile: device cpu, Whisper small.en int8, no batching, 3600 s section timeout
SJ_WHISPER_CPU_THREADS=2          # thread budget: 2 + 2 + 4 = 8 vCPUs
SJ_PHONEME_THREADS=2
SJ_OLLAMA_NUM_THREAD=4
SJ_LLM_JUSTIFICATIONS=true        # false = about 2x faster open answers, no per-dimension comments
SJ_HOST=127.0.0.1                 # reached only through the SSH tunnel
```

### 8.8 Pre-download the models (so the first start is quick)

```bash
export HF_HOME=/opt/speechjudge/hf-cache
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8'); print('whisper ok')"
.venv/bin/python -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; n='facebook/wav2vec2-xlsr-53-espeak-cv-ft'; Wav2Vec2Processor.from_pretrained(n); Wav2Vec2ForCTC.from_pretrained(n); print('phoneme model ok')"
du -sh /opt/speechjudge/hf-cache                        # about 1.7 GB
```

### 8.9 Start under PM2, survive reboots

**Layout A** (one process; supports your test console's default sync mode):

```bash
# .env: SJ_JOB_BACKEND=inline  and  SJ_LOAD_MODELS_IN_API=true
pm2 start deploy/pm2/ecosystem.config.js --only judge-api
```

**Layout B** (queue; async endpoints only; recommended for real assessments):

```bash
# .env: SJ_JOB_BACKEND=celery  and  SJ_LOAD_MODELS_IN_API=false
pm2 start deploy/pm2/ecosystem.config.js --only judge-api,judge-worker
```

Then make PM2 start automatically each time the instance boots (after Ollama and Redis are up):

```bash
pm2 save
pm2 startup systemd -u ubuntu --hp /home/ubuntu
# PM2 prints one line beginning with "sudo env PATH=..."  -> copy that exact line and run it, then:

sudo mkdir -p /etc/systemd/system/pm2-ubuntu.service.d
printf '[Unit]\nAfter=network-online.target ollama.service redis-server.service\nWants=ollama.service redis-server.service\n' \
  | sudo tee /etc/systemd/system/pm2-ubuntu.service.d/order.conf
sudo systemctl daemon-reload
pm2 save

pm2 install pm2-logrotate && pm2 set pm2-logrotate:max_size 50M && pm2 set pm2-logrotate:retain 7
```

Switching layouts later: edit the two `.env` lines, then `pm2 delete all && pm2 start deploy/pm2/ecosystem.config.js --only <apps> && pm2 save`.

### 8.10 Check it

```bash
pm2 ls                                                  # judge-api (and judge-worker) "online"
pm2 logs judge-api --lines 50                           # "Loading service ... All shared services loaded"
bash deploy/scripts/wait_ready.sh                       # prints "ready" (first start: 1 to 4 minutes)
curl -s http://127.0.0.1:8000/api/v1/ready              # {"status":"ready","whisper":true,"gopt":true,"acoustic":true,"qwen":true,"mocked":[]}
free -h                                                 # roughly 9 GiB used
```

If `"mocked"` is not `[]`, a switch in `.env` is off (production mode would have refused to start).

---

## 9. GitHub Actions automation

### 9.1 Repository settings

GitHub → repo → **Settings → Secrets and variables → Actions**

| Kind | Name | Value |
|---|---|---|
| Variable | `AWS_REGION` | your region |
| Variable | `EC2_INSTANCE_ID` | `i-0123456789abcdef0` |
| Secret | `AWS_ROLE_ARN` | ARN of `speechjudge-github` (section 5.3b) |

### 9.2 `ci.yml`: run the tests on every push and pull request

`.github/workflows/ci.yml`

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: speech-judge
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: speech-judge/requirements-dev.txt
      - run: pip install -r requirements-dev.txt praat-parselmouth==0.4.7 pyloudnorm==0.2.0
      - run: python -m pytest -q
```

### 9.3 `deploy.yml`: release the newest code to the running server

`.github/workflows/deploy.yml` (manual, so a push never spends money or restarts a live exam)

```yaml
name: Deploy to EC2
on:
  workflow_dispatch:
    inputs:
      layout:
        description: "A = judge-api only, B = judge-api + judge-worker"
        type: choice
        options: [A, B]
        default: B
      branch:
        description: Branch to deploy
        default: main

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    env:
      AWS_REGION: ${{ vars.AWS_REGION }}
      INSTANCE_ID: ${{ vars.EC2_INSTANCE_ID }}
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}

      - name: Run deploy.sh on the server through SSM
        run: |
          APPS="judge-api"; [ "${{ inputs.layout }}" = "B" ] && APPS="judge-api,judge-worker"
          CMD="sudo -iu ubuntu env PM2_APPS=$APPS bash /opt/speechjudge/repo/speech-judge/deploy/scripts/deploy.sh ${{ inputs.branch }}"
          PARAMS=$(jq -nc --arg c "$CMD" '{commands:[$c],executionTimeout:["3000"]}')
          CID=$(aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
                --parameters "$PARAMS" --query Command.CommandId --output text)
          for i in $(seq 1 600); do
            S=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" --query Status --output text 2>/dev/null || echo Pending)
            case "$S" in
              Success) break ;;
              Failed|Cancelled|TimedOut) echo "deploy $S"; break ;;
            esac
            sleep 5
          done
          aws ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" \
            --query '{Status:Status,Out:StandardOutputContent,Err:StandardErrorContent}' --output yaml
          [ "$S" = "Success" ]
```

What `deploy.sh` does on the server: `git fetch` and `git reset --hard origin/<branch>` (your `.env` and `.venv`
are untracked so they stay), reinstalls Python packages only when a `requirements*.txt` changed,
`pm2 startOrReload` the chosen processes (zero-downtime reload), `pm2 save`, then waits until `/ready` succeeds.
The server must be **running** for this workflow; if it is stopped, its next start already runs the last deployed
code, and you deploy after starting it.

### 9.4 `server-control.yml`: start, stop and check the server from GitHub

`.github/workflows/server-control.yml` (manual)

```yaml
name: Server control
on:
  workflow_dispatch:
    inputs:
      action:
        description: What to do with the EC2 server
        type: choice
        options: [status, start, stop]
        default: status

permissions:
  id-token: write
  contents: read

jobs:
  control:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      AWS_REGION: ${{ vars.AWS_REGION }}
      INSTANCE_ID: ${{ vars.EC2_INSTANCE_ID }}
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}

      - name: Status
        if: inputs.action == 'status'
        run: |
          aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress,InstanceType]' --output text

      - name: Start and wait until the API is ready
        if: inputs.action == 'start'
        run: |
          aws ec2 start-instances --instance-ids "$INSTANCE_ID"
          aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
          PARAMS=$(jq -nc '{commands:["sudo -iu ubuntu bash /opt/speechjudge/repo/speech-judge/deploy/scripts/wait_ready.sh"]}')
          for i in $(seq 1 20); do          # the SSM agent needs a moment after boot
            CID=$(aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
                  --parameters "$PARAMS" --query Command.CommandId --output text 2>/dev/null) && break
            sleep 15
          done
          [ -n "$CID" ] || { echo "SSM not reachable"; exit 1; }
          for i in $(seq 1 400); do
            S=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$INSTANCE_ID" --query Status --output text 2>/dev/null || echo Pending)
            case "$S" in Success) break ;; Failed|Cancelled|TimedOut) echo "wait_ready $S"; exit 1 ;; esac
            sleep 5
          done
          echo "Ready. Public IP:"
          aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

      - name: Stop
        if: inputs.action == 'stop'
        run: |
          aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
          aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
          echo "Stopped: compute billing has ended; the EBS disk is still billed."
```

The SSM commands run as root, hence `sudo -iu ubuntu` to use the `ubuntu` user's PM2 and Node setup.

---

## 10. First start and smoke tests

After section 8.10 shows `ready`, run on the server (replace the key):

```bash
export SJ_KEY='<your server credential>'

# 1. text only, no models (Layout A answers it; Layout B returns 503 because the synchronous endpoint is off by design)
curl -s -X POST http://127.0.0.1:8000/api/v1/evaluate -H "Authorization: Bearer $SJ_KEY" -H "Content-Type: application/json" \
  -d '{"request_id":"smoke-1","question_type":"JUMBLED","expected_text":"the cat sat on the mat","question_config":{"submitted_text":"the cat sat on the mat"}}' | jq '.status,.score'

# 2. missing credential must be rejected: expect 401
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/api/v1/evaluate -H "Content-Type: application/json" -d '{}'

# 3. asynchronous call with real audio (works in both layouts). Copy a 16 kHz WAV first:
#    scp -i C:\keys\speechjudge.pem speech-judge\tests\fixtures\audio\ravi_correct.wav ubuntu@<IP>:~/sample.wav
JOB=$(jq -n --arg a "$(base64 -w0 ~/sample.wav)" \
  '{request_id:"smoke-2",question_type:"REPEAT",audio:$a,expected_text:"The doctor suggested that the apple is good for your health."}' \
  | curl -s -X POST http://127.0.0.1:8000/api/v1/evaluate/async -H "Authorization: Bearer $SJ_KEY" -H "Content-Type: application/json" -d @- | jq -r .job_id)
sleep 15; curl -s http://127.0.0.1:8000/api/v1/jobs/$JOB -H "Authorization: Bearer $SJ_KEY" | jq '.status,.result.score,.result.flags'
```

Expected: step 1 gives `"COMPLETED"` and `1` (Layout A) or a 503 saying synchronous evaluation is not served (Layout B);
step 2 gives `401`; step 3 shows `COMPLETED`, a score near 0.9 and an empty flag list. A `MOCKED_*` flag means a switch is off.

---

## 11. Measure the real speed (recommended before any exam)

The load scripts are in the repo. On the server:

```bash
cd /opt/speechjudge/repo/speech-judge
# the generated voice clips are git-ignored; copy them from your laptop once:
#   scp -r -i C:\keys\speechjudge.pem speech-judge\tests\fixtures\audio ubuntu@<IP>:/opt/speechjudge/repo/speech-judge/tests/fixtures/
.venv/bin/pip install psutil

.venv/bin/python tests/load/bench_threads.py          # thread scaling on THIS CPU (stop the PM2 apps first: pm2 stop all)
pm2 start all
.venv/bin/python tests/load/concurrency_load.py --type repeat --max 6     # Layout A (sync endpoint)
.venv/bin/python tests/load/concurrency_load.py --type exam --max 6
```

Use the `bench_threads.py` output to pick the final `SJ_WHISPER_CPU_THREADS`, `SJ_PHONEME_THREADS` and
`SJ_OLLAMA_NUM_THREAD`, and the per-item times to recompute the capacity table in section 3. (`concurrency_load.py`
uses the synchronous endpoint, so run it in Layout A; in Layout B time a batch of async jobs instead.)

---

## 12. Connect your local frontend to the deployed backend

The frontend proxy (`frontend/server.py`) sends `/api/v1/*` to `BACKEND_URL` with the credential from
`FRONTEND_API_KEY`; both are read from environment variables first, so you don't edit files.

1. Server is running and ready (section 13.1).
2. Stop your **local** backend (or use a different local port, as below).
3. Terminal 1, keep it open: an SSH tunnel from your port 8001 to the server's 8000.

```powershell
ssh -i "C:\keys\speechjudge.pem" -N -L 8001:127.0.0.1:8000 ubuntu@<PUBLIC_IP>
```

   (SSM alternative without SSH: `aws ssm start-session --target <INSTANCE_ID> --document-name AWS-StartPortForwardingSession --parameters "portNumber=8000,localPortNumber=8001"`, needs the Session Manager plugin.)

4. Terminal 2, from the repo root:

```powershell
$env:BACKEND_URL      = "http://127.0.0.1:8001"
$env:FRONTEND_API_KEY = "<the SERVER credential, not your local one>"
python frontend\server.py
```

5. Open http://127.0.0.1:8080. The header badge should say **Backend ready**.

**Layout B note:** the gateway does not serve the synchronous endpoint. In the console, set **Mode: Asynchronous** on
every test tab (Q&A and Storytelling already default to it); otherwise Reading/Repeat/Jumbled return
"This node does not serve synchronous evaluation". Layout A has no such limit.

Test checklist: badge is ready; **Edge cases → Run all** shows every row "as expected"; a Reading recording returns
a transcript, pronunciation panel and acoustic metrics; a Q&A answer returns after roughly 30 to 90 seconds (CPU);
**History** lists the runs. Then close the tunnel (Ctrl+C) and stop the server (section 13.2).

---

## 13. Daily operation

### 13.1 Start before a session

GitHub → **Actions → Server control → Run workflow → start**, or:

```powershell
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region <REGION>
aws ec2 wait instance-status-ok --instance-ids <INSTANCE_ID> --region <REGION>
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --region <REGION> --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

PM2, Ollama and Redis come back by themselves (systemd). Allow about 2 to 5 minutes until `/ready` says ready. Start
the server before students arrive.

### 13.2 Stop after a session

Wait until the queue is empty (`pm2 logs judge-worker --lines 30` shows no activity; Layout A: no requests), collect
any results you still need (Layout A results live only in memory), then stop the instance (*Server control → stop* or
`aws ec2 stop-instances ...`). **Stop, never terminate**: terminating deletes the disk and the downloaded models.

### 13.3 PM2 cheat sheet

```bash
pm2 ls                              # status, restarts, memory
pm2 logs judge-api --lines 100      # add judge-worker for Layout B
pm2 monit                           # live CPU/memory
pm2 reload judge-api                # zero-downtime restart (after editing .env use: pm2 reload judge-api --update-env)
pm2 restart all --update-env
pm2 stop all                        # e.g. before running benchmarks
sudo journalctl -u ollama -n 50     # Qwen runtime
sudo journalctl -u redis-server -n 50
```

### 13.4 Safety nets

- **Scheduled stop:** EventBridge → **Scheduler** → create a schedule with target *EC2 StopInstances* for your instance
  a few hours after each session, so a forgotten server cannot run all week.
- **Budget alert:** Billing → **Budgets** → cost budget (for example $40 per month), alerts at 50 % and 100 %.
- **Deploying code:** merge to `main` (CI runs), then *Actions → Deploy to EC2*. Roll back by running the deploy with
  an older branch/tag name as the `branch` input.

---

## 14. Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| `judge-api` keeps restarting (`pm2 ls`) | `pm2 logs judge-api`. Usually Ollama or Redis not up yet (PM2 retries with growing delays), or production mode rejected the config (message names the setting). |
| `Unsafe production configuration: ...` | Fix the named line in `.env` (credential under 24 characters, `SJ_MOCK_MODE`, `SJ_ENABLE_DOCS`, a model switch off). |
| `/ready` stays 503 | Models still loading, or Ollama cannot serve `qwen3:4b`: `ollama list`, `sudo journalctl -u ollama -n 50`. |
| Layout B: `/ready` 503 but worker is online | The gateway reads worker heartbeats from Redis: `redis-cli keys 'speechjudge:worker:*'`; check `SJ_REDIS_URL` is the same for both processes. |
| Process killed, `Killed` in logs | Out of memory: `dmesg | grep -i kill`; check `free -h`; use the swap file (8.1) or a bigger instance. |
| Everything is slow | Thread oversubscription: `top -H`; keep the 2 + 2 + 4 split; do not run benchmarks while serving. |
| Open answers take minutes | Expected on CPU (about 1 minute each); use `SJ_LLM_JUSTIFICATIONS=false` or more servers. |
| HTTP 429 from `/evaluate` | More than `SJ_MAX_CONCURRENT_EVALUATIONS` sync calls in flight; retry after the `Retry-After` seconds or use the async endpoint. |
| Frontend `502 Backend unreachable` | Tunnel closed, wrong `BACKEND_URL`, or server not ready. |
| Frontend `401` | `FRONTEND_API_KEY` is not the server's credential. |
| Frontend 503 "does not serve synchronous evaluation" | Layout B: switch the test's Mode to Asynchronous. |
| GitHub `Not authorized to perform sts:AssumeRoleWithWebIdentity` | Trust policy `sub` does not match `repo:<OWNER>/<REPO>:ref:refs/heads/main`, or the workflow ran from another branch. |
| SSM `InvalidInstanceId` | Agent not yet registered after boot (wait a minute) or the instance role lacks `AmazonSSMManagedInstanceCore`. |
| `deploy.sh` fails on `git fetch` | Deploy key missing or not added to the repo (section 8.5): `ssh -T git@github.com`. |
| No space left | `df -h`; `pm2 flush`; `sudo journalctl --vacuum-size=200M`; `pip cache purge`. |

---

## 15. Security checklist

- [ ] Only port 22 is open, and only to your IP (or none, using SSM). The API, Ollama and Redis listen on localhost.
- [ ] `SJ_ENVIRONMENT=production`, docs disabled, credential at least 24 random characters, `.env` is `chmod 600`.
- [ ] No AWS keys or server credentials in GitHub; the OIDC role is limited to `main` and to this one instance.
- [ ] EBS encrypted, IMDSv2 required, the instance role has only SSM.
- [ ] The deploy key is read-only.
- [ ] For a real Main Application integration put HTTPS in front (Caddy or a load balancer with a certificate) rather
      than tunnelling, and open only 443 to the Main Application's address range.
- [ ] Student audio is not stored or logged; results exist only in memory (Layout A) or Redis for 24 h (Layout B).
- [ ] Budget alert and scheduled stop are configured.

## 16. Teardown

1. Stop, then **terminate** the instance (this deletes the disk and the models).
2. Delete the security group, the key pair, the two IAM roles and the OIDC provider.
3. Remove the deploy key from the GitHub repo and the three Actions settings.

---

## Appendix: limits worth knowing

- **Capacity is modest on CPU.** About 2.4 to 4.7 students per hour per server for the 60-item pattern (section 3).
  Scale by adding servers, by `SJ_LLM_JUSTIFICATIONS=false`, or by moving to the GPU guide.
- **The phoneme model and Whisper barely use extra threads**, so a bigger single CPU instance is a poor way to go faster.
- **Sapphire Rapids has AMX/AVX-512 units** that the current code does not use explicitly; a bf16/int8-optimised
  pronunciation model could be faster, but that is untested.
- **Single points of failure:** one instance, one Ollama, one Redis. For a high-stakes exam run a second server and
  split the students between them.
