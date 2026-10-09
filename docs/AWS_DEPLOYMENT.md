# Deploying Speech Judge (backend) on AWS EC2

Audience: you, deploying **only the backend** to one EC2 GPU server that is switched on just for assessment
sessions, with **GitHub Actions** automating build and server control, and your **local frontend** (the test
console in `frontend/`) talking to the deployed backend.

> **Honesty note.** The backend has been run and tested on a Windows CPU laptop (unit, integration and
> end-to-end tests). It has **not** been run on a GPU, in this Docker image, or on AWS. Section 13 lists the
> specific things most likely to need a fix on the first GPU boot. Prices and timings below are planning
> estimates; confirm prices in the AWS Pricing Calculator and measure speed on your own server (section 4).

---

## 1. What you will end up with

```
 GitHub (your repo)                       AWS (one region, e.g. ap-south-1)
 ┌───────────────┐  push to main   ┌────────────────┐
 │ GitHub Actions│ ──build/push──▶ │ ECR (Docker    │
 │  ci.yml       │                 │  image store)  │
 │  build-push   │                 └───────┬────────┘
 │  server-ctrl  │ ──start/stop────▶ EC2   │ pull :latest at every boot
 └───────────────┘   (OIDC role, SSM)  ┌───▼─────────────────────────────┐
                                       │ EC2 GPU (Ubuntu, Docker, NVIDIA)│
   Your laptop                         │  systemd → docker run speechjudge│
 ┌──────────────────┐   SSH tunnel     │  API :8000 (models loaded once)  │
 │ browser :8080    │ ───────────────▶ │  model cache on EBS (/opt/...)   │
 │ frontend/server.py│  (encrypted)    └──────────────────────────────────┘
 └──────────────────┘
```

Design choices (and why):

| Choice | Reason |
|---|---|
| **One EC2 GPU instance, one Docker container** ("single-node mode": API loads the models, jobs run in-process) | Simplest thing that works with your frontend, which uses both the synchronous and asynchronous endpoints. No Redis, no second GPU copy of the models. |
| Stop the instance when not in use | You pay for compute only while it is running. The EBS disk keeps the Docker image and the ~12 GB of downloaded models, so later starts are fast. |
| Image in **ECR**, built by GitHub Actions | Building a multi-GB PyTorch image on the server at every deploy is slow and error-prone. |
| GitHub → AWS via **OIDC role**, server control via **SSM** | No AWS keys stored in GitHub, no SSH key stored in GitHub, no inbound port needed for deploys. |
| Reach the API through an **SSH tunnel** for testing | The API is never exposed to the internet; traffic (student audio, token) is encrypted. |

The repo also contains a `docker-compose.yml` for a **queue mode** (Redis + API gateway + separate GPU
worker). Use it later if you need several GPUs. Section 13 explains its one catch with your frontend.

---

## 2. Server requirements

### 2.1 Workload (what has to fit on the machine)

| Component | Runs on | Approx. memory (estimate) |
|---|---|---|
| Whisper `large-v3`, fp16 | GPU | ~3–4 GB VRAM |
| Qwen3-4B (`transformers`, bf16) | GPU | ~8–10 GB VRAM incl. working memory |
| wav2vec2 phoneme model (1.2 GB download) | **CPU** (the current code does not move it to the GPU) | ~1.5–2 GB RAM |
| Acoustic engine, scoring, FastAPI | CPU | small |
| Peak RAM while models load | CPU RAM | ~10–12 GB |

So: about **13–15 GB VRAM** and comfortably **≥ 24 GB system RAM** during start-up.

### 2.2 Instance choice

| Instance | vCPU / RAM / GPU | Verdict |
|---|---|---|
| **`g5.2xlarge`** | 8 / 32 GiB / 1× A10G (24 GB) | **Recommended.** Matches the architecture doc (8 vCPU, 32 GB, 24 GB GPU). |
| `g5.xlarge` | 4 / 16 GiB / 1× A10G (24 GB) | Minimum. 16 GiB RAM is tight while models load; fine for a pilot. On-demand ≈ $1.0/h in us-east-1. |
| `g6.xlarge` / `g6.2xlarge` | 4 or 8 vCPU / 16 or 32 GiB / 1× L4 (24 GB) | Also fine (the architecture doc names L4). Choose whichever is available and cheaper in your region. |
| `g4dn.*` | T4 16 GB | Not recommended: T4 has no native bf16, so Qwen would need a code change (fp16) and is slower. |
| CPU-only (`c6i`, `m6i`) | — | Only for the dry run in section 7. Real models on CPU take ~5–7 s per reading and 40–60 s per open answer on your laptop. |

Availability of GPU families differs per region; check the EC2 console for your region. For Indian
students, `ap-south-1` (Mumbai) is the natural choice **if** your chosen instance type is offered there.

### 2.3 Storage, OS, network

| Item | Value |
|---|---|
| Root volume | **150 GB gp3** (OS + Docker image ~8–10 GB + models ~12–13 GB + headroom), encrypted |
| OS / AMI | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**, provided by AWS, so the NVIDIA driver is already installed (section 8 verifies Docker and the GPU container runtime) |
| Login user | `ubuntu` |
| Instance metadata | IMDSv2 required |
| Public IP | Auto-assigned IP changes at every stop/start. Fine with SSM + tunnel (you look the IP up each time); attach an Elastic IP only if you want a fixed address |
| Ports | see section 5.3 |

### 2.4 Indicative cost (us-east-1 list prices; verify for your region)

| Item | Approx. |
|---|---|
| `g5.2xlarge` running | ≈ $1.2 per hour (`g5.xlarge` ≈ $1.0) |
| 150 GB gp3 while stopped | ≈ $12 per month |
| ECR (one ~9 GB image) | ≈ $1 per month |
| Public IPv4 address | ≈ $3.6 per month if you keep an Elastic IP |
| Example: 10 sessions × 6 h on `g5.2xlarge` | ≈ 60 h × $1.2 ≈ **$72 + ~$16 fixed** |

Set an **AWS Budget alert** (section 12.4). A forgotten running GPU instance is the main cost risk.

---

## 3. Capacity planning

Numbers here are **planning assumptions, not measurements** on this code on a GPU. Measure in section 10.3.

Assumed per-request processing time on one `g5.2xlarge` (models are shared, so requests on the same model
queue behind each other):

| Question type | Work | Assumed time |
|---|---|---|
| Jumbled (on-screen) | text only | < 0.1 s |
| Reading / Repeat / spoken Jumbled | Whisper + phoneme model (CPU) + acoustic | ~3–6 s |
| Q&A / Storytelling | Whisper + Qwen generation (~250 tokens) | ~10–25 s |

Planning formula: `server hours ≈ (students × Σ seconds per question) ÷ (3600 × effective_parallelism)`,
with `effective_parallelism ≈ 1.5` for this single-GPU setup.

Example: 100 students × (2 reading/repeat × 5 s + 2 open answers × 20 s) = 5,000 s ÷ 1.5 ≈ **55 minutes**
of processing. 500 students ≈ 4.6 hours. The design is asynchronous, so students can submit while processing
lags behind, but **the server must stay on until the backlog is empty**. For big cohorts you either run more
than one instance (queue mode, section 13) or speed up the language model (a batching inference server such
as vLLM would be a separate project).

Start-up time budget: instance boot 1–2 min + image pull (only changed layers) + model load from the EBS cache
~1–3 min → plan **5–8 minutes** from "start" to "ready". The very first start also downloads ~12–13 GB of models
(10–20 min).

---

## 4. Prerequisites

### 4.1 Accounts and tools

- An AWS account with billing enabled, and a user that can create IAM roles, EC2, ECR and SSM resources.
- A GitHub repository containing this project. In the steps below the repo root is `CommunicationJudge/`
  (with `speech-judge/` and `frontend/` inside). If your repo root is `speech-judge/` itself, remove the
  `speech-judge/` prefixes in the workflow files.
- On your Windows laptop: PowerShell, the **OpenSSH client** (built into Windows 10/11), **AWS CLI v2**
  (`aws --version`), and (only for option 11.A-SSM) the **Session Manager plugin**.
- Python 3.10+ for the local frontend (already set up).

### 4.2 GPU quota: do this first

New accounts often have **0 vCPUs** of quota for GPU instances (the default is 0 or a small number). Without
quota the instance will not launch.

1. AWS Console → **Service Quotas** → *AWS services* → *Amazon Elastic Compute Cloud (Amazon EC2)*.
2. Find **Running On-Demand G and VT instances** (use the same region you will launch in).
3. Request at least **8 vCPUs** (one `g5.2xlarge`) or 16 if you may want two instances later.
4. Approval can take from minutes to a couple of days; ask before anything else.

### 4.3 Repository hygiene

- `speech-judge/.env` is git-ignored. **Never commit it.** The server gets its own `.env` (section 8.4).
- Generate the production credential once and keep it in a password manager:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 5. One-time AWS setup

Pick one region and use it everywhere (examples use `ap-south-1`; replace `<ACCOUNT_ID>`, `<OWNER>`, `<REPO>`).

### 5.1 ECR repository (image store)

Console: **ECR → Create repository** → private → name `speech-judge` → create.
Or CLI:

```powershell
aws ecr create-repository --repository-name speech-judge --region ap-south-1 --image-scanning-configuration scanOnPush=true
```

Optional cost control, keep only the newest images: ECR → repository → *Lifecycle policy* → "Any image count more
than 5".

### 5.2 Key pair (for SSH)

EC2 → **Key Pairs → Create key pair** → name `speechjudge-key`, type RSA, format `.pem`. Save it as
`C:\keys\speechjudge.pem`. Restrict its permissions (Windows requires it):

```powershell
icacls "C:\keys\speechjudge.pem" /inheritance:r
icacls "C:\keys\speechjudge.pem" /grant:r "$($env:USERNAME):R"
```

### 5.3 Security group and ports

EC2 → **Security Groups → Create**: name `speechjudge-sg`, default VPC.

| Direction | Port | Source | Needed? |
|---|---|---|---|
| Inbound | **22/tcp** (SSH) | **your public IP /32** (check at https://checkip.amazonaws.com) | Yes for SSH and the SSH tunnel. If you use only SSM, you may leave it closed. |
| Inbound | 8000/tcp (API) | your IP /32 | **Only** for option 11.B (direct, unencrypted). Not needed for the tunnel. |
| Inbound | 80/tcp, 443/tcp | 0.0.0.0/0 (or your users' ranges) | **Only** for option 11.C (HTTPS with a domain). |
| Outbound | all | 0.0.0.0/0 | Yes (model downloads, ECR, SSM, apt). |

Never open 8000 to `0.0.0.0/0`. Your home IP changes: update the rule when it does.

### 5.4 IAM role for the EC2 instance

IAM → **Roles → Create role** → trusted entity *AWS service → EC2* → attach managed policies:

- `AmazonSSMManagedInstanceCore` (lets SSM run commands and open sessions)
- `AmazonEC2ContainerRegistryReadOnly` (lets the server pull the image)

Name it `speechjudge-ec2-role`. (An instance profile with the same name is created automatically in the console.)

### 5.5 IAM role for GitHub Actions (OIDC, no stored keys)

1. IAM → **Identity providers → Add provider** → *OpenID Connect* → provider URL
   `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
2. IAM → **Roles → Create role** → *Web identity* → provider above → audience `sts.amazonaws.com` → finish, then
   edit the **trust policy** so only your repo's `main` branch can assume it:

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

3. Attach an inline permissions policy named `speechjudge-github-deploy` (fill in region, account and instance ID
   after section 6):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    { "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability","ecr:InitiateLayerUpload","ecr:UploadLayerPart",
                 "ecr:CompleteLayerUpload","ecr:PutImage","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],
      "Resource": "arn:aws:ecr:ap-south-1:<ACCOUNT_ID>:repository/speech-judge" },
    { "Effect": "Allow",
      "Action": ["ec2:StartInstances","ec2:StopInstances"],
      "Resource": "arn:aws:ec2:ap-south-1:<ACCOUNT_ID>:instance/<INSTANCE_ID>" },
    { "Effect": "Allow",
      "Action": ["ec2:DescribeInstances","ec2:DescribeInstanceStatus"], "Resource": "*" },
    { "Effect": "Allow", "Action": "ssm:SendCommand",
      "Resource": ["arn:aws:ssm:ap-south-1::document/AWS-RunShellScript",
                   "arn:aws:ec2:ap-south-1:<ACCOUNT_ID>:instance/<INSTANCE_ID>"] },
    { "Effect": "Allow",
      "Action": ["ssm:GetCommandInvocation","ssm:ListCommandInvocations","ssm:DescribeInstanceInformation"],
      "Resource": "*" }
  ]
}
```

Note the role ARN for section 9.

---

## 6. Launch the EC2 instance

Console: **EC2 → Launch instance**

| Field | Value |
|---|---|
| Name | `speechjudge-gpu` |
| AMI | search *"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"* (Quick Start → *AWS Marketplace*/Deep Learning AMIs); pick the newest |
| Instance type | `g5.2xlarge` (or `g5.xlarge` / `g6.*` per section 2.2) |
| Key pair | `speechjudge-key` |
| Network / Security group | default VPC, existing group `speechjudge-sg` |
| Storage | 1 × **150 GiB gp3**, *Encrypted* |
| Advanced details → IAM instance profile | `speechjudge-ec2-role` |
| Advanced details → Metadata version | **V2 only** |
| Advanced details → Shutdown behavior | **Stop** (default) |

Launch, wait for "2/2 checks passed", then note the **Instance ID** (`i-0123...`) and **Public IPv4 address**.
Put the instance ID into the IAM policy from section 5.5.

Optional: **Elastic IP** (EC2 → Elastic IPs → Allocate → Associate) if you want an address that survives
stop/start. Otherwise look up the current IP each time (section 12.1 shows how).

---

## 7. Optional dry run on a cheap CPU instance (recommended)

GPU quota approval can take a while, and most of the moving parts (ECR, GitHub Actions, SSM, the tunnel, your
frontend) do not need a GPU. You can prove all of them first on a small instance with **mocked models**:

1. Launch a `t3.large` (2 vCPU, 8 GiB) with the same AMI family *or* plain Ubuntu 22.04, 60 GiB disk, same
   security group, same instance role. (No GPU, so no quota needed.)
2. Follow sections 8 to 11, but in the server `.env` (section 8.4) use the **dry-run profile**:
   `SJ_ENVIRONMENT=local`, `SJ_MOCK_MODE=true`, `SJ_GPU_PRESENT=false`. (Production mode forbids mock mode on purpose.)
3. With the tunnel open, your local frontend should show "Backend ready" and return mocked scores.
4. Stop and terminate the dry-run instance, then repeat on the GPU instance with the production profile.

---

## 8. Prepare the server (one time)

### 8.1 Connect to Ubuntu

**SSH from PowerShell** (needs port 22 open to your IP):

```powershell
ssh -i "C:\keys\speechjudge.pem" ubuntu@<PUBLIC_IP>
```

Accept the host-key prompt. You are in when the prompt reads `ubuntu@ip-...:~$`.

**No-SSH alternative (SSM Session Manager)**, uses the instance role and no open port:

```powershell
aws ssm start-session --target <INSTANCE_ID> --region ap-south-1
```

(Console alternative: EC2 → select instance → *Connect* → *Session Manager* or *EC2 Instance Connect*.)

### 8.2 Verify the GPU, Docker and tools

Run on the server:

```bash
nvidia-smi                                     # must list the A10G/L4 and a driver version
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # GPU visible inside a container
aws --version
curl --version | head -1
df -h /                                        # ~150 GB
free -h                                        # 32 GB on g5.2xlarge
sudo apt-get update && sudo apt-get install -y jq
```

If `docker` or the GPU-in-container test fails, install them (standard upstream instructions):

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER            # log out and back in afterwards

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If `aws` is missing: `sudo snap install aws-cli --classic`.
Confirm the SSM agent is running: `systemctl status snap.amazon-ssm-agent.amazon-ssm-agent` (or
`amazon-ssm-agent`); it must be *active*.

### 8.3 Directories

```bash
sudo mkdir -p /opt/speechjudge/hf-cache
sudo chown -R 10001:10001 /opt/speechjudge/hf-cache      # UID of the 'judge' user inside the image
sudo chown ubuntu:ubuntu /opt/speechjudge
```

### 8.4 Environment file (`/opt/speechjudge/.env`)

This is the server's own configuration (it is **not** your laptop's `.env`). Create it:

```bash
nano /opt/speechjudge/.env
chmod 600 /opt/speechjudge/.env
```

**Production profile (GPU instance):**

```ini
SJ_ENVIRONMENT=production
SJ_ENABLE_DOCS=false
SJ_LOG_LEVEL=INFO

# Paste the value you generated in section 4.3. The service refuses to start if it is shorter than 24 chars.
SJ_SERVICE_CREDENTIALS=PASTE_LONG_RANDOM_VALUE_HERE

SJ_MOCK_MODE=false
SJ_GPU_PRESENT=true                   # GPU profile (replaces the old SJ_DEVICE=cuda)
SJ_JOB_BACKEND=inline
SJ_LOAD_MODELS_IN_API=true

SJ_WHISPER_MODEL=large-v3
SJ_WHISPER_COMPUTE_TYPE=float16
SJ_WHISPER_LANGUAGE=en

# Service switches: both must be true in production (false = fake placeholder scores; the service refuses to start)
SJ_PRONUNCIATION_ENABLED=true
SJ_QWEN_ENABLED=true
SJ_PRONUNCIATION_BACKEND=phoneme
SJ_QWEN_BACKEND=transformers
SJ_QWEN_MODEL_PATH=Qwen/Qwen3-4B
SJ_LLM_RETRY_LIMIT=2

SJ_FFMPEG_PATH=ffmpeg
SJ_MAX_AUDIO_BYTES=15728640
SJ_MAX_AUDIO_SECONDS=180
SJ_REQUEST_TIMEOUT_S=180
SJ_MAX_CONCURRENT_EVALUATIONS=12      # more simultaneous sync calls get HTTP 429 + Retry-After instead of timing out
SJ_PROCESSING_TIMEOUT_S=300
SJ_JOB_RETRY_LIMIT=2
SJ_JOB_RETENTION_S=86400
```

**Dry-run profile (section 7, CPU instance):** same file but
`SJ_ENVIRONMENT=local`, `SJ_MOCK_MODE=true`, `SJ_GPU_PRESENT=false` (this also selects `small.en` / `int8`; remove the
explicit `SJ_WHISPER_MODEL` / `SJ_WHISPER_COMPUTE_TYPE` lines), and `SJ_SERVICE_CREDENTIALS` any value.

Notes:

- `SJ_HOST` and `SJ_PORT` are set by the start script (the container must listen on `0.0.0.0:8000`).
- Keep the real credential out of GitHub, chat and screenshots. To rotate it: edit this file, then
  `sudo systemctl restart speechjudge`, and update the frontend (`FRONTEND_API_KEY`) and your Main Application.
- Inline jobs are kept in memory and expire after `SJ_JOB_RETENTION_S`; a restart clears them. Nothing permanent
  is stored.

### 8.5 Start script and deploy settings

```bash
# which registry/image to pull (edit the three values)
cat > /opt/speechjudge/deploy.env <<'EOF'
REGION=ap-south-1
REGISTRY=<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
REPO=speech-judge
EOF

# container start script: pulls :latest, then runs the API in the foreground (systemd supervises it)
cat > /opt/speechjudge/start.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/speechjudge/deploy.env
IMAGE="$REGISTRY/$REPO:latest"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker pull "$IMAGE"
docker rm -f speechjudge >/dev/null 2>&1 || true

GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then GPU_FLAG="--gpus all"; fi

# Published on 127.0.0.1 only (tunnel / reverse proxy). For option 11.B change to: -p 8000:8000
exec docker run --name speechjudge --rm $GPU_FLAG \
  --env-file /opt/speechjudge/.env \
  -e SJ_HOST=0.0.0.0 -e SJ_PORT=8000 \
  -p 127.0.0.1:8000:8000 \
  -v /opt/speechjudge/hf-cache:/home/judge/.cache \
  "$IMAGE"
EOF

# readiness helper (used by GitHub Actions and by you)
cat > /opt/speechjudge/wait_ready.sh <<'EOF'
#!/usr/bin/env bash
for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:8000/api/v1/ready >/dev/null; then echo "ready"; exit 0; fi
  sleep 5
done
echo "not ready after 30 minutes"; exit 1
EOF

chmod +x /opt/speechjudge/start.sh /opt/speechjudge/wait_ready.sh
```

### 8.6 systemd service (starts the backend automatically whenever the instance boots)

```bash
sudo tee /etc/systemd/system/speechjudge.service >/dev/null <<'EOF'
[Unit]
Description=Speech Judge backend container
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
ExecStart=/opt/speechjudge/start.sh
ExecStop=/usr/bin/docker stop speechjudge
Restart=on-failure
RestartSec=20
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable speechjudge          # start at every boot; do NOT start it yet: no image exists until section 10
```

---

## 9. GitHub Actions automation

### 9.1 Repository settings

GitHub → repo → **Settings → Secrets and variables → Actions**

| Kind | Name | Value |
|---|---|---|
| Variable | `AWS_REGION` | `ap-south-1` |
| Variable | `ECR_REPOSITORY` | `speech-judge` |
| Variable | `EC2_INSTANCE_ID` | `i-0123456789abcdef0` |
| Secret | `AWS_ROLE_ARN` | ARN of the role from section 5.5 |

Only `main` can assume the role (trust policy). Do not add AWS access keys to GitHub.

### 9.2 Workflow 1: CI (tests on every push/PR)

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
      - name: Install (mirrors the local test environment, without the heavy ML stack)
        run: pip install -r requirements-dev.txt praat-parselmouth==0.4.7 pyloudnorm==0.2.0
      - name: Unit + integration tests
        run: python -m pytest -q
```

### 9.3 Workflow 2: build the image and push it to ECR

`.github/workflows/build-push.yml`

```yaml
name: Build and push image
on:
  push:
    branches: [main]
    paths: ["speech-judge/**", ".github/workflows/build-push.yml"]
  workflow_dispatch:

permissions:
  id-token: write      # OIDC
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # The ML image is several GB; free runner disk before building.
      - name: Free disk space
        uses: jlumbroso/free-disk-space@main
        with:
          tool-cache: false
          android: true
          dotnet: true
          haskell: true
          large-packages: false

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}

      - id: ecr
        uses: aws-actions/amazon-ecr-login@v2

      - uses: docker/setup-buildx-action@v3

      - uses: docker/build-push-action@v6
        with:
          context: speech-judge
          push: true
          tags: |
            ${{ steps.ecr.outputs.registry }}/${{ vars.ECR_REPOSITORY }}:latest
            ${{ steps.ecr.outputs.registry }}/${{ vars.ECR_REPOSITORY }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

This only updates the image in ECR. It never starts the expensive server. The instance pulls `:latest` every
time it starts (section 8.5), or when you run "redeploy" below.

### 9.4 Workflow 3: start / stop / redeploy the server from GitHub

`.github/workflows/server-control.yml` (manual: *Actions → Server control → Run workflow*). This is the complete file.

```yaml
name: Server control
on:
  workflow_dispatch:
    inputs:
      action:
        description: What to do with the EC2 server
        type: choice
        options: [status, start, stop, redeploy]
        default: status

permissions:
  id-token: write
  contents: read

jobs:
  control:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    env:
      AWS_REGION: ${{ vars.AWS_REGION }}
      INSTANCE_ID: ${{ vars.EC2_INSTANCE_ID }}
      BASH_ENV: /tmp/sj_helpers.sh        # bash sources this file in every later step
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}

      - name: Define SSM helper
        run: |
          cat > /tmp/sj_helpers.sh <<'EOF'
          send_and_wait() {               # run a shell command on the instance and wait for the result
            local cid=""
            for i in $(seq 1 20); do      # the SSM agent needs a moment after boot
              cid=$(aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
                    --parameters "commands=[\"$1\"]" --query Command.CommandId --output text 2>/dev/null) && break
              sleep 15
            done
            [ -n "$cid" ] || { echo "could not reach the instance through SSM"; return 1; }
            for i in $(seq 1 400); do     # up to ~33 minutes (the first boot downloads the models)
              s=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$INSTANCE_ID" \
                    --query Status --output text 2>/dev/null || echo Pending)
              case "$s" in
                Success) return 0 ;;
                Failed|Cancelled|TimedOut) echo "SSM command $s"; return 1 ;;
              esac
              sleep 5
            done
            return 1
          }
          EOF

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
          send_and_wait "/opt/speechjudge/wait_ready.sh"
          echo "Server is ready. Public IP:"
          aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

      - name: Redeploy (pull the newest image and restart the service)
        if: inputs.action == 'redeploy'
        run: send_and_wait "systemctl restart speechjudge; sleep 20; /opt/speechjudge/wait_ready.sh"

      - name: Stop
        if: inputs.action == 'stop'
        run: |
          aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
          aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
          echo "Instance stopped. Compute billing has ended; EBS storage is still billed."
```

Notes:

- `start` waits until the API reports ready (normally 5–8 minutes; up to ~20 minutes on the very first boot while
  ~12 GB of models download). If a step fails, open the command output in **Systems Manager → Run Command**.
- The job runs on GitHub's servers, not yours, so it needs no open port on the EC2 instance; it talks to it
  through SSM.
---

## 10. First start and verification

### 10.1 Build the first image

Push your code (with the three workflow files) to `main`, or run **Actions → Build and push image → Run workflow**.
Wait for green, then confirm in ECR that the repository holds `latest`.

### 10.2 Start the service on the server

```bash
sudo systemctl start speechjudge
journalctl -u speechjudge -f            # watch: image pull, "Loading service ...", "All shared services loaded"
```

The first start downloads the models into `/opt/speechjudge/hf-cache` (~12–13 GB). In another terminal:

```bash
/opt/speechjudge/wait_ready.sh          # prints "ready" when all four services are loaded
curl -s http://127.0.0.1:8000/api/v1/ready
# {"status":"ready","whisper":true,"gopt":true,"acoustic":true,"qwen":true}
nvidia-smi                              # memory used by the process (expect roughly 12-15 GB)
```

### 10.3 Smoke tests and speed measurement (on the server)

```bash
export SJ_KEY='<your server credential>'

# 1. no models involved: should return score 1.0
curl -s -X POST http://127.0.0.1:8000/api/v1/evaluate \
  -H "Authorization: Bearer $SJ_KEY" -H "Content-Type: application/json" \
  -d '{"request_id":"smoke-1","question_type":"JUMBLED","expected_text":"the cat sat on the mat","question_config":{"submitted_text":"the cat sat on the mat"}}' | jq '.status,.score'

# 2. wrong/missing credential must be rejected (expect 401)
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/api/v1/evaluate -H "Content-Type: application/json" -d '{}'

# 3. real audio: copy any 16 kHz WAV to the server first (scp -i key.pem sample.wav ubuntu@IP:~/sample.wav)
time (jq -n --arg a "$(base64 -w0 ~/sample.wav)" \
  '{request_id:"smoke-2",question_type:"REPEAT",audio:$a,expected_text:"The weather is beautiful today."}' \
  | curl -s -X POST http://127.0.0.1:8000/api/v1/evaluate \
      -H "Authorization: Bearer $SJ_KEY" -H "Content-Type: application/json" -d @- | jq '.status,.score,.evaluation.gopt.method')
```

The timing from check 3 (run it 3–5 times, and once with a `QUESTION_ANSWER` request) replaces the planning
numbers in section 3. Use them to decide how many students one server can take per hour.

For a proper measurement use the saved load-test scripts in `tests/load/` (clone the repo on the server, or run
them from your laptop with `--url`). `python tests/load/concurrency_load.py --type exam --max 15` fires the
8/16/10/24/2 exam mix at increasing concurrency, and `bench_whisper.py`, `bench_phoneme.py` and `bench_acoustic.py`
time each model. `tests/load/README.md` holds the CPU-laptop baseline to compare against.

**Trying things with fewer models:** in a non-production `.env` (`SJ_ENVIRONMENT=local`), set
`SJ_PRONUNCIATION_ENABLED=false` and/or `SJ_QWEN_ENABLED=false` to mock those services. Reading/Repeat and
Q&A/Storytelling then return fixed placeholder scores, flagged `MOCKED_PRONUNCIATION` / `MOCKED_LANGUAGE`, and no
model is loaded for them. Production mode refuses to start that way on purpose.

---

## 11. Connect your local frontend to the deployed backend

The frontend's local proxy (`frontend/server.py`) forwards `/api/v1/*` to whatever `BACKEND_URL` says and adds
the credential. Environment variables win over the values it reads from `speech-judge/.env`, so **you do not
edit any file**: set two variables in the terminal you start the proxy from.

Checklist before starting: the server is running and ready (section 12.1), and you stopped your **local**
backend (or use a different local port, below) so it does not confuse things.

### Option A (recommended): SSH tunnel. Nothing is exposed; traffic is encrypted

Terminal 1, keep it open (`8001` avoids clashing with a local backend on 8000):

```powershell
ssh -i "C:\keys\speechjudge.pem" -N -L 8001:127.0.0.1:8000 ubuntu@<PUBLIC_IP>
```

Terminal 2, from the repo root:

```powershell
$env:BACKEND_URL     = "http://127.0.0.1:8001"
$env:FRONTEND_API_KEY = "<the SERVER credential, not your local one>"
python frontend\server.py
```

Open http://127.0.0.1:8080. The header badge should read **Backend ready** with four green service badges.

*SSM variant (no SSH port, needs the Session Manager plugin):*

```powershell
aws ssm start-session --target <INSTANCE_ID> --region ap-south-1 `
  --document-name AWS-StartPortForwardingSession --parameters "portNumber=8000,localPortNumber=8001"
```

### Option B: direct to the public IP (quick, but unencrypted)

1. Security group: allow **8000/tcp from your IP /32**.
2. On the server, change the publish flag in `/opt/speechjudge/start.sh` to `-p 8000:8000`, then
   `sudo systemctl restart speechjudge`.
3. Locally:

```powershell
$env:BACKEND_URL     = "http://<PUBLIC_IP>:8000"
$env:FRONTEND_API_KEY = "<server credential>"
python frontend\server.py
```

Use this only for short tests: the token and the audio cross the internet in plain text. Close the rule afterwards.

### Option C: HTTPS with your own domain (what the Main Application should use in production)

1. Point an `A` record (for example `judge.example.com`) at the server's Elastic IP.
2. Security group: allow 80 and 443.
3. Install Caddy on the server (automatic Let's Encrypt certificates):

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
echo 'judge.example.com { reverse_proxy 127.0.0.1:8000 }' | sudo tee /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

4. Locally: `BACKEND_URL=https://judge.example.com`, `FRONTEND_API_KEY=<server credential>`.

With a load balancer or proxy in front, raise the idle/read timeout above 60 s or use the asynchronous mode.

### Test checklist (any option)

1. Header shows **Backend ready**; refresh the page after the server finishes loading.
2. **Edge cases** tab → *Run all*: every row should show "as expected".
3. **Reading** → record or upload a WAV → score, transcript, pronunciation panel appear.
4. **Q&A** (async) → result arrives after some seconds; **History** tab lists the runs.
5. Optional automated check through the same path:
   `cd speech-judge; $env:SJ_E2E_URL="http://127.0.0.1:8080"; .\.venv\Scripts\python -m pytest tests\e2e\test_real_stack.py --real -q`
6. When done, close the tunnel (Ctrl+C) and **stop the server** (section 12.2).

If the tunnel or `BACKEND_URL` is wrong the proxy returns `502 Backend unreachable`; a wrong credential gives `401`.

---

## 12. Day-to-day operation

### 12.1 Start before a session

Either run **Actions → Server control → start**, or from PowerShell:

```powershell
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region ap-south-1
aws ec2 wait instance-status-ok --instance-ids <INSTANCE_ID> --region ap-south-1
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --region ap-south-1 `
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

The service starts by itself (systemd). Wait until `/api/v1/ready` says ready (about 5–8 minutes), then open
the tunnel. Start it **before** students arrive, not when they do.

### 12.2 Stop after a session

Wait until the queue is empty (no more jobs `QUEUED`/`PROCESSING`; `journalctl -u speechjudge -n 50` shows no
activity), fetch any results you need (jobs expire from memory after 24 h and on restart), then:

```powershell
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region ap-south-1
```

or run *Server control → stop*. **Stopped means not billed for compute.** Do not use "terminate", which deletes
the disk and the model cache.

### 12.3 Safety nets against a forgotten server

- **Scheduled stop (recommended):** EventBridge → *Scheduler* → create a schedule (one-time or recurring) whose
  target is the universal target `EC2 StopInstances` with your instance ID, set a few hours after each session.
- **Idle shutdown on the server (optional):** stops the instance after an hour with no evaluation traffic.

```bash
sudo tee /opt/speechjudge/idle_shutdown.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
# Shut down if no evaluate/jobs request was logged in the last 60 minutes and the service has been up > 60 min.
UP=$(( $(date +%s) - $(date -d "$(systemctl show speechjudge -p ActiveEnterTimestamp --value)" +%s) ))
[ "$UP" -lt 3600 ] && exit 0
N=$(docker logs --since 60m speechjudge 2>&1 | grep -cE 'POST /api/v1/evaluate|GET /api/v1/jobs' || true)
[ "$N" -eq 0 ] && /sbin/shutdown -h now
EOF
sudo chmod +x /opt/speechjudge/idle_shutdown.sh
echo '*/10 * * * * root /opt/speechjudge/idle_shutdown.sh' | sudo tee /etc/cron.d/speechjudge-idle
```

Shutdown behaviour is "stop" (section 6), so this stops, not terminates. Remove the cron file while you are
testing interactively if it would interrupt you.

### 12.4 Budget alert

Billing → **Budgets → Create budget** → cost budget (e.g. $50/month) → alerts at 50 % and 100 % to your e-mail.

### 12.5 Deploying new code

1. Merge to `main` → CI runs → *Build and push image* updates `:latest`.
2. If the server is **stopped**: nothing else to do; the next start pulls the new image.
3. If the server is **running**: run *Server control → redeploy*.

Roll back by pointing `start.sh` at a previous tag (every build is also tagged with its commit SHA), e.g.
`IMAGE="$REGISTRY/$REPO:<old-sha>"`, then `sudo systemctl restart speechjudge`.

### 12.6 Logs and health

```bash
journalctl -u speechjudge -n 100 --no-pager      # service + container output
docker logs -f speechjudge                       # request log and errors
curl -s http://127.0.0.1:8000/api/v1/health      # liveness
curl -s http://127.0.0.1:8000/api/v1/ready       # per-model readiness
nvidia-smi -l 2                                  # GPU use while a batch is running
df -h / ; free -h
```

---

## 13. Optional: queue mode for larger cohorts

The repo's `docker-compose.yml` runs **Redis + API gateway + a GPU worker**. Jobs then survive API restarts and
you can add workers (one per GPU instance pointing at the same Redis). It needs Docker Compose and the NVIDIA
runtime on the server, and `.env` with a strong credential.

**One catch with your frontend:** in this mode the API container does not load models, so the
*synchronous* `/evaluate` endpoint returns `503 "This node does not serve synchronous evaluation"`. The frontend
offers a per-test "Mode" selector, so choose **Asynchronous** for every test type (the default for Q&A and
Storytelling only). Your Main Application should also use `/evaluate/async`. Single-node mode (this document)
supports both.

---

## 14. Known risks and first-boot fixes (read before the first GPU boot)

These are code/environment points I could not test from a laptop. Check them in section 10.2 and fix only if they
actually appear.

| # | What may happen | Symptom | Fix |
|---|---|---|---|
| 1 | The `transformers` Qwen backend needs the `accelerate` package (it loads with `device_map`). It is now pinned in `requirements-ml.txt`, but the GPU Qwen path itself has never been run. | Container exits at start; log shows an error for service `qwen`. | Read the log line; alternative: install Ollama on the host and set `SJ_QWEN_BACKEND=ollama` with a reachable `SJ_OLLAMA_URL`. |
| 2 | `faster-whisper` on GPU needs cuDNN/cuBLAS libraries findable by the loader; the `python:3.12-slim` image gets them only through pip packages. | Log shows `Could not load library libcudnn...` or `libcublas...` while loading `whisper`. | Rebuild on an NVIDIA CUDA runtime base image (for example `nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04` plus Python 3.12), or set `LD_LIBRARY_PATH` to the `nvidia/*/lib` folders inside the installed `site-packages`. |
| 3 | The phoneme model currently runs on the **CPU** even on a GPU server. | Reading/Repeat take a couple of seconds longer than they would on GPU. | Acceptable to start with; moving it to the GPU is a small code change for later. |
| 4 | Image is large (several GB). | Slow first pull; GitHub runner may run out of disk. | The workflow frees disk first; if it still fails use a larger GitHub runner. |
| 5 | GPU vCPU quota is 0. | `VcpuLimitExceeded` when launching. | Section 4.2. |
| 6 | Pronunciation thresholds and the Indian-English tolerance list are engineering values. | Scores feel too strict/lenient on real students. | Tune `config/scoring.yaml` and `config/accents/indian_english.yaml` with human-rated recordings; no deployment change needed beyond rebuilding the image. |
| 7 | CI differs from your laptop (no ML libraries). | A prosody/acoustic unit test fails in GitHub Actions only. | The CI step installs `praat-parselmouth` and `pyloudnorm` to mirror the laptop; if a test still differs, compare versions with `pip freeze` locally. |

---

## 15. Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `VcpuLimitExceeded` / "insufficient capacity" | Quota (4.2), or the AZ has no capacity: retry later or launch in another subnet/AZ. |
| SSH `Permission denied (publickey)` | Wrong user (use `ubuntu`), wrong key file, or key permissions (5.2). |
| SSH times out | Security group port 22 not open to your **current** IP. |
| `systemctl start speechjudge` fails on `docker pull` | Instance role lacks ECR read, wrong `deploy.env`, or no image pushed yet (10.1). |
| Container exits immediately | `docker logs speechjudge` / `journalctl -u speechjudge`. Production mode refuses weak credentials, mock mode or docs enabled; the message says which. |
| `/ready` stays 503 | Models still loading (first boot: 10–20 min) or a service failed to load (see logs, section 14). |
| `502 Backend unreachable` in the frontend | Tunnel closed, wrong `BACKEND_URL`, server stopped/not ready. |
| `401` from the frontend | `FRONTEND_API_KEY` is not the server's credential. |
| `504` on a synchronous call | Raise `SJ_REQUEST_TIMEOUT_S` or use the asynchronous mode. |
| GitHub `Not authorized to perform sts:AssumeRoleWithWebIdentity` | Trust policy `sub` doesn't match `repo:<OWNER>/<REPO>:ref:refs/heads/main`, or the workflow runs from another branch. |
| `aws ssm send-command` says the instance is not valid | SSM agent not running yet, or the instance role is missing `AmazonSSMManagedInstanceCore`. |
| Disk full | `docker system prune -af` (keeps volumes), old images; or enlarge the EBS volume. |

---

## 16. Security checklist

- [ ] Port 8000 is **not** open to `0.0.0.0/0`; SSH limited to your IP (or closed, using SSM).
- [ ] `SJ_ENVIRONMENT=production`, docs disabled, credential ≥ 24 random characters, `.env` is `chmod 600`.
- [ ] No AWS keys or server credentials in GitHub; OIDC role restricted to `main` and to this one instance/repo.
- [ ] EBS encrypted, IMDSv2 required, instance role has only SSM + ECR read.
- [ ] HTTPS (option C) for anything beyond your own testing; the Main Application sends `Authorization: Bearer <credential>`.
- [ ] The service stores no student data permanently (jobs live in memory for 24 h at most); student audio is not logged.
- [ ] Budget alert and scheduled stop are configured.

## 17. Teardown (when you no longer need it)

1. Stop the instance, then **terminate** it (deletes the disk and model cache).
2. Release the Elastic IP if you created one.
3. Delete the ECR repository (or its images), the security group, and the IAM roles/OIDC provider.
4. Remove the GitHub variables/secret.
