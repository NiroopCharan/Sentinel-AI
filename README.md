# Sentinel-AI (HMR)

Autonomous container threat detection and self-healing: **Sense → Reason → Decide → Heal → Verify**, closed
into a feedback loop that re-weights its own fusion model from verified outcomes.

Kernel-level telemetry (via **Falco's eBPF driver** — not custom eBPF probes) and live container resource
stats feed the **HMR (Hybrid Multi-Vector Reconciliation)** algorithm, which fuses four independent signals
into one Threat Score and drives a four-tier graduated response against the Docker Engine API.

```
[ Container Workload ]
        |  syscalls via Falco/eBPF
        v
[ Sense ]     Falco + docker stats/psutil           sensors/
        v
[ Reason ]    HMR fusion: RF + AS + RDS + ICS        engine/hmr_engine.py, engine/fusion.py
        v
[ Decide ]    adaptive-threshold 4-tier response     engine/hmr_engine.py
        v
[ Heal ]      Docker SDK: isolate / kill+respawn      engine/hmr_engine.py
        v
[ Verify ]    canary window on the new container      engine/hmr_engine.py
        v
   verified outcome re-weights the fusion model -> back to Reason    engine/adaptive_weights.py
```

## What's real, what's self-collected, what's a smoke test

| Claim | Status |
|---|---|
| RF (signature match) | **Real-data-validated.** `RandomForestClassifier` trained on NSL-KDD (Tavallaee et al., 2009). Run `scripts/download_nsl_kdd.sh` then `engine/train_models.py`, then `engine/evaluate_benchmark.py` for numbers. |
| AS (anomaly) | **Real-data-validated.** `IsolationForest` trained on NSL-KDD normal rows only. Same scripts as above. |
| RDS (resource drift) | **Self-collected, real but small-N.** `W_rds` is fit from correlations in `data/resource_baseline.csv`, which you generate by actually running containers (`utils/collect_resource_baseline.py`). Until that file exists, the engine uses equal weights and logs a warning — it never fabricates a fitted value. |
| ICS (identity consistency) | **Self-collected, real but small-N, the key novelty.** Trained on `data/identity_sessions.csv` from real owner/intruder SSH sessions against the victim container (`utils/session_logger.py`). Not derivable from NSL-KDD, which has no identity concept. |
| Kernel sensing | Falco's eBPF driver. **No custom eBPF probes are written in this repo.** |
| Adaptive weight re-balancing | Implemented and unit-tested (`engine/adaptive_weights.py`, `tests/test_fusion.py`), with a full audit trail (`data/weight_audit.jsonl`). Framed as a research contribution / patent-draft basis, not a proven improvement — it has not been evaluated against a static-weight baseline on held-out data in this repo. That comparison is future work. |
| Verification / confidence-attested healing | Implemented (`_verify_replica_details` in `engine/hmr_engine.py`) and exercised end-to-end in `tests/test_engine_heal.py` against a fake Docker daemon, including the case where verification **fails**. |
| Two-machine live demo | Scripted (`scripts/setup_demo_network.sh`, `victim/Dockerfile`, `utils/live_attack_demo.py`). Requires a second machine/VM, Docker, and (for the attack side) `nmap`/`hydra`/`hping3`/`sshpass` actually installed — nothing here is simulated in-process. |
| Benchmark numbers (accuracy/F1/ROC-AUC/MCC) | **Only real if you ran `engine/evaluate_benchmark.py` against real `KDDTest+.txt`.** `--allow-synthetic` exists purely so the pipeline can be smoke-tested without the dataset; every synthetic run is loudly watermarked in logs, filenames (`*_synthetic.*`), and figure titles, and the script refuses to mix a synthetically-trained model with the real test set (or vice versa). |
| Live flow features (`sensors/flow_features.py`) | An **approximation** of NSL-KDD-style features from `/proc/net/tcp` + resource deltas on a running container — not the original 1999-era connection-logging tooling. This is a documented domain gap: live feature distributions are not guaranteed to match NSL-KDD's, which is exactly why RDS/ICS exist as independent signals that don't depend on that approximation. |

**Numbers from the actual synthetic smoke-test run on this machine** (clearly not NSL-KDD, included only to
show the pipeline runs end-to-end): HMR (RF+AS fusion) ROC-AUC 0.884, MCC 0.767, balanced accuracy 0.884 on
3000 synthetic rows. Do **not** cite these as NSL-KDD results. Run the real pipeline for real numbers.

## Repository layout

```
Sentinel-AI/
├── config.py                     central configuration (paths, weights, thresholds, env overrides)
├── engine/
│   ├── data_loader.py            NSL-KDD loading + synthetic-data smoke-test fallback (loudly labeled)
│   ├── fusion.py                 TS = sum(w_i . V_i), tiering — pure, no docker/flask dependency
│   ├── adaptive_weights.py       novel contribution #1: adaptive fusion weights + audit trail
│   ├── resource_model.py         RDS: fits W_rds from data/resource_baseline.csv
│   ├── identity_model.py         ICS: the key novelty, trained on data/identity_sessions.csv
│   ├── hmr_engine.py             Reason -> Decide -> Heal -> Verify -> feedback loop
│   ├── train_models.py           trains RF/AS (+ RDS/ICS if their CSVs exist), tunes tau
│   └── evaluate_benchmark.py     honest benchmark: HMR vs NB/SVM/LogReg/KNN, confusion matrices, ROC
├── sensors/
│   ├── resource_monitor.py       live docker stats / psutil -> cpu_util, ram_util, net_io_rate
│   ├── flow_features.py          /proc/net/tcp -> approximate NSL-KDD-style flow features
│   ├── session_audit.py          parses the victim container's SSH session audit log
│   └── telemetry_ingestor.py     Stage 1 (Sense): Falco tail + sensors above -> POST /api/ingest_telemetry
├── forensics/forensic_snapshot.py  encrypted before/after snapshots + MITRE ATT&CK tagging
├── utils/
│   ├── live_attack_demo.py       spawns a victim, runs one real scenario, logs docker-stats rows
│   ├── collect_resource_baseline.py  drives live_attack_demo across scenarios N times -> resource_baseline.csv
│   └── session_logger.py         records real owner/intruder SSH sessions -> identity_sessions.csv
├── victim/                       Dockerfile + session-audit shell hooks for the demo victim pod
├── falco/sentinel_rules.yaml     extra Falco rules (network tools, miner-pattern processes, drop-and-exec)
├── web/app.py, web/templates/dashboard.html   Flask + Socket.IO SOC dashboard
├── scripts/download_nsl_kdd.sh, scripts/setup_demo_network.sh
├── tests/                        22 unit tests (fusion, adaptive weights, identity/resource models, full heal path against a fake Docker daemon)
└── models/                       populated by training; not committed
```

## 0. Environment setup

Target: **WSL2 (Ubuntu) inside Windows, with VS Code's WSL extension.** See `requirements.txt` for exact
pinned versions (flask, flask-socketio, docker, psutil, scikit-learn, pandas, numpy, joblib, loguru,
matplotlib, seaborn, cryptography).

```bash
# Windows PowerShell (admin), if WSL isn't installed:
wsl --install -d Ubuntu

# Inside Ubuntu WSL:
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip git curl build-essential sshpass hping3 nmap hydra

cd ~/Sentinel-AI
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Docker**: install Docker Desktop for Windows, enable WSL2 integration for your Ubuntu distro (Settings →
Resources → WSL Integration), then verify from inside WSL: `docker ps`.

**Falco** (kernel event sensor, eBPF driver):
```bash
curl -fsSL https://falco.org/repo/falcosecurity-packages.asc | sudo gpg --dearmor -o /usr/share/keyrings/falco-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/falco-archive-keyring.gpg] https://download.falco.org/packages/deb stable main" | sudo tee /etc/apt/sources.list.d/falcosecurity.list
sudo apt update && sudo apt install -y falco
sudo cp falco/sentinel_rules.yaml /etc/falco/falco_rules.local.yaml
```
Edit `/etc/falco/falco.yaml`: set `json_output: true` and `file_output: {enabled: true, filename: /var/log/falco/events.log}`,
then `sudo systemctl restart falco`. If eBPF driver probing fails on your specific WSL kernel, this is a
known WSL limitation: run Falco on a native Linux VM for the Sense stage instead and point
`sensors/telemetry_ingestor.py --falco-log` at wherever that VM's log is mounted/shared.

## 1. Build and run, step by step

```bash
# 1. Data pipeline
scripts/download_nsl_kdd.sh                 # fetches KDDTrain+.txt / KDDTest+.txt into data/raw/
python -m engine.train_models                # trains RF + AS, tunes tau, writes models/*.joblib + meta.json

# 2. Honest evaluation
python -m engine.evaluate_benchmark           # real test-set numbers + confusion_matrix_real.png + roc_comparison_real.png

# 3. Victim container
scripts/setup_demo_network.sh                 # builds victim/, runs sentinel-victim-pod on host port 2222

# 4. Resource baseline (self-collected)
python -m utils.collect_resource_baseline --runs 10 --duration 25
python -m engine.train_models                 # re-run: now also fits models/rds_weights.json

# 5. Identity sessions (self-collected) — run a real "owner" session, then a real "intruder" session
ssh demoUser@127.0.0.1 -p 2222                 # type a few plausible commands, then exit
python -m utils.session_logger --container sentinel-victim-pod --label owner
# ... repeat a few times for a real distribution, then simulate an intruder session (fast, mechanical, wrong hours):
python -m utils.live_attack_demo --scenario session_hijack
python -m utils.session_logger --container sentinel-victim-pod --label intruder --scenario session_hijack
python -m engine.train_models                 # re-run: now also fits models/identity_model.json

# 6. Dashboard + sensors (two terminals)
python -m web.app                              # http://localhost:5000
python -m sensors.telemetry_ingestor --container sentinel-victim-pod

# 7. Tests
python -m unittest discover -s tests -v        # or: pytest
```

`web/app.py` auto-detects whether a Docker daemon is reachable. Without one it starts in **dry-run**: it
still scores every event and opens/escalates incidents, but Isolate/Kill&Restart become
`dry_run_isolate` / `dry_run_kill_restart` log entries instead of real Docker API calls. Use the dashboard's
"inject benign" / "inject attack" buttons (`POST /api/demo/inject`) to rehearse the UI without live traffic
— every event they produce is tagged `demo: true` and uses invented telemetry values, never real sensor data.

## 2. The HMR algorithm

```
TS = w_rf·RF + w_as·AS + w_rds·RDS + w_ics·ICS      (weights renormalised over whichever vectors are present)
```

- **RF** — `RandomForestClassifier.predict_proba` on NSL-KDD flow features (`duration, src_bytes, dst_bytes,
  count, srv_count, same_srv_rate, diff_srv_rate` + `protocol_type` by default; `--feature-set full` uses
  all 41 NSL-KDD columns).
- **AS** — `IsolationForest` trained only on normal rows; `AS = clip((-score - ASmin)/(ASmax-ASmin), 0, 1)`.
- **RDS** — `RDS = X_r · W_rds` over `(cpu_util, ram_util, net_io_rate)` drift vs. a rolling per-container
  healthy baseline (falls back to a self-collected global baseline for brand-new containers). One saturated
  feature yields `RDS` near that feature's weight, not 1.0 — this is the specified weighted-sum behaviour,
  reported as-is.
- **ICS** — Mahalanobis-style distance of `(hour_dev, timing_entropy, lineage_dev, origin_dev)` from an
  identity's own historical session distribution. 0 = consistent with the owner, 1 = clearly not. This is
  what catches a stolen-credential/session-hijack attacker holding a *valid* shell that RF/AS/RDS alone
  would score as normal — see `tests/test_identity_resource.py` for owner-vs-intruder separation on
  self-generated fixtures.

**Starting weights**: `rf=0.40, as=0.25, rds=0.15, ics=0.20` (sum to 1.0), overridden at runtime by the
adaptive mechanism.

**Decision thresholds** (`tau` is F1-tuned per training run on the held-out 20% of the train partition,
default fallback 0.60; `margin=0.20`):

| Threat Score | Tier |
|---|---|
| `TS < tau/2` | Monitor |
| `tau/2 <= TS < tau` | Log & Watch |
| `tau <= TS < tau+margin` | Isolate (quarantine network, drop egress, keep internal mesh) |
| `TS >= tau+margin` | Kill & Restart (forensic snapshot → freeze → kill → respawn → verify) |

Disruptive actions (Isolate, Kill & Restart) require `SENTINEL_ACT_STREAK` (default 2) consecutive
over-threshold events, to debounce single-sample spikes.

## 3. Novel contribution #1 — adaptive fusion weights

`engine/adaptive_weights.py`. Each vector keeps a rolling accuracy (`alpha=0.05` EMA), updated **only from
verified kill/restart outcomes**, never from unverified predictions. Weights are
`prior_i · exp(3 · acc_i)`, renormalised — at the 0.5 starting accuracies this reproduces the prescribed
0.40/0.25/0.15/0.20 start, and diverges from it as a vector's track record diverges. Every update is
appended to `data/weight_audit.jsonl` with a timestamp, the triggering incident id, and the before/after
weights — the audit trail behind both the "dynamic weight re-balancing" future-work claim and any patent
draft ("method for reweighting a multi-vector threat score based on post-remediation verification
outcomes").

A deliberate departure from the literal build-prompt snippet: updating *all four* vectors with the same
boolean on every verified heal would keep their accuracies identical forever, so weights could never
diverge. Instead each vector is judged on whether *its own* verdict (score ≥ 0.5) agreed with whether the
incident was independently confirmed — by a severe-priority Falco rule hit, or a known-bad process name
(`xmrig`, `stress-ng`, `nc`, `hydra`, …) in the forensic snapshot. See `_apply_verified_feedback` in
`engine/hmr_engine.py`.

## 4. Novel contribution #2 — verification / confidence-attested healing

`_execute_kill_restart` in `engine/hmr_engine.py` never assumes a kill-and-respawn worked: it snapshots,
freezes, kills, removes, respawns from the clean image, then runs `_verify_replica_details` — a canary
window (default 5s) checking the new container is running and its CPU stays under a calibrated healthy
ceiling (0.5% default, or the collected benign baseline mean + 3σ once one exists). This directly answers
*"what if the AI wrongly kills a healthy container?"* — `tests/test_engine_heal.py` exercises both the
verified-clean and failed-verification paths end-to-end, including that a failed verification leaves the
adaptive weights untouched.

## 5. Forensics + MITRE ATT&CK

Every Kill & Restart captures a **before** snapshot (pre-kill: process table + cmdlines + `/proc/<pid>/maps`
where host PIDs are visible, `/proc/net/tcp`, filesystem diff, container logs, SSH session audit log,
`docker inspect`) and an **after** snapshot on the healed replica. Artifacts are packed into a `tar.gz` and
encrypted with Fernet (`data/forensics/forensics.key`, mode 600, generated on first use); a plaintext
`*.summary.json` sits alongside each for the dashboard's before/after viewer. MITRE ATT&CK technique tags
come from two sources, and the dashboard visually distinguishes them: Falco rule names (authoritative,
e.g. `T1003.008` from "Read sensitive file untrusted") and explicit vector-value heuristics (labelled
`source: vector-heuristic`, e.g. a sustained high-ICS session tagged `T1078 Valid Accounts`).

Decrypt a snapshot: `python -m forensics.forensic_snapshot --decrypt data/forensics/INC-..._before.tar.gz.enc --out /tmp/out.tar.gz`

## 6. Two-machine live demo

See `scripts/setup_demo_network.sh` and build-prompt §3 for full detail. Summary:

- **Defender machine** (this repo): Docker, Falco, `python -m web.app`, `python -m sensors.telemetry_ingestor`.
- **Attacker machine**: a second laptop or a Kali/Ubuntu VM on the same isolated hotspot (not shared/campus
  Wi-Fi — get sign-off before running brute-force/flood traffic on any monitored network). Needs `nmap`,
  `hydra`, `hping3`, an SSH client.

```bash
# Attacker machine, against the Windows host's LAN IP (ipconfig, Wi-Fi adapter), port 2222:
nmap -sV <defender_lan_ip>
hydra -l demoUser -P scripts/rockyou-sample.txt ssh://<defender_ip>:2222
ssh demoUser@<defender_ip> -p 2222          # cat /etc/shadow, off-hours, mechanical commands
hping3 -S -p 2222 --flood <defender_ip>
```
Run one pass as a scripted "owner" session (correct key/plausible hour/human-paced typing) and one as the
intruder path above, to show the Identity Consistency Score differentiating them even though both hold a
valid shell. **Audience sees**: attacker's SSH session dies mid-command when Sentinel-AI kills the
container; defender's dashboard shows the threat gauge climbing, all four vectors updating, tier
escalating to Kill & Restart, the verification check passing, and a forensic snapshot entry appearing.

**Rehearse twice before presenting**, including once where the attack *isn't* caught (lower `SENTINEL_ACT_STREAK`
to 1, or widen `tau`, to force a miss on purpose), so you know what that looks like and can discuss it
honestly if it happens live.

## 7. Known limitations (future work, not done)

- Adaptive weights vs. a static-weight baseline: not A/B-evaluated in this repo.
- `sensors/flow_features.py` is an approximation of NSL-KDD-style features from a live container, not the
  original dataset's connection-logging methodology — a documented domain gap between training and
  deployment distributions.
- RDS and ICS have no equivalent in NSL-KDD and so cannot be included in `engine/evaluate_benchmark.py`'s
  NSL-KDD numbers; their quality is assessed only via the self-collected CSVs and the unit tests in
  `tests/test_identity_resource.py`.
- `web/app.py`'s dashboard has no authentication beyond an optional shared-secret header on the ingest
  endpoint — fine for an isolated demo network, not for production exposure.
