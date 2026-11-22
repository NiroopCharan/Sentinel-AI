#!/usr/bin/env bash
# One-shot defender-side setup for the two-machine demo (build prompt §3.2). Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] building victim image"
docker build -t sentinel-victim victim/

echo "[2/4] (re)starting sentinel-victim-pod on host port 2222"
docker rm -f sentinel-victim-pod >/dev/null 2>&1 || true
docker run -d --name sentinel-victim-pod -p 0.0.0.0:2222:22 sentinel-victim

echo "[3/4] waiting for sshd"
for i in $(seq 1 15); do
  docker exec sentinel-victim-pod sh -c 'ps aux | grep -q "[s]shd -D"' && break
  sleep 1
done

echo "[4/4] LAN IP candidates for the attacker machine to target (use the Wi-Fi adapter's IPv4):"
if command -v ip >/dev/null; then ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.'; fi
cat <<MSG

Victim pod is listening on container port 22, host port 2222.
On WSL2, that host port is on the WSL VM's IP, not your Windows LAN IP directly - run the above from
Windows PowerShell instead: ipconfig | findstr IPv4
and make sure port 2222 is allowed through Windows Defender Firewall (see README §Demo).
Verify reachability from the attacker machine:  nmap -p 2222 <windows-lan-ip>
MSG
