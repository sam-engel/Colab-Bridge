---
name: colab
description: |
  Run code on Google Colab GPU runtimes via the `colab` CLI (the colab_bridge
  package). Use this skill whenever the user asks to provision, list, or release
  a Colab runtime; submit code to a Colab GPU; check, replay, follow, or cancel
  a job that was previously submitted via `colab`; or whenever the user mentions
  "the bridge", "the colab bridge", a `colab --…` command, or a Colab GPU type
  like T4 / L4 / A100 / H100. Also use this skill for any task involving the
  colab_bridge directory or .direct_kernel_jobs/ state files.
---

# colab — operational reference

`colab` is a console script registered by the `colab_bridge` Python package
(source: `colab_bridge/direct_kernel.py`). Install via `pip install
git+https://github.com/<user>/colab_bridge.git` into the env you use for the
project; if you want it callable from any shell, symlink `$(which colab)` into
`~/.local/bin/`. It talks to the same private API the official VS Code Colab
extension uses, and reaches real Colab GPUs (CPU/T4/L4/A100/H100) without a
browser, tunnel, or notebook UI.

---

## CLI commands

Runtimes:

| Command | Purpose | Example |
|---|---|---|
| `colab --auth` | One-time OAuth (browser pops once) | `colab --auth` |
| `colab --assign -a TYPE [-d DESC]` | Provision a new runtime; prints endpoint on stdout | `colab --assign -a A100 -d 'sweep'` |
| `colab --runtimes` (alias `--list`) | Active runtimes table (`SH ENDPOINT ACCEL SHAPE REGION TIMEOUT DESC`) | `colab --runtimes --all` |
| `colab --cost` | Live balance + per-runtime estimate + observed burn rate (CPU ~0.1 u/h, T4 ~1.84, L4 ~2.85, A100 ~5.50, H100 ~8) | `colab --cost` |
| `colab --unassign QUERY` | Release a runtime (letter/accelerator/`all`/prefix/substring) | `colab --unassign a` |
| `colab --set-timeout QUERY MINS` | Change idle timeout mid-flight (`0` disables) | `colab --set-timeout a 90` |

Jobs:

| Command | Purpose | Example |
|---|---|---|
| `colab [-r R] -c "CODE"` | Foreground job; output streams to terminal | `colab -r a -c "import torch; print(torch.cuda.is_available())"` |
| `colab [-r R] -f FILE` | Foreground job from file | `colab -r a -f train.py` |
| `colab --no-stream -c "..."` | Detached job; prints JID + watcher PID and returns | `JID=$(colab --no-stream -f long.py)` |
| `colab --jobs [QUERY]` | List jobs (defaults to active runtimes; `--all` for full history) | `colab --jobs A100` |
| `colab --job-id JID` (alias `--follow JID`) | Replay + tail one job to completion | `colab --follow $JID` |
| `colab --latest` | Replay + tail the newest job | `colab --latest` |
| `colab --status JID` | Print job metadata + connection status (connected / stale / disconnected / queued) | `colab --status $JID` |
| `colab --cancel JID` | Cancel a job. Queued (kernel hasn't started it) → just removes from queue. Running → sends `KeyboardInterrupt` | `colab --cancel $JID` |
| `colab --reattach JID [--no-stream]` | Reattach to a job whose original watcher died; captures NEW stdout/stderr/error events into the same `events.jsonl` until the kernel goes idle. Lost output (between disconnect and reattach) is unrecoverable | `colab --reattach $JID` |
| `colab --wait-for-job JID` | Block until done; exit 0/1/2 = done/error/unknown | `colab --wait-for-job $JID` |
| `colab --replay JID` | Print stored events without tailing | `colab --replay $JID` |

Live views:

| Command | Purpose | Example |
|---|---|---|
| `colab --live` | Full-screen alt-screen dashboard (runtimes + jobs + recent output). Keys: `o`/SPACE overview, `r`/`l` runtimes, `j` jobs, `c` cost, `f` follow, `e` events, `a` toggle "all" on jobs/runtimes, `↑`/`↓`/PgUp/PgDn/Home/End scroll, `/` open palette, `q` quit. Palette commands: `/help`, `/jobs [all]`, `/runtimes [all]`, `/follow [runtime]`, `/latest`, `/status JID`, `/cancel JID`, `/reattach JID`, `/run RT CODE`, `/release RT`, `/timeout RT MIN`, `/desc RT/JID TXT`, `/balance`, `/keepalive`, `/log`, `/env`, `/clear`. Tab completes; arrow keys scroll panel when /help or other long output is shown | `colab --live` |
| `colab --follow [QUERY]` | Multiplex EVERY running job (or filter by runtime) | `colab --follow A100` |
| `colab --events [--type T] [--jid J] [--from-start] [--once]` | Tail `events.jsonl` (machine-readable) | `colab --events --type job_error` |

`-r RUNTIME` and all QUERY arguments use the same resolver: `all` →
accelerator (multi-match) → letter shorthand → exact endpoint → unique prefix
→ unique substring. Ambiguous prefix/substring is an error.

---

## Concepts

- **Runtimes** are Colab VMs. Each has an `ENDPOINT` (long string) and a
  letter shorthand (`a`, `b`, …) stable as long as the active runtime set
  doesn't change. Use the letter for typing, the endpoint for scripts.
- **Jobs** are queued submissions; serialized per-runtime. Always pass `-r`
  to target the runtime *you* provisioned, never an implicit one.
- **`-d "..."`** attaches a one-line description that surfaces in
  `--runtimes`, `--live`, and `--follow`. Use it on every `--assign`.
- **`.env` injection**: `colab_bridge/.env` (mode 0600) is loaded into
  `os.environ` before user code runs. Use `colab --env-set NAME` to add
  values; never put secrets in submitted code. Override per-job with
  `--no-inject-env` or `--inject-env NAME ...`.
- **Idle timeout** (default 30 min): a per-runtime reaper subprocess
  unassigns after the configured idle window. Override with
  `--idle-timeout MINS` at assign time, or `--set-timeout` later. `0` disables.
- **Persistent Jupyter-WS keepalive**: the reaper also opens a long-lived
  Jupyter WebSocket so Colab's scheduler sees `connections >= 1` and doesn't
  reclaim the runtime as idle. Diagnostic lines in `reaper.log` are prefixed
  `ws-keepalive:`. See `colab_bridge/README.md` for details.
- **gRPC keep-alive (the load-bearing fix for ~20-min A100 preemption)**:
  the reaper also POSTs the same `RuntimeService.KeepAliveAssignment` RPC
  the browser uses, every 60 s, to `colab.clients6.google.com`. **As long
  as this ping keeps firing once a minute, preempt is mostly defeated** —
  idle Pro+ A100s now routinely survive hours/days instead of dying at
  ~20 min. Server-side preemption still happens occasionally but is rare;
  checkpoint anyway. The dominant reason a kept-alive runtime dies is **the
  bridge stops pinging**: laptop sleep ≥20 min, reaper killed, machine
  rebooted — Colab reclaims the runtime ~20 min after the last successful
  ping. Overnight workarounds: `caffeinate -dis` to block sleep, or run the
  bridge on an always-on machine and target it remotely. See
  `colab_bridge/README.md` (*gRPC keep-alive* section) for full details.
- **Preemption-event reliability**: `runtime_released reason=preempted`
  events in `events.jsonl` use a strict double-check (`list_runtimes()`
  must show endpoint missing on TWO consecutive calls ≥60 s apart) before
  firing. False-positive rate is ~zero post-fix. Acting on the event
  without a separate `colab --runtimes` cross-check is now safe; the
  cross-check remains cheap insurance if you want belt-and-suspenders.
- **Connection status** (per-job, surfaced by `--status`, `--live`, the
  `colab --jobs` table, and palette tab-completion):
  - `connected` (●green): events written within last 60 s.
  - `stale`     (●yellow): no events for 60 s–5 min; cell may just be silent.
  - `disconnected` (●red): no events >5 min OR handler PID is dead → use
    `colab --reattach JID` to recover output.
  - `queued`    (●cyan): the watcher attached and the store says
    `running`, but the kernel is still busy with a prior cell so our
    request is sitting in the kernel's queue. `--cancel` on a queued job
    skips the interrupt and just removes it from the queue (the kernel
    keeps running whatever IT was on).

---

## State files

All under `colab_bridge/.direct_kernel_jobs/` (gitignored, same project tree
as the bridge):

| Path | Contents |
|---|---|
| `index.json` | Job registry (flock-protected JSON array). |
| `runtimes.json` | `{endpoint: {accelerator, assigned_at, idle_timeout_min, released_at, ...}}` metadata. |
| `<JID>.events.jsonl` | Append-only event log per job (stdout/stderr/result/error/timeout). |
| `<JID>.pid` | Handler PID; presence + alive PID = "this job has a live driver". |
| `.reap_<endpoint>.lock` | `fcntl` lock held by the live reaper. |
| `reaper.log` | Timestamped audit of every reaper + ws-keepalive decision. |
| `events.jsonl` | Cross-runtime event stream (`job_*`, `runtime_*`). |
| `balance.jsonl` | Compute-unit balance samples for `--cost`'s observed-rate. |
| `ws_coverage.log` | gRPC keep-alive + WS probe + resources poll diagnostics. View in `--live` via `/keepalive`. |

OAuth creds live in `colab_bridge/.direct_kernel_creds.json` (auto-refreshed).

---

## Common patterns

- **Submit a long job and walk away**:
  ```bash
  JID=$(colab --no-stream --accelerator A100 -d 'train' -f train.py)
  colab --follow $JID                 # later, anywhere
  ```
- **What's running right now**: `colab --live` (full-screen, multi-runtime).
- **Wait inside a script**: `colab --wait-for-job $JID` (blocks; exit code
  reflects status).
- **Why did my runtime disappear**: grep `colab_bridge/.direct_kernel_jobs/reaper.log`
  for the endpoint. Look for `REAP` (we unassigned it), `PREEMPTED` (Colab
  killed it server-side), or `ws-keepalive: PREEMPTED` (the new WS-thread's
  double-checked confirmation).

---

## Pitfalls

- **Don't tight-loop `colab --list`** or `colab --runtimes` — every call hits
  Colab's assignments API and `_resurrect_reapers`. Cache the output.
- **Always provision a fresh runtime** with `colab --assign` before submitting
  work; reusing implicit/shared runtimes queues your job behind whatever else
  is running.
- **`-d "DESC"` on every `--assign`** — it surfaces in every view and is the
  only way to keep concurrent runtimes straight.
- **Always `--unassign` when done** (or rely on idle-timeout) to stop billing.
- **`tqdm` interleaving**: when a job uses a `tqdm` progress bar, prefer
  `tqdm.write("...")` over `print("...")` for any inline messages — it
  cooperates with the bar's redraw and avoids the bar getting fragmented
  across multiple lines. Plain `print` works but looks ugly in `--follow` /
  `--live` recent-output.
- **Use `from tqdm import tqdm`, NOT `from tqdm.auto import tqdm`** —
  `tqdm.auto` falls back to ipywidgets in Colab and emits no stderr we can
  capture, so progress is invisible from the bridge.
- **Jupyter magics don't work** (`!pip`, `%matplotlib`). Use `subprocess`:
  ```python
  import subprocess, sys
  subprocess.run([sys.executable, "-m", "pip", "install", "pkg", "-q"], check=True)
  ```
- **Never `os._exit()` from inside a job** — it severs the kernel process and
  effectively unassigns the runtime (`/content/` and installed packages
  evaporate). To clear Python state, call `dk.restart_kernel()` from the host.
- **Cancel during native code** (CUDA op, hung syscall) may not honor the
  interrupt. Use `dk.force_interrupt()` (Python API) to kill + recreate the
  kernel.

---

## Don't run without confirming

- `colab --auth` — opens a browser; only useful on first install.
- `colab --unassign` — releases a runtime that may have unsaved state.
- `colab --cancel` — kills work in progress.
