# colab_bridge

Programmatic access to Google Colab runtimes from your terminal — no browser, no copy-paste, no `cloudflared` tunnel. The bridge talks directly to the same private API the official VS Code Colab extension uses, so you get real Colab GPUs (T4, L4, A100, H100) with first-class job streaming, queueing, cancellation, and persistence.

```
[your terminal / Claude Code]  ←── HTTPS + WebSocket ──→  [Colab GPU runtime]
       direct_kernel.py                                        Jupyter kernel
```

The single entry point is `colab_bridge/direct_kernel.py`.

---

## Install

`colab_bridge` is a normal Python package. **Install it into the same virtual env / conda env that your project uses** — not into your global Python — so the `colab` CLI and the code you run from it share one interpreter.

### Option A — conda env (recommended for ML workflows)

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install git+https://github.com/sam-engel/Colab-Bridge.git
# With GCS upload/download helpers:
pip install 'git+https://github.com/sam-engel/Colab-Bridge.git#egg=colab_bridge[gcs]'
```

### Option B — venv

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install git+https://github.com/sam-engel/Colab-Bridge.git
```

### Option C — editable install from a local clone (for hacking on `direct_kernel.py`)

```bash
git clone https://github.com/sam-engel/Colab-Bridge.git
cd Colab-Bridge
pip install -e .
# Or with GCS extras:
pip install -e '.[gcs]'
```

Any of the above registers a `colab` console-script in the env's `bin/` dir. It only works while that env is active. To make it callable from any shell without activating, symlink it onto your `PATH`:

```bash
# Run this while the env is active:
ln -sf "$(which colab)" ~/.local/bin/colab
```

(`~/.local/bin` is on most users' `PATH`. On macOS you may need to add it: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc`.)

### Authenticate once

```bash
colab --auth
```

`--auth` opens a browser for Google OAuth (use the account that has Colab Pro / Pro+, or a free account for CPU-only). Credentials are saved next to the installed package as `.direct_kernel_creds.json` (gitignored) and refreshed automatically.

> **Note on storage location.** The bridge stores OAuth creds, the throw-away notebook ID, the `.env` secret file, and per-job event logs **next to the installed `colab_bridge` module** (`Path(__file__).parent`). With `pip install -e .` that's your clone dir. With a non-editable `pip install` that's somewhere under `site-packages/colab_bridge/`. Editable installs are recommended if you want easy access to the state files — or for inspecting logs.

### Verify

```bash
colab --test-cpu
```

This provisions a free CPU runtime, runs a small `import torch` script, prints the result, and exits.

### Claude Code skill (optional)

This repo ships a Claude Code skill at [`skills/colab/SKILL.md`](skills/colab/SKILL.md). It teaches Claude / Claude Code how to drive the bridge end-to-end — when to assign vs. reuse a runtime, how to read `--live`, what `--reattach` does, the connection-status semantics, common pitfalls. To install it, copy the directory into your user-level skills dir:

```bash
mkdir -p ~/.claude/skills
cp -r skills/colab ~/.claude/skills/colab
```

After that, Claude Code will load the skill automatically and use the `colab` CLI when you ask it to "run X on a GPU", "follow that A100", etc.

---

## Common workflows

### Run a script on a GPU and stream output here

```bash
colab --accelerator T4 -c "import torch; print(torch.cuda.get_device_name(0))"
colab -a A100 -f experiments/train.py
```

`--accelerator` (alias `-a`) accepts `CPU`, `T4`, `L4`, `A100`, `H100`, `G4`.  Default: `CPU`.  (Run `--runtimes` against an account to see what your tier is eligible for.)  Add `--high-ram` for the high-memory machine shape.

stdout / stderr / `tqdm` bars all stream live to the terminal (and are persisted on disk — see below).

### Descriptions (`-d` / `--desc`)

Tag a runtime or a job with a one-line description. **Recommended but not required.** The description shows up everywhere the runtime / job appears: `--runtimes`, `--jobs`, `--cost`, `--live`, `--follow` banners + lifecycle lines. Truncated to 25 chars in narrow tables, full text retained on disk.

```bash
colab --assign -a A100 --desc "paired-activations training"
colab -r a -f train.py --desc "step 5000 eval — sanity"
colab --no-stream -r a -d "warm-restart from ckpt-2300" -f train.py
```

In Python:

```python
dk = ColabDirectKernel.connect(accelerator="A100")
dk.run("...", desc="hparam sweep, lr=3e-4")
dk.submit("...", desc="long ablation")
```

Without a description, columns just show `—` and lifecycle lines have nothing extra. Why bother: when you have multiple runtimes alive (`a`, `b`, `c`) and dozens of jobs in history, the description is what tells you which experiment / branch / sweep you were running. Otherwise you're squinting at endpoint hashes and code previews trying to remember what was what.

### Multiple runtimes at once

Colab enforces 1:1 between a notebook and a runtime, so to keep N runtimes alive simultaneously the bridge owns N notebooks. The mechanics are automatic:

```bash
colab --assign -a A100              # 1st runtime — uses the bridge's primary notebook
colab --assign -a T4                # 2nd runtime — bridge sees primary is bound,
                                    #   creates a fresh throwaway notebook in Drive,
                                    #   provisions a T4 against it, prints the endpoint
colab --assign -a CPU               # 3rd runtime — another fresh throwaway notebook
colab --runtimes                    # all three show up with letter shorthands a/b/c
```

Caveats:

- Each fresh `--assign` against a bound primary creates one new `direct_kernel_runtime_pool` `.ipynb` file in your Drive root (~300 bytes, gitignored, harmless). The bridge doesn't try to recycle them — Colab eventually cleans up unused ones, or you can delete them by hand.
- Once a runtime is provisioned, the bridge talks to it via its endpoint (not its notebook), so submit / cancel / unassign / follow all work uniformly across pool and primary.
- The "primary" notebook (`.direct_kernel_notebook_id`) is **only** used by the implicit no-flag flow described below.

### Targeting a specific runtime (recommended when you have more than one alive)

```bash
colab --assign -a A100              # provision an A100, exit (prints endpoint)
colab -r a -c "..."                 # submit to runtime with shorthand letter 'a'
colab -r A100 -c "..."              # submit to *the* A100 (errors if more than one)
colab -r gpu-a100 -c "..."          # …or by endpoint prefix / substring
colab --no-stream -r a -f train.py  # detach + send job to runtime 'a'
```

Without `-r`, the bridge does an implicit `assign_runtime` via its **primary** notebook (`.direct_kernel_notebook_id`).  Colab tells us whether *that notebook* is already bound to a runtime:

- **Bound** → the bridge silently reuses it (no new VM, regardless of what `--accelerator` you pass).
- **Not bound** → the bridge provisions a fresh runtime of the requested `--accelerator` type **on the primary notebook** (does NOT auto-create a pool entry — that's `--assign`'s job).

That means **other runtimes on your account that aren't tied to the bridge's notebook (web-Colab tabs, other notebooks, runtimes from older bridge sessions) are NOT auto-picked**.  When you have multiple runtimes alive at once, always use `-r RUNTIME` to make sure the job lands where you expect.  Use `colab --runtimes` to see the letter shorthands.

> **Etiquette: don't submit to a runtime you didn't provision yourself.**  Jobs are serialized per kernel — your `-r a -c "..."` will queue *behind* whatever's currently running on runtime `a`, which is a problem if that job takes hours or if someone else is watching live output.  Before targeting an existing runtime, run `colab --jobs RUNTIME` and check for `running` entries.  If there's an active job (or you didn't start the runtime), provision a fresh one with `colab --assign -a TYPE` and use that instead.

### Submit a long job and walk away

```bash
JID=$(colab --no-stream --accelerator A100 -f experiments/long_train.py)
echo "submitted as $JID"
```

`--no-stream` forks a detached watcher process that holds the kernel WebSocket open, persists every event, and exits when the kernel finishes. The CLI returns immediately with the `job_id` on stdout. The job survives if you close the terminal, log out, or your laptop sleeps.

Reattach later (from any terminal, any time, replays full history):

```bash
colab --job-id $JID
colab --latest          # newest job, whichever runtime
```

### `--live` and `--events` — at-a-glance dashboards

Two interactive Ctrl+C-to-exit views, both reading from the same event log:

```bash
colab --live          # full-screen dashboard, redraws every ~1s
colab --events        # human-readable scrolling log of every event
```

`colab --live` clears the screen and shows a single live frame:

```
●  colab live dashboard — Ctrl+C to exit  (17:32:41)

  Balance: 568.68 units    Burn (est): 11.00 u/h    Observed: +1.95 u/h    Spent so far: 2.41 u

  Active runtimes (2):                                                      Open jobs (2):
  SH  ENDPOINT                ACCEL  UPTIME  RATE    COST  DESC             SH  JOB_ID    STATUS    AGE     DESC          CODE
  a   gpu-a100-s-kkb-ass1c1-… A100   0.18h   5.50/h  1.00u  hparam sweep     a   af189312  running   5.5m    sweep step    import torch; …
  b   gpu-a100-s-kkb-usw4b1-… A100   0.26h   5.50/h  1.41u  ablation         b   a9577edc  running   13.8m   ablation v2   """ Experiment 07 …

  Recent output (last 2):
  [a  af189312]  [sig] === C2 steered (α=3 w=5 t=0.5) ===
  [b  a9577edc]  Epoch 4/10  ▏100%|██████████|

  Events (last 2):
  17:32:34  job_done            abc12345  elapsed=42.0s
  17:36:00  runtime_released    reason=user
```

Runtimes and jobs are rendered **side-by-side** to keep the dashboard short.  Recent output and events default to **last 2 lines each** in overview; switch to focused `f` or `e` (see below) for ~40-50.  The whole thing runs in **alternate-screen mode** (like `top` / `vim`) — scrolling inside is contained and on exit your previous shell scrollback is restored intact.

Letters are color-keyed the same way as `--follow` (deterministic per-letter palette), so a runtime keeps the same color across all three views.  The "Recent output" tail is a 6-line global ring buffer — each tick we read new bytes from each open job's `events.jsonl` and append text lines (split on `\n` / `\r`).  First render seeds near end-of-file so we don't dump megabytes of history.  Balance is sampled at most once a minute; "Spent so far" is the sum of per-runtime estimated cost across the currently-active set, not a true running total.

**Keyboard navigation** (single-key, no Enter required):

| Key | Action |
|---|---|
| `o` / Space | overview (default) — runtimes + jobs + recent output + events |
| `r` / `l` | runtimes only (full ENDPOINT + TIMEOUT + RATE/COST columns) |
| `j` | open jobs only (full DESC + flattened CODE preview) |
| `c` | cost breakdown |
| `f` | follow (recent output, larger window) |
| `e` | events log |
| `a` | (jobs / runtimes views only) toggle "all" — include terminal-status jobs / released runtimes |
| `↑` / `↓` / PgUp / PgDn / Home / End | scroll within any non-overview view (and inside long palette output like `/help`). For `f` / `e` (tail-style), reaching the bottom re-arms tail mode |
| `/` | command palette (see below) — type a command, Enter to run, Esc to cancel; arrow keys scroll the palette overlay (long output) and don't dismiss it |
| `q` / Ctrl-C | quit (terminal mode + cursor restored) |

ESC is intentionally a no-op at the dashboard level (not Esc-to-overview): occasionally arrow-key escape sequences race the lone-ESC detector, and a misdetect was yanking the user back to overview mid-scroll. Use `o` or Space for overview.

**Pinned footer + fixed body.**  The key-hint bar lives at a fixed row (computed from terminal height) and does **not** move when you switch views or as data sizes change.  The body above it is clipped + padded to a constant number of rows: in overview, runtimes / jobs / output / events each get exactly 2 data rows (blank space if there's less data).  Focused `f` / `e` show the *last* N output / event lines that fit.  View switches redraw immediately on keypress using the already-fetched data — they don't wait for the next data refresh — so navigation feels snappy even though the underlying state is only re-polled every ~1 s.

**Command palette (`/`).**  Press `/` and the dashboard turns the line below the footer into a prompt; as you type, an autocomplete list (matching command names + their parameter signatures) appears below.  **Tab** completes to the longest unambiguous prefix (or fills in the full name + a space if there's only one match).  **Enter** runs, **Esc** cancels.  Result lines stay on screen until you press any non-`/` key (with a 5-minute safety fade if you wander off).

| Command | Effect |
|---|---|
| `/help` | list commands (alias `/h`, `/?`) |
| `/jobs [all]` / `/runtimes [all]` / `/list [all]` / `/cost` / `/follow [runtime]` / `/latest` / `/events` / `/overview` | switch the dashboard view (with optional argument). `/follow a` scopes the recent-output stream to runtime `a`. `/latest` jumps to follow view scoped to the newest running job. |
| `/cancel <jid>` | cancel a job. Queued jobs (kernel hasn't picked them up) are removed from the queue with no kernel interrupt; running jobs get a graceful `KeyboardInterrupt` |
| `/reattach <jid>` | spawn a detached reattach daemon for a job whose original watcher died (captures NEW output to the same `events.jsonl`) |
| `/release <runtime>` | unassign a runtime (alias `/unassign`); `<runtime>` accepts letter / accelerator / endpoint prefix / `all` |
| `/timeout <runtime> <min>` | change idle-timeout (`0` disables); reaper picks it up within ~30 s (alias `/set-timeout`) |
| `/desc <rt\|jid> <text>` | set / overwrite the description on a runtime or a job |
| `/assign <accel> [desc]` | spawn a detached `colab --assign -a ACCEL` in the background; new runtime appears in the runtimes view in 30-60 s |
| `/run <runtime> <code>` | spawn a detached `colab --no-stream -r RUNTIME -c CODE` job (alias `/submit`) |
| `/status <jid>` | print the full metadata + connection status for one job |
| `/balance` | force-fetch and print current paid-compute balance |
| `/env list \| set NAME=VAL \| rm NAME \| show NAME` | manage `colab_bridge/.env` from the dashboard.  Multi-line values are not supported in the palette — use `colab --env-set NAME --from-file PATH` for those. |
| `/log [n]` | dump the last `n` lines of `reaper.log` (default 20) |
| `/keepalive [n]` | dump the last `n` lines of `ws_coverage.log` (gRPC keep-alive + WS probe + resources poll diagnostics) |
| `/clear` | clear the recent-output and events ring buffers |
| `/refresh` | force an immediate data re-fetch (otherwise polled every ~1 s) |
| `/notebook-url` | print the bridge's Drive-notebook URL (alias `/url`) |
| `/quit` | exit the dashboard (alias `/q`, `/exit`) |

The palette uses the same `<runtime>` shorthand resolver as `--unassign` / `--follow` / `--set-timeout`, so `a`, `A100`, an endpoint prefix, or `all` all work.  HTTP-bound commands (`/cancel`, `/release`, `/timeout`) run synchronously and pause the dashboard for ~1-2 s; `/assign` and `/run` detach a subprocess and return instantly.  After any command runs, the outer loop refetches state immediately so you see the effect (released runtime disappearing, cancelled job flipping to `cancelled`, etc.).

**Connection-status indicator.** Each open job in the dashboard has a colored dot in the leftmost column reflecting how well its local watcher is hearing the kernel:

- **● green — connected**: events written to `<JID>.events.jsonl` within the last 60 s.
- **● yellow — stale**: events 60 s–5 min old; cell may just be silent.
- **● red — disconnected**: events >5 min old, OR the handler PID is dead. Job is still running on Colab but no one is capturing output locally → `colab --reattach JID`.
- **● cyan — queued**: the watcher attached and the store says `running`, but the kernel is still busy with a prior cell. The displayed status flips to "queued" everywhere. `colab --cancel` on a queued job removes it from the queue without sending a kernel interrupt (which would kill whatever IS executing — almost certainly your prior job, not this one).

The indicator surfaces in `--live` (left column of the jobs table), `colab --status JID`, `/status JID` in the palette, and the autocomplete entries when you tab-complete a JID for `/cancel`/`/status`/`/reattach`.

**Scrollable views.** Every non-overview view (`runtimes`, `jobs`, `cost`, `follow`, `events`) supports `↑`/`↓`/PgUp/PgDn/Home/End. The view's title + column header stay pinned; only the body items scroll. When content overflows the visible window, a `[↑ N above · ↓ M below]` indicator appears at the last body row. For `f` (follow) and `e` (events), reaching the bottom re-arms tail mode so new events auto-appear; scrolling up pauses tailing until you scroll back down. Long palette output (e.g. `/help` on a small terminal) supports the same scroll keys, so you don't lose access to the full text. Auto-repeat with held keys works at a fixed visible rate (renders are coalesced when more keystrokes are queued).

**Stable runtime letters.** Once a runtime is assigned, its letter (`a`, `b`, …) is stamped into `runtimes.json` and stays put for the runtime's entire lifetime — even when other runtimes are assigned or released around it. After a runtime is released, its letter slot becomes available for the next new one. So `colab -r b -f train.py` always points to the same runtime as long as `b` is alive.

`colab --events` is a scrolling, color-coded, human-readable feed of every lifecycle event as it happens:

```
17:32:34  ++  runtime assigned   gpu-a100-s-kkb-ass1c1-…  (A100, us-central1)
17:32:36  +   job queued         abc12345  on gpu-a100-…  (A100)
17:32:36  ▶   job started        abc12345  (A100)
17:35:18  ✔   job done           abc12345  elapsed=162.4s
17:36:00  ✗   runtime released   gpu-a100-…  reason=user
```

Same source as `--watch` but prettier — `--events` is for humans, `--watch` is for scripts (raw JSON).  Both only show **lifecycle events** (job start/finish, runtime assign/release); job stdout / stderr / progress bars do NOT appear here — those are per-job and live in `<job_id>.events.jsonl`, surfaced by `--follow` / `--job-id` / `--latest`.

### Alerts — be notified when something happens

```bash
colab --watch                                # tail every event, forever
colab --watch --type job_error               # only error events
colab --watch --type runtime_released        # only release events
colab --watch --jid abc12345                 # only events for one job
colab --watch --from-start                   # replay history first
colab --wait-for-job  abc12345               # block until job done; exit 0/1/2
colab --wait-for-runtime a                   # block until runtime is released
```

The bridge writes a structured event line to `colab_bridge/.direct_kernel_jobs/events.jsonl` at every state transition.  Event types: `job_queued`, `job_started`, `job_done`, `job_error`, `job_cancelled`, `job_timeout`, `runtime_assigned`, `runtime_released`.  Each line is one JSON object with `ts`, `type`, and type-specific fields like `jid`, `endpoint`, `accel`, `reason`, `elapsed_s`.

`--watch` tails the log (open file → `seek` → poll) and prints one JSON line per matching event.  `--once` exits 0 after the first match — useful for "ping me when this finishes" workflows:

```bash
JID=$(colab --no-stream -r a -f train.py)
colab --wait-for-job $JID && osascript -e 'display notification "Training done"'
```

For Claude Code / agent integration, run `colab --watch …` in a backgrounded `Bash` and attach `Monitor` — each new event line becomes a notification, no polling needed.

### Watch every runtime + every job, live

```bash
colab --follow                # multiplex ALL runtimes
colab --follow A100           # restrict to the A100 runtime
colab --follow gpu-t4         # restrict by endpoint prefix
colab --follow putvb          # or by any unique substring
```

`--follow` is a live multiplexer. It prints, in real time:

- **Per-line color is keyed off the runtime's letter shorthand** (`a` → green, `b` → yellow, …, deterministic), so all output from runtime `a` shares one color regardless of how many jobs each runtime has run. Stderr / errors render in red on top of the per-letter color.
- **`[SH/ACCEL JID]`** prefix on every output line — letter shorthand, accelerator, and job_id together so you can attribute each line to a runtime instantly.
- Job lifecycle events (`+ queued`, `▶ started`, `✔ done`, `✗ cancelled`, `✖ errored`) include the same `[SH/ACCEL JID]` tag so they sort visually with that runtime's output stream.
- **Multi-line attention banner** when a runtime gets released, with the reason (`preempted` / `idle_timeout` / `user` / `auto_reconcile` / `already_released`), endpoint, accelerator, region, and run-duration. Red for preempted, yellow for everything else.
- Cyan `●` system lines for context — currently-active runtimes, jobs in progress, the follow filter.

Output from multiple jobs running concurrently on different runtimes is **interleaved line-by-line**, never blocked by another job's stream. `\r`-style tqdm updates are promoted to newlines so they don't clobber the prefix. If you only want to follow one runtime, pass any unique fragment of the endpoint or just the accelerator name.

Want a single job? Use `--job-id JID` (full history + tail of one job, no multiplexing).

### List, cancel, manage jobs

```bash
colab --jobs                # jobs on currently-assigned runtimes (default)
colab --jobs --all          # full history — every job ever submitted
colab --jobs RUNTIME        # filter to one runtime (letter / accelerator / substring)
colab --jobs --json         # raw JSON
colab --status JID          # full metadata + connection status (one job)
colab --cancel JID          # cancel: queued → drop from queue; running → KeyboardInterrupt
colab --reattach JID        # reattach to a job whose original watcher died (foreground)
colab --reattach JID --no-stream    # …same, but as a detached daemon
```

`--jobs` shows `JOB_ID`, `STATUS`, `RUNTIME` (endpoint), `ACCEL`, when it started, elapsed time, and a one-line code preview — newest at the bottom.  By default it filters to jobs whose runtime is still alive (so the table reflects "what's actually relevant right now"); pass `--all` to see every job ever submitted, including those on long-released runtimes. The same job state file is read by `--latest`, `--job-id`, `--follow`, and `--cancel`, so multiple terminals stay consistent.

The `STATUS` column applies the same display override as `--live` and `--status`: when a job's store status is `running` but no events have been written past the 30-second warmup grace, it's shown as `queued` (the kernel is still busy with a prior cell, and our request is waiting in the kernel's queue). `colab --cancel` on such a job skips the kernel interrupt and just removes it from the queue.

`colab --reattach JID` opens a fresh Jupyter WebSocket to the kernel in observe-only mode (no `execute_request` is sent), captures stdout/stderr/result/error events into the same `events.jsonl` with a `_reattached: true` flag, tracks `execution_state` transitions, and marks the job `done` / `error` when the kernel goes idle. **Output emitted between the original disconnect and the reattach is unrecoverable** — Jupyter doesn't buffer for disconnected clients. With `--no-stream` it spawns a detached daemon (mirrors `--no-stream` for normal jobs).

### List and release runtimes

```bash
colab --runtimes              # pretty table — currently-active runtimes (alias: --list)
colab --runtimes --all        # also include released ones (history)
colab --runtimes --json       # raw API response
colab --unassign a            # release runtime with shorthand letter 'a'
colab --unassign A100         # release ALL A100 runtimes
colab --unassign gpu-a100     # endpoint prefix (must be unique)
colab --unassign putvb        # endpoint substring (must be unique)
colab --unassign all          # release every active runtime
```

`--runtimes` (alias `--list`) prints `SH  ENDPOINT  ACCEL  SHAPE  REGION  TIMEOUT` for every active runtime. The `SH` column gives each runtime a single-letter shorthand (`a`, `b`, …, then `aa`, `ab`, …) sorted by endpoint, stable as long as the active set doesn't change. The region is parsed out of the proxy URL (`us-central1`, `asia-southeast1`, etc.).

The same shorthand resolver is shared by `--unassign`, `--follow`, and `--set-timeout`. Resolution order:

1. **`all`** — every active runtime.
2. **Accelerator name** (`A100`, `T4`, `CPU`, …) — every runtime of that type. *Multi-match is OK here* — `colab --unassign A100` releases all A100s.
3. **Letter shorthand** (`a`, `b`, …) — exactly one runtime.
4. **Exact endpoint** — exactly one runtime.
5. **Endpoint prefix** — must be unique; otherwise the bridge prints the candidates and exits non-zero.
6. **Endpoint substring** — same uniqueness rule.

Multiple runtimes can be active at once (different regions / accelerators). Each job record carries the endpoint it ran on, so `--jobs` always tells you which runtime a job belongs to.

### Cost — credits, burn rate, per-runtime spend

```bash
colab --cost
```

Output:

```
Balance:  594.10 compute units remaining

SH  ENDPOINT                               ACCEL  UPTIME   RATE/h     EST COST
─────────────────────────────────────────────────────────────────────────────────
a   gpu-a100-s-kkb-…                       A100   0.25h    11.77      2.89 u
─────────────────────────────────────────────────────────────────────────────────
    TOTAL                                                  11.77      2.89 u

Estimated burn right now: 11.77 units/hour
Observed rate (from balance log): 11.50 units/hour  burning
```

What's where:

- **Balance** — live `paidComputeUnitsBalance` from `https://colab.pa.googleapis.com/v1/user-info`. The actual remaining-credits number on your account.
- **Per-runtime EST COST** — `(now - assigned_at) × rate[accelerator]`. The rate table (`_COST_RATE_TABLE` in `direct_kernel.py`) is hardcoded with approximate Pro/Pro+ rates: T4 ~1.84/h, L4 ~4.82/h, A100 ~11.77/h, H100 ~13.20/h, CPU ~0.1/h. **These rates may drift** when Google updates pricing.
- **Estimated burn right now** — sum of the above, in units/hour.
- **Observed rate** — actual rate measured from successive balance snapshots written to `colab_bridge/.direct_kernel_jobs/balance.jsonl`. Each `colab --cost` call appends a sample. After two samples >5 min apart, you get an "observed" rate that double-checks the static table. If the observed and estimated numbers disagree by more than a couple percent, the rate table needs updating.

### Idle-timeout auto-unassign + automatic keepalive

By default, every runtime gets a **30-minute idle timeout** — if no job has been queued or running on it for that long, the bridge unassigns it for you so you don't burn compute units on a forgotten VM.

```bash
colab --accelerator A100 --idle-timeout 60 -f train.py   # 60-min timeout for this runtime
colab --idle-timeout 0   --accelerator T4  -c "..."      # 0 = disable timeout
colab --set-timeout a 90                                 # change timeout later (mins)
colab --set-timeout A100 0                               # off for all A100s (Pro+ only)
```

The same per-runtime daemon that does the timeout also pings Colab's keep-alive endpoint every ~30s, so on Pro / Free where Colab disconnects after ~90 min idle, your runtime stays alive between jobs as long as you've set a non-zero idle-timeout.  In effect: **as long as the bridge plans to keep the runtime around, it also keeps it alive.**  On Pro+ this is moot (no idle disconnect) but cheap enough that it doesn't matter.

How it works:

- The countdown clock is **time since the most recent job on that runtime ended**.  Submitting a new job resets it.
- A small `--internal-reap ENDPOINT` subprocess is spawned the moment a runtime is assigned and again at the end of every job.  It holds an exclusive `flock` on a per-endpoint lock file, so there's never more than one reaper per runtime regardless of how many jobs end concurrently.
- The reaper sleeps in 30s slices.  Each slice it (a) hits `/tun/m/{endpoint}/keep-alive/` to defeat Colab's idle disconnect, (b) re-reads the timeout from disk so `colab --set-timeout` takes effect mid-flight, (c) checks whether a new job has started — if so, exits.
- If the runtime is already gone when the reaper wakes (e.g. you ran `--unassign`, or Colab disconnected first), the unassign call no-ops.
- The timeout is shown in the `TIMEOUT` column of `--runtimes` (`30m` / `off` / `—` if no metadata).

#### What about computer sleep and shutdown?

| Event | Reaper behavior |
|---|---|
| Computer goes to sleep | The reaper sleeps with the system.  When the computer wakes, the reaper wakes too; if the deadline passed during the sleep, it reaps on the spot.  In practice: a runtime you forgot about gets unassigned almost immediately on resume. |
| Computer shutdown / hard reboot | The reaper subprocess dies.  The Colab runtime keeps running until either (a) you next run **any** `colab` command — every CLI invocation calls `_resurrect_reapers()` at startup which respawns missing reapers (no-op if one is already alive), or (b) Colab's own idle timeout fires (90 min on Pro). |
| `colab --unassign ENDPOINT` | The runtime is released and `runtimes.json` is updated; reaper exits silently next slice. |

So a forgotten runtime is *eventually* cleaned up either by your next `colab` invocation or by Colab itself; there's no durable resource leak.

#### "My runtime just got unassigned and I didn't ask for it!"

If `colab --runtimes` shows your A100 (or whichever) is gone but you never ran `--unassign`, the question is whether the bridge reaped it or Colab killed it server-side. **Use `colab --log`** (or read `colab_bridge/.direct_kernel_jobs/reaper.log` directly) — every reaper decision (spawn, sleep, exit-because-active, REAP) gets a timestamped line:

```bash
colab --log              # last 200 lines (default)
colab --log 50           # last 50 lines
colab --log --all        # entire file
```

Sample line:

```
2026-04-27 03:31:15  pid=15499  ep=gpu-a100-…  countdown — timeout=240min  base=03:30:46  deadline=07:30:46  wait=14370s
```

If there's no `REAP` line for your endpoint, the bridge didn't unassign it. That means either (a) Colab killed it server-side, or (b) the bridge's keep-alive thread stopped firing long enough for Colab to reclaim it. With the gRPC keep-alive (see below) the second case is now the dominant reason — usually because the bridge's reaper subprocess was suspended (laptop sleep) or terminated (shutdown, reboot, killed by hand). True server-side termination (the 12 h / 24 h max-lifetime cap, an account-level hiccup, or capacity-pressure preemption that out-races the keep-alive) does still happen but is much rarer than it used to be.

Google does NOT publish termination reasons via any API or webhook (compare AWS Spot's 2-min warning + `termination-reason`, or GCP Preemptible's 30-second SIGTERM hook — Colab has nothing equivalent).  The bridge **infers preemption locally** via two independent signals:

1. **Keep-alive status code.** Every per-runtime ping to `/tun/m/{endpoint}/keep-alive/` checks for `HTTP 400 / 404 / 410`.  Empirically Colab returns **400** when the runtime is gone (confirmed by direct probing of multiple released endpoints — *not* 404 as you'd expect from REST-style "resource not found"). So we accept all three as "gone" signals.
2. **Periodic `list_runtimes()` cross-check (with double-check).** Every ~5 minutes the reaper hits the assignments API and verifies the endpoint is still present. A single missing observation is no longer enough to declare preemption — the reaper now runs `_is_preempted_double_check`, which requires the endpoint to be missing on **two consecutive `list_runtimes()` calls ≥60 s apart** before stamping `preempted`. This was added after a transient API blip caused one false-positive `runtime_released reason=preempted` event during testing on a runtime that was actually alive. The cost: when preemption IS real, declaration is delayed by 60 s. Worth it — `events.jsonl` consumers can now act on `runtime_released` events without needing their own cross-check.

Either signal triggers the same handling: a `PREEMPTED` line in `reaper.log` with a wall-clock timestamp + `runtimes.json[endpoint]` gets `{released_at, preempted: true}` stamped on it.  This won't bring the GPU back, but it gives you a precise correlation point for "when did my job die" — useful for matching up checkpoints, billing math, and any support tickets.

In practice, **the bridge mostly defeats Colab preemption** as long as its per-runtime gRPC keep-alive thread keeps firing once a minute — see the *gRPC keep-alive* subsection below for the load-bearing details. Unprompted preemption events are now rare; when they do happen they usually trace back to the bridge's reaper going quiet (laptop slept, machine rebooted, reaper killed) for ≥20 min, occasionally to a genuine server-side preemption that out-races the ping. Still: **checkpoint to `/content/` or GCS** (`from colab_bridge.gcs_agent import GCSAgent`) every few minutes on long runs — a 12 h / 24 h account-level cap, an API-key rotation, a process crash, or the occasional server-side hiccup can all end a session, and a checkpoint is much cheaper than a re-run.

When this happens the bridge eventually notices via `_resurrect_reapers()` (called at the top of every `colab` command).  That helper cross-checks `list_runtimes()`; if a metadata entry isn't in the live API result, it's stamped `released_at` and the reaper for it is NOT respawned.  This stops the zombie-reaper spiral that otherwise resulted in dozens of `--internal-reap` subprocesses each 404-ing on every unassign attempt.  The released entry remains visible under `colab --runtimes --all`.

The matching change in `unassign_runtime`: any 404 from the unassign API endpoint is now treated as "already released" (returns `status: already_released`, stamps `released_at`, does not raise).  So when you (or another tool) try to unassign a runtime Colab has already killed, it's a clean no-op rather than an exception.

#### Keepalive — on by default

Every foreground `colab -c "..."` / `colab -f train.py` and every detached `--no-stream` watcher starts a background keepalive thread that pings `/tun/m/{endpoint}/keep-alive/` every 60s for the duration of the job.  The thread is silent on success, prints a one-line stderr warning if it sees a 404/410 (preemption), and exits at job-end.  Pass `--no-keepalive` to disable for one job.

Why on by default: between jobs, the per-runtime reaper subprocess already pings; during a job, that reaper has exited (it sees the active job and returns).  Without the per-job thread there'd be a gap where Pro/Free's 90-min idle disconnect could fire if your job runs long but the kernel is mostly waiting on disk I/O.  The thread is cheap (one HTTP GET / minute) and harmless on Pro+ where the idle disconnect doesn't apply.

#### Persistent Jupyter WebSocket — the real keepalive

The HTTP `/tun/m/{ep}/keep-alive/` ping above is necessary but not sufficient.  Colab's server-side scheduler ignores it unless the runtime currently has at least one open Jupyter WebSocket (the extension's `keepServerAlive` short-circuits when `kernels.list[].connections == 0`).  Browser Colab keeps a WS open for the whole session and so always wins.  Our bridge originally opened a WS only for the duration of `_exec_sync`, leaving `connections == 0` between jobs — which Colab's scheduler treats as idle and reclaims aggressively (observed: Pro+ A100s preempted ~15-20 min after assign, vs. 90+ min in browser Colab).

To fix this, every reaper subprocess now also runs a daemon thread that holds a persistent Jupyter WebSocket open for the runtime's lifetime.  The thread sends a side-effect-free `kernel_info_request` every 25 s (under the typical 60 s tunnel-proxy idle timer), responds to inbound RFC 6455 ping frames with pongs, rotates the connection before the 3600 s proxy-token expiry, and reconnects on disconnect with `1, 2, 5, 15, 30, 60` s backoff.

While a job runs, `_exec_sync` opens a *separate* WS for its `execute_request` — total `connections == 2` during the job, falling back to `1` when the job ends.  Replies route by `parent_header.session_id`, so the keepalive's `kernel_info_reply` does not pollute the job's stdout/stderr.

Failure-mode handling:

| Event | Behavior |
|---|---|
| Inbound CLOSE / EOF / socket error | Silent reconnect with backoff. Not interpreted as preemption. |
| Handshake 4xx (token expiry, warmup) | Refresh OAuth token, retry once; otherwise reconnect with backoff. Not interpreted as preemption. |
| `_refresh_proxy_for_endpoint` returns `None` (endpoint missing from `list_runtimes()`) | Increment a counter; only on **two** consecutive misses **≥60 s apart** do we stamp `released_by="preempted"` and exit. This double-check exists because past versions of this code stamped preemption on a single 4xx during runtime warmup, corrupting `runtimes.json`. |
| `released_at` observed in metadata (set by reaper, `--unassign`, etc.) | Exit cleanly. |
| Reaper process exits | `ws_stop` event is set in the reaper's `finally`; the thread sees it and logs `ws-keepalive: stop_event set, exiting`. |

The keepalive thread runs automatically for every runtime that has a non-zero `idle_timeout_min` (i.e. wherever a reaper exists).  No new CLI flag.  Diagnostic lines in `colab_bridge/.direct_kernel_jobs/reaper.log` are prefixed with `ws-keepalive:` — search for `ws-keepalive: connected` to see when it attached, `ws-keepalive: PREEMPTED` to see confirmed preemption (distinct from the legacy `PREEMPTED` line emitted by the older HTTP-only path).

#### gRPC keep-alive — the actually-load-bearing fix for ~20-min A100 preemption

Despite the WS keepalive above, **Pro+ A100s were still being preempted at exactly ~20 min** after assign — independent of WS connection count, browser tab presence, Tailscale, time-of-day, or whether jobs were running. The root cause: the bridge's `/tun/m/{ep}/keep-alive/` ping was hitting an *older / auxiliary* keep-alive endpoint. Colab's preemption scheduler watches a different one.

A mitmproxy capture of a real browser Colab session revealed the actual keep-alive — a gRPC-Web RPC, fired every 60 s exactly:

```
POST https://colab.clients6.google.com/$rpc/google.internal.colab.v1.RuntimeService/KeepAliveAssignment
Body: ["{endpoint}"]      (JSON-protobuf)
Headers:
    content-type:    application/json+protobuf
    origin:          https://colab.research.google.com
    referer:         https://colab.research.google.com/
    x-goog-api-key:  AIzaSyA2BvntLwNwFthUB4w6_Bhn0cMlVHwyaHc
    x-goog-authuser: 0
    x-user-agent:    grpc-web-javascript/0.1
```

The bridge's existing OAuth bearer token authenticates this RPC fine — no cookie hackery needed. Implementation: `_grpc_keepalive_loop` runs as a daemon thread inside every reaper, calling this RPC every 60 s. Diagnostic lines: `grpc-keepalive #N: HTTP 200` in `ws_coverage.log`.

**Verdict: this mostly defeats Colab preemption.** As long as the gRPC keep-alive keeps firing once a minute, idle Pro+ A100s that used to die at exactly 20 min now routinely survive hours-to-days. Unprompted server-side preemption still happens occasionally — Colab's scheduler clearly has other inputs we don't understand — but it's rare enough that you can plan around it with checkpoints rather than expect it. The common reasons a "kept-alive" runtime dies are:

1. **The bridge stops pinging** — laptop sleep, reaper killed, machine rebooted. Colab reclaims the runtime ~20 min after the last successful ping. (Dominant cause.)
2. **Account-level 12 / 24 h max-lifetime cap.**
3. **Genuine capacity-pressure preemption** that out-races the keep-alive. Uncommon, but non-zero.
4. **You ran `colab --unassign`.**

So the practical rule is: **keep something pinging, and still checkpoint.** The bridge does the pinging automatically while the reaper subprocess is alive on a host that isn't suspended. If you need a runtime to survive overnight on a laptop:

- `caffeinate -dis` (macOS) before walking away, OR
- run the bridge on an always-on machine (small VPS / Raspberry Pi / desktop with sleep disabled) and target it from any client.

Once a runtime stops getting pings, Colab reclaims it ~20 min after the last successful ping. That's not a bridge bug, it's the literal meaning of "as long as something is pinging".

Caveat — **API key drift**: the API key above is the public Colab client key shared with the VS Code extension and the browser. If Google rotates it, this fix breaks. Re-capture via mitmproxy and update `_GRPC_KEEPALIVE_API_KEY` in `direct_kernel.py`.

---

## Python API

```python
from colab_bridge.direct_kernel import ColabDirectKernel

dk = ColabDirectKernel.connect(accelerator="A100")
dk.start_keepalive()                        # ping the runtime so it doesn't idle out

# Blocking
dk.run("import torch; print(torch.cuda.get_device_name(0))")

# Non-blocking (returns job_id; persisted to disk just like the CLI)
jid = dk.submit("long_training_loop()")
for ev in dk.stream(jid):
    print(ev["text"], end="")

# Interrupt
dk.interrupt()             # graceful KeyboardInterrupt
dk.force_interrupt()       # kill kernel + recreate (use when frozen)
dk.restart_kernel()        # clears Python state; /content/ files survive

# Release the runtime
dk.unassign()
```

`dk.submit()` writes to the same `.direct_kernel_jobs/` registry the CLI uses, so Python-submitted jobs show up in `--jobs` / `--latest` / `--follow` from your shell.

### Shared kernel namespace

Every job submitted to the same kernel runs in one shared `globals()` dict, exactly like a Jupyter kernel. Variables, imports, model weights persist across calls until `restart_kernel()`:

```python
dk.run("import torch; model = MyModel().cuda()")
dk.run("train(model, epochs=10)")    # `model` is still there
dk.run("evaluate(model)")
```

### Colab Secrets

Colab's "Secrets" feature (the side panel where you store `HF_TOKEN`, `OPENAI_API_KEY`, etc., scoped to your Google account) accessed via `from google.colab import userdata; userdata.get(name)`.

**Secrets do not work from this bridge.**  Confirmed empirically: with "Notebook access" toggled ON for the bridge's notebook, `userdata.get()` from a bridged kernel still returns:

```
TimeoutException: Requesting secret HF_TOKEN timed out.
                  Secrets can only be fetched when running from the Colab UI.
```

This is a hard server-side gate — Colab requires its own UI client to mediate the metadata-server hand-off; the bridge's bare WebSocket isn't recognized as such.  The probe lives at `colab --secrets` for verification (we confirmed it stays an `ERROR` even with toggles on):

```bash
colab --secrets                    # checks a default list (HF_TOKEN, GITHUB_TOKEN, …)
colab --secrets MY_SECRET          # check a specific name
colab --notebook-url               # print the bridge notebook URL
colab --repair-notebook            # if the notebook URL says "corrupted"
```

### Local secrets (the actual workaround)

Since Colab's own secret store is unreachable, the bridge ships its own: a local `colab_bridge/.env` file that's read at job-submit time and injected into the runtime as `os.environ` entries before your code runs.

**Storage**: `colab_bridge/.env`, mode `0600` (gitignored). Standard `KEY=VALUE` per line; quoting handled automatically; **multi-line values are supported** for things like service-account JSON keys.

**Manage secrets**:

```bash
colab --env                          # list NAMEs only (values not shown)
colab --env-set HF_TOKEN              # interactive prompt (hidden input)
colab --env-set HF_TOKEN=hf_xxx       # direct (note: lands in shell history)
colab --env-set gcs_creds --from-file path/to/key.json   # multi-line / JSON
colab --env-show HF_TOKEN             # print one value (for scripts)
colab --env-rm HF_TOKEN
```

**Inject behavior** at job submit time:

By default every key in `.env` is injected.  Per-job overrides:

```bash
colab --no-inject-env -c "..."                    # inject nothing
colab --inject-env HF_TOKEN -c "..."              # inject only HF_TOKEN
colab --inject-env HF_TOKEN GITHUB_TOKEN -f train.py
```

In your job code, just read from the env:

```python
import os, json
hf  = os.environ["HF_TOKEN"]
gcs = json.loads(os.environ["gcs_creds"])   # multi-line JSON parses cleanly
```

For GCS specifically, prefer the helper rather than re-parsing the JSON yourself (requires the `[gcs]` extra or a local `pip install google-cloud-storage`):

```python
from colab_bridge.gcs_agent import GCSAgent
gcs = GCSAgent.from_env("my-bucket")               # reads os.environ['gcs_creds']
gcs.upload("local/dir/", "experiments/run-42/")
```

`GCSAgent.from_colab_secrets()` is kept as a deprecated alias (falls back to
`from_env`) — **don't write new code that calls it**. The Colab Secrets API
(`google.colab.userdata`) doesn't work via the bridge.

**Python API** uses the same machinery:

```python
dk.run("import os; print(os.environ['HF_TOKEN'])")     # injects by default
dk.run(code, inject_env=False)                          # skip
dk.run(code, inject_env=["HF_TOKEN"])                   # filter
```

**On-disk hygiene** — what's where:

- The values themselves live ONLY in `colab_bridge/.env` (mode 0600).
- `index.json` (the job registry) stores your code verbatim and an `inject_env` flag (`true` / `false` / list of names) — **never the secret values**.
- `<job_id>.events.jsonl` only captures kernel output.  If your code itself prints `os.environ`, the values land here — your responsibility, not the bridge's.
- The injection prelude wraps in `try/except` that surfaces only the exception type, never values, so a malformed entry can't leak via traceback.
- The prelude flies over TLS to Colab as part of the `execute_request`.  We have no visibility into what Colab logs server-side.

**Useful even though Colab Secrets don't work**: `--notebook-url` prints `https://colab.research.google.com/drive/{id}` so you can open the kernel's runtime, browse `/content/`, etc. in Colab's UI.

### State lifecycle: what survives what

There are four very different ways a session can end, with very different consequences. The terminology matters: "disconnect" gets used loosely but the bridge has three distinct operations.

| Action | In-memory Python state | `/content/` files | Installed pip packages | GPU / runtime |
|---|---|---|---|---|
| `dk.run(...)` again | preserved | preserved | preserved | preserved |
| CLI process exits (no `--unassign`) | **preserved** — the kernel keeps running on Colab. Reattach with `colab --latest` / `--job-id`. | preserved | preserved | preserved |
| `dk.restart_kernel()` | **LOST** | preserved | preserved | preserved (same runtime, same kernel ID) |
| `dk.force_interrupt()` | **LOST** | preserved | preserved | preserved (same runtime, **new** kernel ID) |
| `dk.unassign()` / `colab --unassign` | **LOST** | **LOST** | **LOST** | **RELEASED** — must `assign_runtime` again |
| ~90-min idle timeout | **LOST** | **LOST** | **LOST** | **RELEASED** — same as unassign |
| `colab --cancel JID` | preserved (just sends `KeyboardInterrupt`) | preserved | preserved | preserved |

**Key implications:**

- **Save checkpoints to `/content/`** if there's any chance you'll need to `restart_kernel()` (or if a job might crash). In-memory tensors and dataclasses are gone after restart.
- **`pip install pkg` survives `restart_kernel()`.** The package files live under `/usr/local/lib/...` which is part of the runtime filesystem, just like `/content/`. So you can install once, restart to clear bad state, and `import pkg` still works.
- **Nothing survives `unassign()`.** Treat it as throwing away the VM. Always re-`pip install` and re-copy data after a fresh assign.
- **CLI exit is harmless.** The kernel runs on Colab, not your laptop. Closing your terminal, killing the CLI, or rebooting your laptop all leave the job running. That's the whole point of `--no-stream`.

### Restart vs. force-interrupt vs. unassign — when to use which

- **`dk.restart_kernel()`** — clears Python state but keeps the runtime. Use when imports got into a weird state, or you want to re-run with fresh globals. Variables of `/content/` files are NOT touched.
- **`dk.force_interrupt()`** — kills the kernel process and creates a new one on the same runtime. Use when `dk.interrupt()` doesn't return (frozen in a CUDA op, hung syscall, deadlock). Same effect as `restart_kernel()` from the user's perspective; the difference is only that `force_interrupt` is the right verb when you need to abort a stuck job.
- **`dk.unassign()`** — release the whole runtime back to Colab. Use when you're done and want to free up compute units. Anything not saved to your laptop is gone.

### Don't kill the kernel from inside a job

```python
# DO NOT DO THIS inside `dk.run(...)`:
import os; os.kill(os.getpid(), 9)
import os; os._exit(0)
```

The Colab Jupyter kernel is the same OS process as the gRPC bridge to the proxy. Killing it severs the runtime cleanly enough that the next request reassigns a fresh VM — meaning you've effectively done an `unassign()` (everything in `/content/` is lost). To restart Python state, call `dk.restart_kernel()` from the host instead — that's the safe, surgical version.

### Installing packages mid-session

Jupyter magics (`!pip`, `%matplotlib`) don't work — use `subprocess`:

```python
dk.run("""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'somepkg', '-q'], check=True)
import somepkg
""")
```

For editable installs (`pip install -e`), don't try to force a kernel restart from inside `exec()` — patch `sys.path` directly:

```python
dk.run("""
import subprocess, sys, importlib
subprocess.run('pip install -q -e /content/myproject', shell=True, check=True)
for p in ['/content/myproject/src']:
    if p not in sys.path: sys.path.insert(0, p)
importlib.invalidate_caches()
""")
```

Or just call `dk.restart_kernel()` from the host — `/content/` survives the restart.

---

## Where job state lives (there is no server on Colab)

The old `colab_bridge` ran a FastAPI server inside the Colab notebook that owned the job queue and exposed `/jobs`, `/cancel/{id}`, etc.  This rewrite **runs nothing on Colab** — the runtime is just a stock Jupyter kernel and the bridge talks to it over the same WebSocket every Jupyter client uses.  The only thing that knows about "jobs" or "runtime metadata" is the local CLI, which writes everything to disk under one directory.

### Storage layout

```
colab_bridge/
├── .direct_kernel_creds.json           OAuth tokens (with expires_at).
├── .direct_kernel_notebook_id          Drive file ID of the throw-away notebook.
└── .direct_kernel_jobs/
    ├── index.json                      Job registry — every job ever submitted.
    ├── runtimes.json                   Per-runtime metadata (idle_timeout_min,
    │                                   assigned_at, accelerator).
    ├── <job_id>.events.jsonl           Line-delimited events for one job
    │                                   (stdout, stderr, results, errors, status).
    ├── <job_id>.events.jsonl
    ├── …
    └── .reap_<endpoint>.lock           Per-runtime fcntl lock — held by the
                                        live reaper subprocess.  Empty file.
```

Everything is gitignored.

### What's in each file

**`index.json`** is a JSON array of job records.  One record per job, ever.  Schema:

```json
{
  "job_id":      "abc12345",
  "status":      "queued | running | done | error | cancelled",
  "endpoint":    "gpu-a100-s-kkb-…",
  "accelerator": "A100",
  "jupyter_url": "https://8080-…-c.us-central1-1.prod.colab.dev",
  "kernel_id":   "31da18a3-…",
  "code":        "<full source of the submitted job>",
  "started":     1761521234.5,           // unix seconds
  "ended":       1761521267.9            // null while running
}
```

`flock`-protected on every write (CLI invocations + watcher subprocesses + the foreground job all touch it concurrently).

**`<job_id>.events.jsonl`** is line-delimited JSON, one event per line.  Each event is one of:

```json
{"type": "stdout",  "text": "…"}             // print(...)
{"type": "stderr",  "text": "…"}             // sys.stderr writes; tqdm bars
{"type": "result",  "text": "42\n", "data": {"text/plain": "42"}}
                                              // execute_result / display_data
{"type": "error",   "text": "<traceback>", "ename": "ValueError", "evalue": "..."}
{"type": "timeout", "text": "[direct_kernel] timed out after 3600s\n"}
```

Append-only, no locking needed for readers (`--job-id` / `--latest` use `seek` + `tail`).  `colab --jobs --json` reads `index.json` directly; `colab --job-id JID` does `tail -F`-style replay over the events file.

**`runtimes.json`** is a JSON object keyed by endpoint.  Schema:

```json
{
  "gpu-a100-s-kkb-…": {
    "accelerator":      "A100",
    "assigned_at":      1761520000.0,
    "idle_timeout_min": 30
  }
}
```

Used by the reaper to decide when to unassign + to render the `TIMEOUT` column of `--runtimes`.  Removed when a runtime is unassigned.

### Implications

- The Colab kernel has no notion of `job_id`.  It's a client-side identifier; on the kernel it's just the most recent `execute_request` `msg_id`.
- **Two laptops sharing the same Colab account have separate registries.**  They both can attach to the same kernel (the `kernel_id` is a Colab/Jupyter concept and is shared), but `colab --jobs` on machine A doesn't see jobs submitted from machine B.
- **If the watcher subprocess for a `--no-stream` job dies hard** (computer crashes, OOM-kill, etc.), the Colab kernel keeps running but `events.jsonl` stops being appended to.  There's no way to recover the missed output — the bridge isn't listening anymore.  The kernel itself is fine; future `dk.run(...)` calls continue to work.
- **Deleting `.direct_kernel_jobs/` doesn't affect anything running on Colab.**  It just wipes your local history.  Runtimes (visible via `colab --runtimes`) are unaffected.  Reaper lock files in there get re-created on the next assign.
- **`--no-stream` exists because of all this**: someone has to hold the WebSocket open and append to `events.jsonl`.  Without a watcher, output is just discarded by the kernel after the WS closes (Jupyter doesn't buffer).
- The job code is stored verbatim in `index.json`, so don't put plaintext secrets in your source — pass them via `os.environ` instead.

---

## How it works

`direct_kernel.py` reverse-engineers the Colab API the official `google.colab` VS Code extension uses (`~/.vscode/extensions/google.colab-*/out/extension.js`):

1. **OAuth**: standard installed-app flow with the VS Code extension's own client ID, saved to a JSON file with `expires_at` so subsequent runs never make a `tokeninfo` round-trip.
2. **Notebook**: lazily creates one Colab notebook on Drive and reuses it for every runtime assignment.
3. **Assign**: `GET /tun/m/assign?nbh=<hash>&accelerator=T4&variant=GPU&authuser=0` returns an XSRF token; `POST` with that token in `X-Goog-Colab-Token` provisions a runtime and returns its proxy URL + token.
4. **Kernel**: standard Jupyter REST (`POST /api/kernels`, `name=python3`).
5. **Execute**: open a raw TLS+WebSocket to `/api/kernels/{id}/channels`, send a Jupyter v5 `execute_request`, parse `stream` / `display_data` / `execute_result` / `error` / `status` messages off the wire.
6. **Persist**: every event is appended to `colab_bridge/.direct_kernel_jobs/<job_id>.events.jsonl`; the `index.json` table is updated under `flock` whenever a job is added or its status changes.
7. **Detach**: `--no-stream` forks `python3 direct_kernel.py --internal-watch <job_id>` in a new session, and the parent exits with the job_id printed to stdout.

A few platform-specific details worth knowing:

- **Tailscale + asyncio**: this code uses synchronous `socket.create_connection` + `ssl.wrap_socket` rather than `asyncio.open_connection`. The non-blocking-socket path interacts badly with Tailscale's virtual NIC; the blocking path (the same one `requests` uses) works fine.
- **HTTP/2 ALPN**: the Colab proxy negotiates h2 by default, but Jupyter kernels speak WebSocket which needs HTTP/1.1. We force `set_alpn_protocols(["http/1.1"])` on the SSL context.

---

## Limitations

| Situation | What happens |
|-----------|-------------|
| Jupyter magics (`!pip`, `%matplotlib`) | Don't work inside `execute_request`. Use `subprocess.run([sys.executable, "-m", "pip", ...])`. |
| `tqdm.auto` | Uses IPython's display widget which bypasses `sys.stderr`. Use `from tqdm import tqdm` instead. For long runs followed via `--follow`, throttle with `tqdm(..., mininterval=2)` (or `mininterval=10` for hour-scale jobs) — each redraw becomes its own log line, so the default 10 Hz default is overkill. |
| Idle timeout & preemption | Mostly defeated as long as the bridge's per-runtime gRPC keep-alive thread keeps firing every 60 s (on by default) — idle A100s now routinely survive hours/days. Server-side preemption still occasionally happens but is rare; checkpoint anyway. If the bridge's reaper process is suspended (laptop sleep) or killed for ≥20 min, Colab reclaims the runtime — use `caffeinate -dis` or run the bridge on an always-on machine for overnight survival. |
| Cancel during native code | A C-extension call (e.g. a PyTorch CUDA op) may not honor `KeyboardInterrupt` until it returns. Use `dk.force_interrupt()` to kill + recreate the kernel. |
| Quota | `--accelerator A100` returns 412 if you've burned through compute units. Check `/v1/user-info` (printed by `--list`). |

---

## Files

| File | Purpose |
|------|---------|
| `colab_bridge/direct_kernel.py` | The whole bridge — CLI + Python API + persistence |
| `colab_bridge/gcs_agent.py` | GCS helper using the `gcs_creds` service-account JSON from `.env` (requires the `[gcs]` extra) |
| `colab_bridge/__init__.py` | Re-exports `ColabDirectKernel` |
| `pyproject.toml` | Package config + the `colab` console-script entry point |
| `.env.example` | Template for `colab_bridge/.env`; documents the keys the bridge auto-injects |
| `skills/colab/SKILL.md` | Ready-to-drop Claude Code skill — copy into `~/.claude/skills/colab/` so Claude knows how to drive the bridge |
| `colab_bridge/.env` | Local secrets injected as `os.environ` into each job (gitignored, mode 0600). Created by `colab --env-set` / `colab --auth`. |
| `colab_bridge/.direct_kernel_creds.json` | OAuth tokens (gitignored) |
| `colab_bridge/.direct_kernel_notebook_id` | Drive file ID of the primary Colab notebook (gitignored) |
| `colab_bridge/.direct_kernel_jobs/` | Per-job event logs + `index.json` + `runtimes.json` (active + released) + `reaper.log` + `balance.jsonl` + `events.jsonl` (gitignored) |
