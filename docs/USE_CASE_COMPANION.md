# Use Case: Always-On Mac AI Companion

> **Note**: iMessage integration is experimental. The core pipeline works; see status notes below.

Octane as an ambient intelligence layer on your Mac — watching, learning, and ready when you need it.

---

## The Daemon — Always Running

```bash
# Start Octane as a persistent background process
octane daemon start

# Check it's alive
octane daemon status

# View live logs
octane daemon logs --follow
```

Once running, the daemon:
- Maintains a priority task queue across all your requests
- Runs Shadows tasks in the background
- Broadcasts live events to Mission Control UI
- Keeps agents warm for fast response

Configure it to start at login:
```bash
# macOS launchd (add to ~/Library/LaunchAgents/)
# See docs/ARCHITECTURE.md for the plist template
```

---

## Shadows — Research While You Sleep

Shadows are perpetual background tasks that run continuously without you asking:

```bash
# All Shadows are managed via the daemon
# View running Shadows
octane daemon status
```

Built-in Shadows:
- **ResearchCycle** — periodically re-researches topics you've flagged
- **PriceMonitor** — watches your portfolio tickers
- **FileWatcher** — monitors `~/Octane/inbox/` for new documents

When FileWatcher detects a new PDF in `~/Octane/inbox/`, it automatically:
1. Extracts the text
2. Trust-scores the content
3. Adds it to your knowledge base
4. Notifies you via the Mission Control event stream

---

## Mission Control — Always-Open Dashboard

```bash
octane ui start
# → http://localhost:44480
```

Keep Mission Control open in a browser tab or a separate window. It shows:
- Live system vitals (CPU, memory, GPU activity, model load)  
- Real-time research event stream ("Extracted: NVDA earnings report", "Finding stored: attention mechanism", ...)
- Globe visualization of research activity
- Integrated terminal with persistent sessions (your shell, inside the browser)
- Portfolio dashboard

---

## File Inbox — Drop and Extract

Drop any document into `~/Octane/inbox/`:
- PDF files
- EPUB books
- HTML files

The FileWatcher Shadow picks them up automatically and adds them to your knowledge base. No command needed.

```bash
# Manually extract if needed
octane files extract ~/Octane/inbox/paper.pdf

# View extracted files
octane files list
```

---

## Quick Queries — Any Time

With the daemon running and BodegaOS Sensors live, any `octane` command responds instantly from your terminal, Alfred, Raycast, or any launcher:

```bash
octane ask "what did I research about NVDA last week?" --deep 1
octane recall search "transformer"
octane watch NVDA
```

---

## iMessage Integration (Experimental)

> Requires iMessage access permissions and BodegaOS Sensors running.

Send messages to yourself on iMessage and Octane responds:

- **"what is the stock price of NVDA?"** → live price lookup
- **"summarize my research on speculative decoding"** → recall from knowledge base
- **"research the latest in MoE models, 4 angles"** → triggers deep research pipeline

Setup:
```bash
# Grant Octane full disk access in System Settings → Privacy & Security
# Then start the iMessage shadow
octane macos imessage watch
```

Status: The text extraction pipeline and intent routing work correctly. Requires BodegaOS Sensors to be running before `imessage watch` starts. Listed as experimental — test with a non-primary iMessage account first.

---

## Notifications

Octane currently emits events to Mission Control's live stream. Native macOS notifications are planned for a future release. Until then, the event stream is the notification layer.

---

## Security and Privacy

Everything stays on your Mac:
- No cloud sync of your queries or findings
- No telemetry
- Database is local Postgres (your machine, your data)
- Audit log tracks every command for your own review
- Air-gap mode (`octane airgap on`) blocks all external network traffic if needed

The `octane vault` uses Touch ID to protect sensitive values (API keys, tokens) — they're encrypted at rest and only accessible when you authenticate.
