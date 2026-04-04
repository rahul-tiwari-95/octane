# Octane Roadmap: April → July 4th, 2026

---

## April 2026 — Launch

**Goal: A stranger clones the repo, installs BodegaOS Sensors, runs setup.sh, and has Octane working in 5 minutes.**

- [x] Session 44: Documentation, README rewrite, setup.sh polish
- [x] `scripts/install_sensors.sh` — guided BodegaOS Sensors installation
- [x] BSL 1.1 license
- [x] Full `docs/` tree (QUICKSTART, ARCHITECTURE, USE_CASE docs, CLI_CHEATSHEET, BODEGA, EYESO)
- [ ] First users on r/selfhosted, r/LocalLLaMA, r/macstudio
- [ ] Fix top 3 issues reported by early users

**Current state at launch:**
- 1,416 tests passing
- ~100 CLI commands
- 5 specialized agents
- 16 Postgres tables
- React Mission Control UI with globe, terminal, WebSocket events
- Composable search → extract → synthesize pipeline
- Touch ID vault, air-gap mode, audit log

---

## May 2026 — Stability + iMessage

**Goal: Every feature listed as "stable" is actually stable. Beta features graduate.**

- [ ] iMessage integration tested end-to-end and documented (`octane macos imessage watch`)
- [ ] Portfolio commands tested with real brokerage CSV data (Schwab, Fidelity)
- [ ] Portfolio charts wired to real data (not mock)
- [ ] eyeso interpreter v0.1 — execute `.eyeso` files for `research`, `compare`, `brief` commands
- [ ] `octane script install/run/schedule` CLI commands
- [ ] Live market data feed for `octane watch` (beyond yfinance)
- [ ] Community issue triage — ship fixes based on early user feedback

---

## June 2026 — Distributed + Mobile

**Goal: Octane is not just one Mac. It's your network.**

### libp2p Distributed Inference
Connect multiple Macs on your local network into a single compute pool:
- Bodega discovers peers automatically via local mDNS
- Large model layers are sharded across machines
- Inference is routed to the machine with the most available unified memory
- Foundation for OctaneMesh (beyond the LAN)

### Mobile Companion (iOS)
A lightweight iOS app that connects to your Mac's Octane instance over LAN:
- View Mission Control (system vitals, live events, knowledge base stats)
- Run CLI commands remotely
- Receive portfolio alerts and research notifications
- **Not a port of Octane** — a remote control for the Mac that does the real work

### Axe ↔ Octane Integration
Code intelligence meets research intelligence:
- `octane investigate "best approach for X"` feeds context into Axe's code generation
- Axe's code analysis feeds into Octane's knowledge base
- Shared memory layer between the two tools

---

## July 4th, 2026 — Independence Day Release

**Goal: Your Mac is sovereign.**

### eyeso v1.0
- Full interpreter: variables, parallel execution, conditionals, scheduling
- Community script gallery at `scripts.octane.dev`
- `octane script install <url>` — one-command script installation
- 20+ community-contributed scripts at launch

### OctaneMesh Alpha
- Peer-to-peer compute sharing beyond the local network
- Contribute idle GPU cycles, consume from trusted peers
- Encrypted, trust-scored, invite-only mesh
- No cloud intermediary — true peer-to-peer

### Bodega 2.0
- NVFP4 quantization: 2x memory efficiency, same quality
- M5 Neural Accelerator auto-tuning (if Apple ships M5 by then)
- JACCL Thunderbolt 5 distributed inference
- Vision model support (image + text inputs)

---

## The Vision

**Independence Day Release Statement:**

> Your Mac is sovereign. It runs its own models, stores its own knowledge, connects to a mesh of trusted peers for extra compute, and sends you intelligence via iMessage. No subscriptions. No cloud dependencies. No permission needed. Independence.

---

## What We Won't Build

To stay focused, there are things Octane intentionally will not become:

- **Not a cloud product**: We will not offer a hosted version. The point is local.
- **Not a voice assistant**: No "hey Octane" wake word. Intentional, considered inputs only.
- **Not a general chatbot UI**: Octane is a research terminal. ChatGPT exists. Build on the OSA pipeline, don't recreate it.
- **Not a browser extension**: The model needs to live close to the GPU. Browser extensions pull everything to the cloud.

---

## How to Follow Progress

- GitHub releases and changelog at `github.com/srswti/octane`
- Session notes at `srswti.com/sessions`
- Community at `r/srswti` (planned)
