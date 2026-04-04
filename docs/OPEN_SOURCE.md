# Open Source at SRSWTI

What SRSWTI open-sources, what is source-available, and why.

---

## The Components

| Component | Repository | License | What It Is |
|-----------|-----------|---------|------------|
| **Octane** | `srswti/octane` | BSL 1.1 | Intelligence OS — research, analysis, knowledge accumulation |
| **BodegaOS Sensors** | — | Proprietary | Native macOS inference engine (DMG distribution) |
| **Axe** | `srswti/axe` | MIT | Agentic coding CLI for local models |

---

## Business Source License 1.1 (Octane)

Octane is source-available under [BSL 1.1](../LICENSE).

**What this means in practice:**

- ✅ **Free for personal use** — research, learning, personal projects
- ✅ **Free for non-commercial use** — academics, hobbyists, communities
- ✅ **Source code is public** — you can read, audit, and fork it
- ✅ **Self-hostable** — run it on your own Mac, audit the security claims
- ❌ **Commercial use requires a license** — contact SRSWTI for pricing
- 📅 **Converts to Apache 2.0 in 4 years** — after the conversion date, Octane is fully open source forever

---

## Why Not Full MIT?

Octane's value is in the orchestration intelligence — the OSA pipeline, BodegaRouter, IntentGate, ChatEngine, DimensionPlanner, power commands (`investigate`, `compare`, `chain`), Shadows engine, and the 44-session engineering depth behind them.

This is proprietary IP that funds SRSWTI's mission. If we MIT-licensed Octane, a cloud provider could lift the entire agent stack, run it at scale, and compete directly — without contributing back to the ecosystem that built it.

BSL is the honest middle ground: the source is public (you can verify our privacy claims), personal use is free, and commercial use funds the next 44 sessions of development.

---

## Why Is the Source Public?

The most important reason: **"your data never leaves your machine" must be verifiable, not a marketing claim.**

Anyone can clone this repo and audit exactly what Octane does with your research queries, portfolio data, and knowledge base. There are no hidden network calls, no telemetry, no silent data collection. The code is the proof.

---

## MIT Components

**Axe** (`srswti/axe`) is MIT because it is a general-purpose coding tool. The more people who use and contribute to Axe, the better it becomes for everyone. There's no commercial conflict.

---

## BodegaOS Sensors

BodegaOS Sensors is proprietary software distributed as a native macOS application. It is not open source, but it is free to use.

The inference engine contains significant proprietary R&D: the speculative decoding implementation, Metal shader optimizations, multi-model scheduling algorithms, and hardware-specific tuning. This is the core of SRSWTI's technical advantage.

---

## Contributing

Contributions to Octane are welcome for personal and educational use cases. By contributing, you agree that SRSWTI may include your contributions in commercial releases under our commercial license.

See the project's GitHub for contribution guidelines.

---

## Contact for Commercial Licensing

If you're building a commercial product on top of Octane's agent stack, contact licensing@srswti.com.

---

## The Long Game

SRSWTI's goal is not to own AI infrastructure forever — it's to build the best local AI stack possible. BSL is a mechanism to fund that work. When Octane converts to Apache 2.0, the entire codebase — 4+ years of engineering — becomes fully free.

We believe strongly in open source. We believe equally strongly that sustainable open source requires a business model.
