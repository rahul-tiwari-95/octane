"""Octane Daemon — persistent background service.

The daemon is the beating heart of Octane. It manages:
- Priority task queue (P0 interactive → P3 batch)
- Shared state (topology, active models, sessions)
- Connection pools (Bodega, Postgres, Redis)
- Intelligent data routing (algorithmic Redis vs Postgres placement)
- Model lifecycle (aggressive unloading, reload on demand)
- Unix socket IPC for CLI ↔ daemon communication

The daemon is OPTIONAL — all CLI commands work without it.
When running, it accelerates everything through shared state and pools.
"""
