"""octane/cli/__init__.py — wires all CLI modules into the root Typer app."""

from __future__ import annotations

import typer


def register_all(app: typer.Typer) -> None:
    """Register every command and sub-app onto *app*."""
    from octane.cli import (
        agents,
        airgap,
        ask,
        audit,
        chat,
        dag,
        daemon,
        db,
        extract,
        files,
        health,
        macos,
        model,
        portfolio,
        power,
        pref,
        project,
        recall,
        research,
        search,
        stats,
        store,
        synthesize,
        trace,
        ui,
        vault,
        watch,
        workflow,
    )

    # ── Top-level commands ────────────────────────────────────
    health.register(app)
    ask.register(app)
    trace.register(app)
    chat.register(app)
    dag.register(app)
    agents.register(app)
    power.register(app)
    stats.register(app)

    # ── Sub-apps ──────────────────────────────────────────────
    app.add_typer(watch.watch_app,    name="watch")
    app.add_typer(pref.pref_app,      name="pref")
    app.add_typer(model.model_app,    name="model")
    app.add_typer(workflow.workflow_app, name="workflow")
    app.add_typer(research.research_app, name="research")
    app.add_typer(db.db_app,          name="db")
    app.add_typer(files.files_app,    name="files")
    app.add_typer(project.project_app, name="project")
    app.add_typer(daemon.daemon_app,  name="daemon")
    app.add_typer(vault.vault_app,    name="vault")
    app.add_typer(airgap.airgap_app,  name="airgap")
    app.add_typer(store.store_app,       name="store")
    app.add_typer(portfolio.portfolio_app, name="portfolio")
    app.add_typer(extract.extract_app,   name="extract")
    app.add_typer(recall.recall_app,     name="recall")
    app.add_typer(search.search_app,     name="search")
    app.add_typer(synthesize.synthesize_app, name="synthesize")
    app.add_typer(ui.ui_app,             name="ui")
    app.add_typer(macos.macos_app,       name="macos")
    audit.register(app)
