# eyeso

eyeso is SRSWTI's scripting language for expressing AI-powered automation on your Mac. It is designed to be readable by anyone — not just programmers.

> **Status: Planned** — The language specification is defined. The interpreter is in development. Example scripts exist and will be executable in a future release.

---

## What eyeso Is

eyeso is a domain-specific language where you describe what you want to research, monitor, or build — and Octane figures out how to do it.

```eyeso
research "AI chip export restrictions" depth: 6 cite: true
compare "NVDA vs AMD" dimensions: 4
watch NVDA alert_above: 900 alert_below: 800
brief to: imessage
```

That script would: research the topic, compare the two companies, watch the stock, and send you a briefing via iMessage.

---

## Design Philosophy

- **Human readable**: A non-programmer should be able to read and understand any eyeso script
- **Declarative**: You describe the goal, not the steps
- **Composable**: Steps chain naturally, with output flowing into the next step
- **Local-first**: All execution is on your Mac, using Octane's pipeline

---

## Language Concepts

### Commands

The core vocabulary mirrors Octane's CLI:

| Command | Description |
|---------|-------------|
| `research` | Deep research on a topic |
| `compare` | Side-by-side comparison |
| `watch` | Live monitoring with alerts |
| `extract` | Pull content from a URL |
| `synthesize` | Produce a report |
| `recall` | Search your knowledge base |
| `brief` | Deliver a summary (to terminal, iMessage, file) |
| `store` | Save a value for later use |

### Parameters

Parameters are key-value pairs after the command:

```eyeso
research "transformer architecture" depth: 4 cite: true verify: false
```

### Variables

```eyeso
topic = "speculative decoding"
research $topic depth: 6
brief $topic to: file path: "~/research/spec-decoding.md"
```

### Parallel Execution

```eyeso
parallel:
  research "NVDA earnings"
  research "AMD earnings"
  watch NVDA AAPL MSFT
```

### Conditionals

```eyeso
watch NVDA
  if price_above: 950 then:
    brief "NVDA hit target" to: imessage
  if price_below: 800 then:
    research "NVDA downside risks" depth: 4
    brief to: imessage
```

### Scheduling

```eyeso
schedule daily at: "08:00":
  research "AI news today" depth: 3
  brief to: imessage
```

---

## Example Scripts

### Morning Brief

```eyeso
# morning_brief.eyeso
research "AI news today" depth: 3
watch NVDA AAPL MSFT GOOGL
recall "recent portfolio findings"
brief "Morning AI + Market Brief" to: imessage
```

### Competitive Intelligence

```eyeso
# comp_intel.eyeso
topic = "AI inference hardware landscape"
research $topic depth: 8 cite: true
compare "NVDA vs AMD vs Intel in AI chips" dimensions: 6
store results as: "comp_intel_q2_2026"
brief to: file path: "~/reports/comp_intel.md"
```

### Document Pipeline

```eyeso
# watch_inbox.eyeso
watch_folder "~/Octane/inbox"
  on: new_file do:
    extract file
    synthesize query: "key points"
    store in: knowledge_base
    brief "New document processed" to: terminal
```

---

## Running eyeso Scripts

> Interpreter is in development. When available:

```bash
# Run a script
octane script run morning_brief.eyeso

# Install a script (makes it available by name)
octane script install comp_intel.eyeso

# Run installed script
octane script run comp_intel

# Schedule a script
octane script schedule morning_brief.eyeso --daily --at 08:00

# View scheduled scripts
octane script list --scheduled
```

---

## Community Scripts

A community script gallery is planned at `scripts.octane.dev` (July 4th 2026 target). Scripts will be shareable, searchable, and installable with one command.

---

## Roadmap

| Milestone | Target |
|-----------|--------|
| Language spec finalized | ✅ Done |
| Example scripts created | ✅ Done |
| Interpreter v0.1 (research + compare + brief) | May 2026 |
| Interpreter v0.5 (variables, parallel, conditionals) | June 2026 |
| eyeso v1.0 + community gallery | July 4th 2026 |
