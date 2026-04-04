# Use Case: Open Source Intelligence (OSINT)

Octane as a local OSINT research platform — structured information gathering, cross-referencing, and analysis with full local data sovereignty.

---

## Why Local OSINT?

Cloud-based research tools log your queries. For sensitive research — corporate due diligence, competitive intelligence, investigative research — your search patterns themselves reveal your interests.

Octane runs entirely on your Mac. No query logs. No external data retention. The network traffic is standard web browsing (search engines, news sites, arXiv) — Octane just automates and structures what you'd do manually.

---

## Subject Research

```bash
# Broad landscape research
octane investigate "AI chip supply chain dependencies" --deep 8 --cite

# Specific entity research
octane investigate "TSMC capacity constraints 2025 2026" --deep 6 --cite

# Trend analysis
octane investigate "shift from CUDA to alternative GPU frameworks" --deep 6
```

---

## Multi-Source Cross-Reference

```bash
# Web + academic papers + news
octane search web "entity name recent developments" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "summarize key findings"

octane search arxiv "topic of interest" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "academic consensus"

octane search news "subject name" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "recent reporting"
```

---

## Structured Comparison

```bash
# Compare two entities across multiple dimensions
octane compare "NVDA vs AMD data center strategy" --deep 6 --cite

# Compare claims
octane compare "company A revenue projections vs analyst consensus" --deep 4
```

---

## Document Analysis

Drop PDFs, HTML files, or documents into `~/Octane/inbox/` and Octane auto-extracts them. Or extract manually:

```bash
# Extract a local document
octane files extract ~/Downloads/earnings_report.pdf

# Extract from a URL
octane extract url "https://sec.gov/filing/..."

# View extracted content
octane files list
octane files show <file-id>
```

---

## YouTube Intelligence

Conferences, interviews, and presentations often contain information not in written form:

```bash
# Extract intelligence from YouTube content
octane search youtube "CEO company name interview 2025" --json \
  | octane extract stdin --json \
  | octane synthesize run --stdin --query "key statements and claims"
```

Octane extracts the full transcript, trust-scores it, and synthesizes the key points.

---

## Persistent Research Projects

```bash
# Create a project for a research operation
octane project create "Supply Chain Analysis Q2 2026"
octane project switch "Supply Chain Analysis Q2 2026"

# All research is tagged to this project
octane investigate "Taiwan semiconductor dependencies" --deep 8 --cite
octane investigate "ASML export restrictions impact" --deep 6
octane compare "TSMC vs Samsung manufacturing capabilities" --deep 4

# Review project findings
octane project status
```

---

## Knowledge Base Mining

After multiple research sessions on a topic, your knowledge base becomes the primary resource:

```bash
# Search everything accumulated
octane recall search "semiconductor supply chain"
octane recall search "TSMC capacity"
octane recall search "HBM memory"

# Combined stats
octane stats
```

---

## Verification and Cross-Check

```bash
# Research with verification pass (cross-checks claims across sources)
octane investigate "subject claim to verify" --deep 6 --cite --verify
```

The `--verify` flag adds a second synthesis pass that explicitly cross-references inconsistencies between sources and flags low-confidence claims.

---

## Air-Gap Research (Maximum Privacy)

For highly sensitive work — research that shouldn't touch the network after data is gathered:

```bash
# Gather all relevant documents first (while connected)
octane search web "topic" --json | octane extract stdin --json
octane files extract ~/relevant-document.pdf

# Enable air-gap — all external network traffic blocked
octane airgap on

# Now synthesize and analyze entirely offline
octane recall search "topic"
octane synthesize run --stdin  # operates on locally cached content

# Re-enable when done
octane airgap off
```

---

## Audit Trail

Every command is logged:

```bash
octane audit log
octane audit log --limit 100
octane audit export > research_audit.json
```

The audit log is append-only and stored in local Postgres. Useful for documenting research methodology.

---

## Important Notes

- **Legal use only**: Octane is for publicly available information. Use responsibly and in compliance with applicable laws including CFAA, GDPR, and terms of service of accessed websites.
- **Rate limiting**: Octane adds delays between requests to be a polite web client. Do not attempt to bypass this.
- **No credential extraction**: Octane does not access password-protected resources or bypass authentication.
