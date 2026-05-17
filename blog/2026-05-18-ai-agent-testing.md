---
title: "#7 AI agent testing — tools by role"
slug: 7
date: 2026-05-18T20:00:00+09:00
authors: [createcentury]
tags: [ai-agents, testing, observability]
---

Surveying the AI-agent testing landscape, organised by what each tool *owns*. The space is fragmented — security, accuracy, observability, and standards each have their own established names — and conflating them is the most common mistake when teams stand up an agent test pipeline.

{/* truncate */}

## Why agents are hard to test

Three properties classical unit tests don't handle:

- **Non-deterministic outputs.** A bug surfaces in 1-in-20 runs and never the same way twice.
- **Multi-step trajectories.** Failures live in tool-call sequences, not in any single function return.
- **Adversarial inputs.** Prompt injection and jailbreaks are a security category, not a corner case.

No single tool covers all three. The mature pattern is layered: security tools throw adversarial inputs at the model, evaluation tools assert on outputs and trajectories, observability tools record what actually happened, and a standard (OTel) glues it all together.

## Security & vulnerability testing

Tools whose job is to **try to break** the agent — prompt injection, jailbreak, hallucination triggers.

### promptfoo
The simplest of the bunch. Define test cases in YAML, run them against your model or agent, assert on outputs with exact match, regex, or LLM-as-judge. Strong fit for repeatable evals in CI. Its strength is **test-runner ergonomics**, not depth of attack patterns.

### PyRIT — Python Risk Identification Tool
Microsoft's automated red-teaming framework. Holds a **multi-turn conversation** with the target agent, probing for injection footholds. Where promptfoo is one-shot, PyRIT is a stateful adversary — closer in spirit to fuzzing.

### Garak — LLM Vulnerability Scanner
The *nmap of LLMs*. Ships with a catalogue of known prompt-injection techniques and hallucination triggers, sprays them at the target, scores robustness category-by-category. Good for a baseline scan; not designed for hand-crafted attack research.

| | promptfoo | PyRIT | Garak |
|---|---|---|---|
| Style | unit-test runner | stateful adversary | scanner |
| Best for | CI assertions | red-team research | baseline robustness scoring |
| Output | pass/fail | trajectory exploit logs | per-category score |

## Behaviour & accuracy evaluation

Tools that assert the agent **did the right thing** — correct tool calls, correct multi-step reasoning, correct final answer.

### Inspect AI (UK AI Safety Institute)
Strong opinions on tool-use evaluation. Designed for measuring whether an agent picked the right tool, with the right arguments, in the right sequence. The framework UK AISI uses for capability evals. Most rigorous option if you care about **trajectory correctness**, not just output correctness.

### DeepEval
"PyTest for LLMs" as a tagline. Declarative metrics — RAG faithfulness, contextual relevance, answer hallucination — wired into Python test functions. CI-native, lower-ceremony than Inspect for "evaluate model output" tasks; less depth on agentic tool-use.

| | Inspect AI | DeepEval |
|---|---|---|
| Focus | tool-use trajectories | output-level metrics |
| Style | research-grade evals | PyTest-like assertions |
| Best for | agent capability research | CI regression checks |

## Observability — what the agent did in production

Tools that record every prompt, every tool call, every response, attached to a session and queryable.

### LangSmith
The LangChain company's hosted platform. Strongest evaluation story among the observability tools: hook your traced runs up to a labelled dataset, rescore them whenever the agent changes. **Tracing + evaluation in one place.** Closed-source / commercial.

### Langfuse
Open-source. Same conceptual model (trace + dataset + score), self-hostable. Tightest OTel integration of the major observability tools — natural choice if you're standardising on open specs.

## Standards & foundations

### OpenTelemetry GenAI Semantic Conventions

[Not a tool — a spec.](https://opentelemetry.io/docs/specs/semconv/gen-ai/) Defines standard attribute names and span structure for recording GenAI activity: which model was called, which tool was invoked with what arguments, prompt and response token counts, even evaluation scores. The point is **vendor neutrality** — emit OTel from the agent, any compliant backend can ingest it.

This is the substrate. Bake OTel GenAI conventions into your agent from day one and the observability backend choice later becomes plug-and-play.

### Datadog / New Relic / Jaeger / Grafana

General-purpose observability backends that consume OTel spans. Dashboards, alerting, anomaly detection — none of them GenAI-specific, but with OTel conventions in place, all of them work fine for tracing agent runs.

The natural pairing: agent emits OTel GenAI traces → fan out to Langfuse / LangSmith **and** Datadog. The GenAI tools handle prompt-level evaluation; the general tool handles ops alerting (latency, error rates, security signals → SIEM).

## How the four pillars compose

```
                  ┌─ Garak / PyRIT       (offline, periodic robustness scans)
                  │
  agent code ──→  ├─ Inspect / DeepEval / promptfoo   (CI, on every change)
                  │
                  └─ OTel GenAI traces ──→ Langfuse / LangSmith  (always-on)
                                       └→ Datadog / Jaeger      (always-on)
```

The categories **don't substitute** for each other:

| Pillar | Answers the question |
|---|---|
| Security tooling (Garak / PyRIT) | Which attacks does the agent fail against? |
| Eval tooling (Inspect / DeepEval / promptfoo) | Did this change break functionality? |
| Observability (LangSmith / Langfuse) | What actually happened in production? |
| OTel | How do all of the above share a trace ID? |

Conflating these is the common mistake. "We have LangSmith, so we have testing" — no, you have tracing and offline scoring. You still need a security scanner. "We have promptfoo, so we have monitoring" — no, you have CI assertions. Production trajectories aren't in CI.

## What this looks like in early stages

For a small team standing up an agent:

1. **Day 1 — pick an observability tool.** Langfuse if open-source matters, LangSmith if you're already on LangChain. Either way, emit OTel GenAI traces alongside whatever proprietary format the tool wants.
2. **Day 1 — set up promptfoo or DeepEval.** Even five test cases in CI catches regressions early.
3. **Week 2 — Inspect AI for tool-use trajectories** once the agent does more than one tool call.
4. **Month 1 — Garak baseline scan.** Robustness floor; informs which jailbreak patterns to add to your eval set.
5. **Ongoing — PyRIT for targeted red-teaming** when high-stakes deployments need adversarial pressure tests.

Don't try to set up all four pillars on day one. The marginal value of adding each is bigger when the previous one is already generating actionable signal.

---

## References

- [promptfoo](https://www.promptfoo.dev/) — eval/test runner
- [PyRIT — Python Risk Identification Tool](https://github.com/Azure/PyRIT) — Microsoft red-teaming framework
- [Garak — LLM Vulnerability Scanner](https://github.com/leondz/garak)
- [Inspect AI](https://inspect.aisi.org.uk/) — UK AISI eval framework
- [DeepEval](https://github.com/confident-ai/deepeval)
- [LangSmith](https://www.langchain.com/langsmith)
- [Langfuse](https://langfuse.com/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

---

*Created: 2026-05-18 / Updated: 2026-05-18*
