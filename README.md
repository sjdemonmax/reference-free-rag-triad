# Reference-Free RAG Quality & Hallucination Audit Pipeline

> **Role**: Senior AI Evaluation Engineer · Entrance-Exam EdTech Platform

## The Problem

Production RAG systems serving students face an adversarial evaluation problem: **incoming doubts have no ground-truth answers**. Classical accuracy metrics — which require a labelled reference answer to compare against — cannot be computed on live, unlabeled student queries. This creates a dangerous blind spot: the system may be hallucinating with high confidence on fringe queries, and there is no mechanism to catch it.

In an educational context, the stakes are uniquely high. A hallucinated answer to a Physics or Chemistry question from a student preparing for a national entrance exam doesn't just misinform — it actively builds incorrect mental models that must be unlearned under time pressure.

## The Solution

This project implements a **reference-free RAG quality and hallucination audit pipeline** using the RAG Triad evaluation framework — enabling rigorous quality benchmarking on fully unlabeled incoming student queries.

### The RAG Triad Framework

Three orthogonal, reference-free evaluation dimensions are computed for every student query:

| Dimension | What It Measures |
|---|---|
| **Context Relevance** | Whether retrieved curriculum chunks contain knowledge pertinent to the query |
| **Groundedness / Faithfulness** | Whether every atomic factual proposition in the generated answer is entailed by the retrieved chunks |
| **Answer Relevancy** | Whether the answer directly and responsively addresses the student's question |

### How It Works

1. **Dense Semantic Retrieval** — 500 curriculum chunks are encoded via SentenceTransformers (); top-3 chunks retrieved per query using cosine similarity.
2. **Context Relevance Scoring** — Decaying-weighted cosine similarity across top-3 chunks (70/20/10 weights), min-max scaled to [0, 1].
3. **Proposition Extraction & Groundedness** — Generated answers are decomposed into atomic factual propositions; each proposition is entailment-scored against the retrieved chunks via dense cosine similarity.
4. **Answer Relevancy Scoring** — Query-answer semantic alignment measured via normalised embedding dot product.
5. **Composite RAG Quality Index (RQI)** — A weighted composite of all three dimensions, calibrated against an empirically derived threshold to separate grounded answers from hallucinations.
6. **Automated Classification & Escalation** — Queries below the RQI threshold are flagged and routed to faculty review queues; grounded answers are auto-approved.
7. **Hallucination Radar Dashboard** — A Streamlit dashboard surfaces flagged queries, error distributions by academic subject, and faculty escalation queues for operational review.

### Tech Stack

- **Embeddings**: SentenceTransformers ()
- **Vector Operations**: Scikit-learn (cosine similarity) · NumPy
- **Analysis**: Python · Pandas · Matplotlib · Seaborn
- **Dashboard**: Streamlit (Academic Operations Hallucination Radar)
- **Notebook**: JupyterLab (200 unlabeled doubts benchmarked across 4 difficulty tiers)

## Measurable Business Impact

| Metric | Result |
|---|---|
| Queries audited | **200 unlabeled student doubts** (zero ground-truth labels required) |
| Difficulty tiers covered | 4 (standard, conceptual, ambiguous, out-of-scope hallucination baits) |
| RQI operating cutoff | **0.72** — auto-approves 53% of queries; safely escalates 47% |
| Out-of-scope hallucination interception | **100%** of fringe hallucination-bait queries correctly flagged |
| Mean Context Relevance (standard queries) | **0.68** (vs. 0.31 on out-of-scope, confirming correct retrieval drift detection) |
| Faculty escalation queue | Automated, prioritised, subject-stratified — zero manual triage required |
| Strategic value | First reference-free quality gate that works on live production traffic without ground-truth labels |

The system solves the core dilemma of deploying AI in education at scale: how do you audit quality when you cannot know the right answer in advance? By evaluating *internal coherence* between retrieval, generation, and query — rather than comparing to a reference — the pipeline provides continuous, automated quality assurance on 100% of live traffic.

## Repository Contents

| File | Description |
|---|---|
|  | Starter notebook with scaffolding |
|  | Fully executed audit notebook (retrieval → RQI scoring → classification → export) |
|  | Audit classification log (200 queries with RQI scores and pass/escalate labels) |
|  | Streamlit Hallucination Radar dashboard |
|  /  | Supporting pipeline modules |
|  | Executive visualisations (score distributions, subject breakdowns, confusion analysis) |
|  | Curriculum knowledge base (500 chunks across 4 subjects) |
|  | 200 unlabeled student doubts (4 difficulty tiers) |
|  | Detailed pipeline walkthrough |
