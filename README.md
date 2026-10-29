# Does AI Feedback That Explains Keep Users Engaged? Evidence from WildChat

An empirical study of real user–chatbot conversations. When a user shares their own attempt (code, an answer, their reasoning), does the *kind* of feedback the AI gives change whether they engage with it next?

**Context:** MSc Business Analytics, Warwick Business School · individual research project · Python

## Research question

How are characteristics of AI feedback (explanation, process focus, giving the solution) associated with what users do next: engaging with the feedback, revising their work, or just asking for the answer?

## Data and sampling

[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (AI2) contains about 838k real ChatGPT conversations, narrowed down step by step:

| Step | Conversations |
|---|---|
| Raw dataset | 837,989 |
| English, 3 or more turns | 97,064 |
| Quality filters (agent loops, non-English text, very short messages removed) | 81,611 |
| Random sample (seed 42) | 5,000 |
| Contain a user attempt (Codebook A) | 831 (16.6%) |
| **Analytic sample:** coding, homework and reasoning turn triples where the user message is an attempt | **560 triples from 314 conversations** |

The unit of analysis is a **turn triple**: user attempt → AI feedback → the user's next message.

## Method: LLMs as measurement instruments

Three codebooks were applied by an LLM classifier via OpenRouter, with structured JSON output, 8 concurrent workers and resumable runs:

- **Codebook A** flags whether a conversation contains a user-produced attempt.
- **Codebook B** rates the feedback's explanation depth (0–3) and whether it addresses process or provides the solution.
- **Codebook C** records the user's next move: engages, revises, or requests the solution.

The first design used *corrective vs elaborative* feedback categories. Pilot testing showed only ~5% of responses were purely corrective, so it was replaced with the depth scale plus independent markers.

**Every classifier was validated against blind human coding:**

| Measure | n | Agreement | κ |
|---|---|---|---|
| Contains an attempt (A) | 100 | 91% | 0.69 |
| Engages with feedback (C) | 50 | 76% | 0.51 |
| Any explanation (B, binary) | 50 | 82% | 0.43 |
| Explanation depth (B, 4-level) | 50 | – | 0.46 (weighted) |

## Analysis

The models are logistic regressions with **standard errors clustered by conversation**, since conversations contribute multiple triples, reported as odds ratios with 95% CIs.

## Findings

| Feedback characteristic | Outcome | Result |
|---|---|---|
| **Any explanation** | Engagement | **OR = 3.53** [1.82, 6.85], p < .001 |
| Explanation depth 1 vs 2 vs 3 | Engagement | No difference between depths (joint p = .33): a **threshold, not a dose-response** |
| Process-focused feedback | Engagement | OR = 1.50, p = .04 |
| Giving the solution | Revision | OR = 2.19, p = .008 (could be copying, not learning) |
| Any explanation | Requesting the solution | No association (OR = 1.01) |

Engagement was 31% after feedback with no explanation, and 66–68% at any explanation depth.

**Robustness:** the main result holds when controlling for response length (OR = 3.44), in coding-only data (OR = 3.24), with total conversation length as a control (OR = 3.29) and in a GEE model (OR = 3.46). VIFs are all below 1.6.

**Caveats:** these are associations, not causal effects. The engagement measure has moderate reliability, and the turn-level attempt filter wasn't separately validated.

## Run it

```bash
pip install datasets huggingface_hub openai pandas tqdm statsmodels scikit-learn patsy
```

1. Accept the WildChat licence on Hugging Face and create an access token.
2. Get an OpenRouter API key.
3. Run `wildchat_feedback_analysis.ipynb`. Both keys are requested with `getpass`, so they're never saved in the notebook.

*Per the WildChat licence, no conversation data, classifier outputs or hand-labelled files are included here. Notebook cells that displayed raw conversation text have had their outputs cleared.*

## Tech stack

Python · Hugging Face Datasets · OpenRouter / LLM classification · statsmodels (clustered SEs, GEE) · scikit-learn (Cohen's κ) · research design
