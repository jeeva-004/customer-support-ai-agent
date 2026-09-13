## 4. Evaluation & Baselines

### Intent Classification

The classifier was evaluated on the fixed 50-case held-out test set.

| Metric | Result |
|---|---:|
| System accuracy | **50.00%** |
| Majority-class baseline | **52.00%** |

The majority baseline predicts `out_of_scope` for every test case, since it is the most frequent class in the held-out set.

The system therefore **does not outperform the simple majority baseline** on this evaluation. This result is reported as-is rather than tuning the system against the test set.

The test set contains only 50 cases and has no `order_change_or_cancellation` examples, limiting the reliability of per-intent conclusions.

### Reply Quality — Human Review

A separate subset of **10 non-out-of-scope cases** was manually reviewed for relevance, helpfulness, and grounding.

| Dimension | Positive rate |
|---|---:|
| Relevance | **100%** |
| Helpfulness | **70%** |
| Grounding | **30%** |

These are **positive rates from a single human rater**, not inter-rater agreement scores.

The results show that the generated responses were generally relevant to the customer message, but grounding remained a significant weakness.

### LLM-Based Evaluation

An LLM-as-judge experiment was also implemented to assess reply relevance, helpfulness, and grounding. An evidence-judge experiment was tested separately.

These experiments were treated as **experimental rather than final metrics** because the small local model produced unreliable judgements. The evidence-judge result was therefore not reported as a system-quality metric.