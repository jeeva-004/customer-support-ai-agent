# Decision Log

This log documents key architectural, engineering, data, and evaluation decisions made during the development of the Customer Support AI Agent for AmazonHelp.

---

## Decision 1 — AmazonHelp Brand Selection
- **Decision**: Selected `@AmazonHelp` as the target brand from the Twitter Customer Support (TWCS) dataset.
- **Why**: `AmazonHelp` contained the highest volume of inbound customer inquiries and outbound resolutions in the raw TWCS dataset (169,840 brand replies), providing a rich historical corpus for retrieval-augmented generation.
- **Trade-off / limitation**: The resulting retrieval corpus and agent behavior are specific to Amazon support domain knowledge and cannot be directly applied to other brands without re-indexing brand-specific data.

---

## Decision 2 — English-Only Scope Filtering
- **Decision**: Filtered customer-brand pairs to restrict dataset scope to English-language customer messages (`src/filter_english.py`).
- **Why**: Standardizes intent classification and semantic vector embeddings using `langdetect` and `all-MiniLM-L6-v2` without requiring multilingual model overhead.
- **Trade-off / limitation**: Non-English customer inquiries (e.g., Spanish or German) are excluded during preprocessing and cannot be handled by the current agent.

---

## Decision 3 — Tweet ID Parent-Child Pairing
- **Decision**: Paired brand responses with inbound customer messages using `in_response_to_tweet_id` (`src/filter_brand.py`).
- **Why**: Reconstructs exact 1-to-1 customer inquiry and brand resolution pairs from raw, unstructured Twitter logs (producing 168,814 customer → AmazonHelp pairs).
- **Trade-off / limitation**: Restricts conversation context to single 1-to-1 tweet exchanges, omitting multi-turn dialogue history that occurred across preceding tweets.

---

## Decision 4 — 7-Intent Classification Taxonomy
- **Decision**: Defined a taxonomy of 6 core e-commerce support intents (`delivery_status`, `delivery_delay_or_missing`, `return_request`, `refund_pending`, `wrong_or_damaged_item`, `order_change_or_cancellation`) plus `out_of_scope`.
- **Why**: Encapsulates common retail support inquiries while isolating non-support or unsupported requests for safe human escalation.
- **Trade-off / limitation**: Fine-grained semantic overlap between adjacent categories (e.g., `delivery_status` vs `delivery_delay_or_missing`) leads to classification confusion.

---

## Decision 5 — 200 Manually Labelled Golden Cases
- **Decision**: Created a 200-case manually annotated golden dataset (`eval/golden_set.csv`) sampled randomly from unique customer tweets (`src/check_golden_data.py`).
- **Why**: Establishes a human-verified ground-truth benchmark for evaluating intent classification accuracy, retrieval quality, and reply generation.
- **Trade-off / limitation**: Manual labeling is labor-intensive, capping the golden evaluation dataset size at 200 total cases.

---

## Decision 6 — Fixed 150 Reference / 50 Test Split
- **Decision**: Split the 200 golden set cases into 150 reference cases (`eval/golden_reference.csv`) and 50 held-out test cases (`eval/golden_test.csv`) using `random_state=42` (`src/split_golden_set.py`).
- **Why**: Provides 150 few-shot prompt reference examples and TF-IDF baseline reference data while retaining a strict, reproducible held-out test set for evaluation.
- **Trade-off / limitation**: The 50-case held-out test set is relatively small and contained zero sample representations for the `order_change_or_cancellation` intent class.

---

## Decision 7 — TF-IDF Retrieval Baseline
- **Decision**: Implemented a TF-IDF vectorizer with cosine similarity (`src/retrieve.py` / `src/classify_tfidf.py`) as an initial retrieval baseline.
- **Why**: Provides a computationally lightweight keyword-matching benchmark to measure the performance gain of dense semantic embeddings.
- **Trade-off / limitation**: TF-IDF relies strictly on token overlaps and fails when customers express identical support intents using different vocabulary.

---

## Decision 8 — Sentence Transformer Dense Retrieval
- **Decision**: Adopted `SentenceTransformer("all-MiniLM-L6-v2")` to generate 384-dimensional dense semantic vectors (`src/retrieve_embeddings.py`).
- **Why**: Enables semantic search across 124,408 English customer-agent pairs based on underlying message meaning rather than literal word matching.
- **Trade-off / limitation**: Embeddings are queried via in-memory NumPy matrix multiplication (`CASE_EMBEDDINGS @ query_embedding`) rather than a dedicated, indexed production vector database.

---

## Decision 9 — Top-3 Retrieval Context Window
- **Decision**: Configured dense retrieval to select the top `k=3` matching historical support cases for reply grounding (`src/retrieve_embeddings.py`).
- **Why**: Provides sufficient historical brand resolution context to the LLM without overwhelming the prompt context window or introducing conflicting evidence.
- **Trade-off / limitation**: If fewer than 3 truly relevant cases exist in the corpus for a niche query, marginal 2nd or 3rd matches may introduce noisy prompt context.

---

## Decision 10 — Local Qwen2.5 1.5B via Ollama
- **Decision**: Deployed the local `qwen2.5:1.5b` model via Ollama (`http://localhost:11434`) for intent classification and reply generation instead of external cloud APIs.
- **Why**: Ensures fully offline, privacy-preserving, cost-free execution without sending customer conversation data to third-party API providers.
- **Trade-off / limitation**: The 1.5B small language model exhibits limited instruction compliance, leading to hallucinated short links (`https://t.co/...`) and a 30% human grounding score.

---

## Decision 11 — Hybrid Keyword Rules + LLM Fallback Classifier
- **Decision**: Implemented a hybrid intent classifier combining deterministic regex/keyword rules with LLM fallback (`src/classify_intent.py`).
- **Why**: Deterministically catches high-confidence intent signals (e.g., "alexa" → `out_of_scope`, "return" → `return_request`, "refund" → `refund_pending`) instantly before invoking slower LLM inference.
- **Trade-off / limitation**: Keyword rules can trigger false positives on complex queries (e.g., "returned to vendor" triggering `return_request` instead of `delivery_delay_or_missing`).

---

## Decision 12 — Similarity Threshold Escalation Safeguard (0.60)
- **Decision**: Enforced a minimum cosine similarity threshold of 0.60 on top-retrieved historical cases (`src/escalate.py`).
- **Why**: Prevents auto-handling customer queries when historical evidence is insufficiently relevant, triggering escalation to human agents (`"decision": "escalate"`).
- **Trade-off / limitation**: Unusually phrased but valid support queries may fall below 0.60 similarity, causing unnecessary human escalation.

---

## Decision 13 — Mandatory Out-of-Scope Escalation
- **Decision**: Configured escalation logic to automatically route all queries classified as `out_of_scope` to human agents.
- **Why**: Prevents the agent from attempting automated resolutions on unsupported domains such as Prime subscription billing or physical device troubleshooting.
- **Trade-off / limitation**: If the intent classifier misclassifies a valid support query as `out_of_scope`, it is unnecessarily escalated rather than auto-handled.

---

## Decision 14 — Rejection of Unreliable LLM Evidence-Judge Metric
- **Decision**: Excluded the LLM evidence-judge evaluation results (100% claim support rate from `src/evaluate_evidence.py`) from reportable metrics.
- **Why**: Empirical verification revealed that the 1.5B LLM judge falsely marked hallucinated links and unsupported claims as "supported", rendering the metric ungrounded.
- **Trade-off / limitation**: Requires relying on manual human evaluation ($n=10$) for reply grounding rather than automated LLM judge scoring.

---

## Decision 15 — Transparent Metric Reporting Against Baseline
- **Decision**: Reported the final held-out intent classification accuracy as 50.00% alongside the 52.00% majority-class baseline without altering evaluation logic or masking results.
- **Why**: Upholds rigorous engineering standards by acknowledging that the current hybrid classifier does not beat the majority baseline on the 50-case test set.
- **Trade-off / limitation**: Explicitly highlights that intent classification requires larger models or fine-tuning prior to any real-world deployment.
