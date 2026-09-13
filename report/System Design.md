## 3. System Design

The system follows a sequential support pipeline:

**Customer Message → Intent Classification → Historical Retrieval → Escalation Decision → Reply Generation**

### 3.1 Intent Classification

The classifier uses a hybrid approach:

- Deterministic keyword rules handle clearly identifiable cases such as returns, refunds, cancellations, and known out-of-scope topics.
- A local **Qwen2.5 1.5B** model running through Ollama handles cases that are not resolved by the rules.

The classifier returns exactly one of the seven supported intents.

### 3.2 Historical Retrieval

For each customer message, the system retrieves similar historical customer-support conversations.

A TF-IDF implementation was used as a simple retrieval baseline. The final retrieval implementation uses the **`all-MiniLM-L6-v2`** Sentence Transformer to create semantic embeddings for the 124,408 English customer messages.

The system compares the new message against the precomputed embeddings and returns the **top 3** most similar historical cases.

Only the historical AmazonHelp responses are passed to the reply-generation stage as grounding evidence.

### 3.3 Escalation

The system uses deterministic safeguards before generating an automatic response.

A request is escalated when:

- The predicted intent is `out_of_scope`.
- No historical cases are retrieved.
- The highest retrieval similarity is below **0.60**.

Otherwise, the request is marked for automatic handling.

### 3.4 Reply Generation

For requests eligible for automatic handling, the local Qwen2.5 1.5B model generates a concise response using the retrieved historical AmazonHelp responses as evidence.

The prompt instructs the model not to copy customer-specific information such as order numbers, usernames, tracking numbers, or other identifiers from historical conversations, and not to invent unsupported actions, policies, links, or resolution details.