## 2. Data & Golden Set

### Dataset Processing

The project uses the Customer Support on Twitter (TWCS) dataset. AmazonHelp was selected as the target brand.

The dataset was processed in the following stages:

1. **169,840** AmazonHelp brand replies were identified.
2. Customer tweets were paired with AmazonHelp replies using the tweet relationship IDs.
3. This produced **168,814 customer → AmazonHelp pairs**.
4. Language filtering retained **124,408 English pairs**.
5. Duplicate customer tweets were identified before creating the evaluation benchmark.

The historical customer messages and their corresponding AmazonHelp responses are used as the basis for retrieval and response grounding.

### Golden Set

A manually labelled golden set of **200 customer messages** was created across the seven supported intent categories.

The 200 cases were split using a fixed random seed:

- **150 cases** — reference examples used during intent classification.
- **50 cases** — held-out test set used for automated evaluation.

The held-out test set was kept separate from the reference examples to provide an independent evaluation of intent classification.

The test set contains **zero `order_change_or_cancellation` examples**, so class-level performance for that intent cannot be meaningfully evaluated from this split.