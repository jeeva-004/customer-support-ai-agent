## 5. Top 5 Failures

The following failures were selected from the 25 incorrect predictions in the 50-case held-out test set.

### Failure 1 — Out-of-scope message classified as delivery status

**Case 18:** The customer message concerns an address-related issue, but the classifier predicted `delivery_status` instead of `out_of_scope`.

**Hypothesis:** The classifier appears to have focused on delivery-related vocabulary without correctly recognizing that the request was outside the supported scope.

### Failure 2 — Delivery problem classified as refund pending

**Case 55:** The message contains both delivery and refund-related signals, including the absence of delivery and refund.

**Hypothesis:** Multiple issue signals created an ambiguous boundary between `delivery_delay_or_missing` and `refund_pending`.

### Failure 3 — Return request classified as delivery status

**Case 59:** The message contains language about an item not being dispatched yet, while the labelled intent is `return_request`.

**Hypothesis:** Delivery-status vocabulary dominated the classifier even though the intended customer-support action was related to a return.

### Failure 4 — Wrong/damaged item classified as delivery delay

**Case 60:** The customer reports a missing item within a multi-order delivery.

**Hypothesis:** The classifier interpreted the missing-item wording as a delivery problem instead of recognizing it as a `wrong_or_damaged_item` case.

### Failure 5 — Delivery delay classified as return request

**Case 72:** The message describes an item as lost/damaged and returned to the vendor.

**Hypothesis:** The word "returned" caused confusion between `delivery_delay_or_missing` and `return_request`.

### Common Failure Pattern

The main weakness is **boundary classification** rather than completely unrelated predictions. Several errors occur between closely related support intents, particularly delivery status vs. delivery delay, delivery delay vs. return, and delivery delay vs. refund. Out-of-scope messages containing support-related vocabulary also cause false in-scope predictions.