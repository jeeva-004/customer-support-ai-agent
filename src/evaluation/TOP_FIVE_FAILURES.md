Top 5 Failure Analysis

1. Case 18 — Out-of-scope → Delivery status

“I’m unable to place deliveries to my new address, it is saying that this address is blocked…”

Expected: out_of_scope
Predicted: delivery_status
Hypothesis: The classifier focused on the word deliveries instead of recognizing that address-blocking is outside the supported intent scope.

2. Case 55 — Delivery delay → Refund

“No delivery and No refund!”

Expected: delivery_delay_or_missing
Predicted: refund_pending
Hypothesis: Multiple issue signals occur in the same message, and the model selected the refund-related signal instead of the primary delivery problem.

3. Case 59 — Return → Delivery status

“Really making me want to preorder something again... Especially when it hasn’t even been despatched yet!”

Expected: return_request
Predicted: delivery_status
Hypothesis: The phrase hasn't been despatched strongly resembles delivery-status queries, even though the labelled intent is return-related.

4. Case 60 — Wrong/missing item → Delivery delay

“An item was missing from a multi-order box…”

Expected: wrong_or_damaged_item
Predicted: delivery_delay_or_missing
Hypothesis: The model interpreted missing item as a missing delivery rather than a missing item within an already received package.

5. Case 72 — Delivery delay → Return

“...deliveries ... have been ... lost, damaged and even returned to vendor…”

Expected: delivery_delay_or_missing
Predicted: return_request
Hypothesis: The message contains several delivery failure terms plus returned to vendor, which likely triggered the return-related classification.
Pattern across failures

The failures mainly occur at intent boundaries rather than simple obvious cases:

delivery_status ↔ delivery_delay_or_missing

delivery_delay_or_missing ↔ return_request

wrong_or_damaged_item ↔ delivery_delay_or_missing

delivery_delay_or_missing ↔ refund_pending

and out-of-scope messages containing support-related keywords.