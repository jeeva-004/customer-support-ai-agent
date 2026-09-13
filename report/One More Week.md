## 7. Next Steps — One More Week

If one additional week were available, the priority would be improving reliability rather than adding more features.

1. **Improve intent boundary handling**  
   Add more labelled examples for confusing pairs such as delivery status vs. delivery delay, delivery delay vs. return, and delivery delay vs. refund.

2. **Improve out-of-scope detection**  
   Strengthen the classifier's ability to reject messages that contain delivery, refund, or order-related words but are actually outside the supported scope.

3. **Improve retrieval quality**  
   Evaluate retrieval systematically on labelled queries and investigate filtering or reranking strategies to reduce irrelevant historical cases.

4. **Improve reply grounding**  
   Prevent the generator from introducing unsupported links, actions, policies, or replacement/refund claims. A claim-level grounding check could be added before allowing automatic handling.

5. **Expand evaluation coverage**  
   Increase the held-out test set and ensure every supported intent has sufficient evaluation examples, including `order_change_or_cancellation`.

6. **Evaluate end-to-end safety**  
   Measure not only whether a response is relevant, but whether the complete system makes the correct auto-handle vs. escalation decision.