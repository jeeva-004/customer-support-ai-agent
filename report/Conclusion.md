## 8. Conclusion

This project implements an end-to-end customer-support AI agent for AmazonHelp using historical Twitter support conversations.

The system combines hybrid intent classification, semantic retrieval, grounded reply generation, and deterministic escalation safeguards. The evaluation also exposes important limitations: the current intent classifier achieved **50% accuracy**, below the **52% majority baseline**, while the human review showed only **30% grounding**.

Rather than presenting the system as production-ready, these results identify the main areas requiring further work: better intent-boundary classification, stronger retrieval evaluation, and stricter control of unsupported generated responses.

The current implementation therefore serves as a working experimental baseline and provides a clear path for improving reliability with additional labelled data and evaluation.