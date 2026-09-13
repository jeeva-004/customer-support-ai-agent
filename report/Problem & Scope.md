## 1. Problem & Scope

### Problem

Customer-support conversations on social media contain a high volume of short, informal customer requests. The goal of this project is to build an AI-assisted support agent that can classify each incoming customer message, retrieve relevant historical support resolutions, generate a grounded draft response, and decide whether the request can be automatically handled or should be escalated.

### Scope

The project uses the Customer Support on Twitter (TWCS) dataset and focuses on **AmazonHelp** conversations. The system is restricted to **English-language customer messages** and supports seven intent categories:

- `delivery_status`
- `delivery_delay_or_missing`
- `return_request`
- `refund_pending`
- `wrong_or_damaged_item`
- `order_change_or_cancellation`
- `out_of_scope`

Requests outside the supported customer-support scope are classified as `out_of_scope` and escalated rather than automatically handled.

The system is designed as an experimental internship submission rather than a production-ready customer-support system.