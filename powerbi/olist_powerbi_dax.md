# Power BI Semantic Layer and Report Design

## Data model

Use a star-style model with \`orders\` as the operational fact grain and related \`order_items\`, \`order_payments\`, \`order_reviews\`, \`customers\`, \`products\`, and \`sellers\`.

Relationships:

- \`orders[order_id]\` → \`order_items[order_id]\`, \`order_payments[order_id]\`, \`order_reviews[order_id]\`
- \`customers[customer_id]\` → \`orders[customer_id]\`
- \`products[product_id]\` → \`order_items[product_id]\`
- \`sellers[seller_id]\` → \`order_items[seller_id]\`

Create a Date table linked to \`orders[order_purchase_timestamp]\`.

## DAX measures

\`\`\`DAX
Delivered Orders =
CALCULATE(DISTINCTCOUNT(orders[order_id]), orders[order_status] = "delivered")

GMV =
CALCULATE(SUM(order_items[price]), orders[order_status] = "delivered")

AOV = DIVIDE([GMV], [Delivered Orders])

GMV Previous Month =
CALCULATE([GMV], DATEADD('Date'[Date], -1, MONTH))

MoM GMV Growth % =
DIVIDE([GMV] - [GMV Previous Month], [GMV Previous Month])

Late Delivered Orders =
CALCULATE(
    DISTINCTCOUNT(orders[order_id]),
    orders[order_status] = "delivered",
    FILTER(orders, orders[order_delivered_customer_date] > orders[order_estimated_delivery_date])
)

SLA Breach Rate % = DIVIDE([Late Delivered Orders], [Delivered Orders])

Payment Value = SUM(order_payments[payment_value])

Average Installments = AVERAGE(order_payments[payment_installments])

Low Review Orders =
CALCULATE(
    DISTINCTCOUNT(order_reviews[order_id]),
    FILTER(order_reviews, order_reviews[review_score] <= 2)
)

Low Review Rate % = DIVIDE([Low Review Orders], DISTINCTCOUNT(order_reviews[order_id]))
\`\`\`

## Report pages

1. **Executive Summary:** KPI cards, monthly GMV trend, payment mix, SLA breach rate, and state hotspot map.
2. **Operations & Delivery:** approval vs transit funnel and state delay distribution.
3. **Customer Retention:** cohort heatmap and RFM segment value/count.
4. **Customer Experience:** late-delivery and review-score relationship.
5. **Experimentation:** checkout-offer conversion, confidence interval, and launch decision.

## Drill-through

Create a **State Delivery Detail** drill-through page using \`customers[customer_state]\` as the drill-through field. Include order ID, purchase date, estimated/actual delivery dates, delay days, seller, payment value, and review score. Add a back button and retain all filters.

