# Aaranya Marketing Analytics

## AI-Powered Marketing Decision Support System

> **Business question:** Where should Aaranya spend its next ₹1 of marketing budget?

Aaranya is an end-to-end marketing analytics and AI decision-support system.

The system combines:

- PostgreSQL
- Python
- Customer analytics
- RFM segmentation
- Cohort analysis
- Marketing attribution
- Marketing Mix Modeling (MMM)
- Adstock / carryover modeling
- Saturation / response curves
- Budget optimization
- Retrieval-Augmented Generation (RAG)
- Ollama / Llama 3.2
- Agent-based routing
- Gradio

## 1. Business Objective

The central business question is:

> Where should Aaranya spend its next ₹1 of marketing budget?

The system analyzes historical customer and marketing data and produces modeled budget-allocation scenarios under defined assumptions and historical spending constraints.

## 2. System Architecture

```text
User / Management
        ↓
Gradio Chat Interface
        ↓
Agent Router
        ↓
   ┌────┴────┐
   ↓         ↓
 RAG      Decision Engine
   ↓         ↓
Llama 3.2   Python Optimizer
   └────┬────┘
        ↓
   Final Answer
```

## 3. Analytics Pipeline

The project includes:

- Data quality checks
- Customer 360
- RFM segmentation
- Cohort analysis
- Marketing attribution
- Geographic analysis
- Fixed-effects modeling
- Marketing Mix Modeling
- Adstock
- Saturated response curves
- Marginal return analysis
- Budget optimization

## 4. Data Quality

The source order table contained duplicated order IDs.

Observed:

- 79,721 source order rows
- 79,483 unique order IDs
- 238 duplicated order IDs
- Each duplicated order ID appeared twice

A cleaned order table was created with one row per order while preserving the source tables.

## 5. Customer Analytics

Customer 360 combines customer and order information to support customer-level analysis.

Customer types include:

- One-time
- Repeat
- Loyal

RFM segmentation includes:

- Potential / Mid-Value
- At Risk
- Champions
- Lost
- High Value At Risk
- New / Promising
- Loyal Customers

Champions and High Value At Risk customers together represent approximately 44.45% of observed revenue.

## 6. Cohort Analysis

Customer cohorts are based on the month of first purchase.

The retention analysis measures observed periodic purchasing activity.

Retention should not automatically be interpreted as customer survival or confirmed churn.

## 7. Marketing Attribution

Marketing sessions are linked to orders using the available tracking relationships.

The system calculates campaign-level performance using:

- Spend
- Attributed orders
- Attributed revenue
- Contribution margin
- ROAS
- Contribution ROAS
- CAC proxy

Important limitation:

> Attribution is observational and does not prove causal or incremental marketing impact.

## 8. Marketing Mix Modeling

A weekly Marketing Mix Model was developed using marketing spend and business outcomes.

The primary model uses adstocked channel spend.

Model specifications were evaluated using rolling-origin validation.

## 9. Adstock

Marketing effects can carry over across weeks.

The adstock formulation is:

```text
adstock_t = spend_t + decay × adstock_(t-1)
```

The primary MMM uses a decay parameter of 0.75.

The saturated response model uses a decay parameter of 0.50.

## 10. Saturation / Response Curve

The response model uses a saturation transformation:

```text
saturated_adstock = adstock / (adstock + K)
```

This represents diminishing modeled response as spending increases.

## 11. Budget Optimization

The decision engine answers:

> Given an additional marketing budget, how should it be allocated across channels?

The optimizer uses:

- Latest observed weekly spending
- Adstock carryover
- Channel-specific saturation
- Historical spending constraints
- The fitted response model

The numerical optimization is performed by deterministic Python code.

The LLM does not perform the financial arithmetic.

## 12. Historical Spending Constraint

The optimizer uses historical P95 channel spending as an upper bound.

This prevents unrestricted extrapolation far beyond observed spending.

P95 is a modeling and feasibility constraint. It is not evidence that P95 spending is optimal.

## 13. Example Budget Decision

For an additional ₹50,000 budget, the current decision engine produces:

| Channel | Incremental Allocation |
|---|---:|
| Affiliate Network | ₹22,018.19 |
| CRM | ₹6,040.27 |
| Google Ads | ₹0.00 |
| Meta Ads | ₹0.00 |
| Influencer (direct deal) | ₹21,941.54 |

Latest observed weekly budget: ₹202,423.14

Recommended weekly budget: ₹252,423.14

Modeled incremental contribution margin: ₹215,188.83

Modeled incremental contribution margin per rupee: ₹4.30

These are modeled outputs and are not guaranteed future results.

## 14. RAG System

Aaranya contains a project-specific knowledge base in:

`docs/aaranya_knowledge.md`

Retrieval combines:

- Word-level TF-IDF
- Character-level TF-IDF
- Intent classification
- Intent-aware retrieval boosting

The retrieved context is passed to the local LLM.

## 15. Local LLM

Aaranya uses Ollama with Llama 3.2.

The model runs locally and does not require OpenAI API credits for normal operation.

The LLM is primarily responsible for:

- Natural-language explanations
- RAG-grounded answers
- Short business interpretation

## 16. Agent Architecture

### Knowledge questions

```text
Question
  ↓
Intent classification
  ↓
Hybrid retrieval
  ↓
Relevant project context
  ↓
Llama 3.2
  ↓
Grounded answer
```

### Budget questions

```text
Question
  ↓
Budget extraction
  ↓
Decision engine
  ↓
Deterministic allocation
  ↓
Llama 3.2 explanation
  ↓
Final answer
```

## 17. Management Interface

The final application uses Gradio.

Management does not need to open Jupyter Notebook.

Start the application by double-clicking:

`START AARANYA.bat`

## 18. Technology Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL |
| Data processing | Python / Pandas |
| Statistical modeling | Statsmodels / Scikit-learn |
| Optimization | SciPy |
| Retrieval | Scikit-learn TF-IDF |
| Local LLM | Ollama / Llama 3.2 |
| Interface | Gradio |
| Development | Jupyter / Anaconda |

## 19. Key Design Principle

> Use AI for interaction and explanation, but deterministic analytics for numerical decisions.

The LLM does not replace the analytics layer.

Instead:

```text
Analytics → Evidence
Optimization → Numerical scenario
RAG → Project context
LLM → Explanation
```

## 20. Important Limitations

- Attribution is observational rather than causal.
- MMM estimates depend on historical data and model assumptions.
- The optimizer produces a modeled scenario rather than a guaranteed outcome.
- P95 is a historical spending constraint, not an optimality guarantee.
- Different model specifications can produce different estimates.

## 21. Project Structure

```text
aaranya-marketing-analytics/
├── app.py
├── aaranya_agent.py
├── db.py
├── generate_data.py
├── check_progress.py
├── START AARANYA.bat
├── README.md
├── data/
├── docs/
├── models/
├── notebooks/
├── powerbi/
└── solutions/
```

## 22. Final Business Question

> **Where should Aaranya spend its next ₹1 of marketing budget?**

Aaranya connects:

**Customer understanding → Marketing measurement → MMM → Response modeling → Optimization → AI decision interface**