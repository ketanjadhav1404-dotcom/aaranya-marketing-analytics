
from pathlib import Path
import copy
import re

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_FOLDER = PROJECT_ROOT / "models"


# ============================================================
# 2. LOAD SAVED MODEL / RAG STATE
# ============================================================

rag_state = joblib.load(MODELS_FOLDER / "rag_state.pkl")
optimizer_state = joblib.load(MODELS_FOLDER / "optimizer_state.pkl")


# RAG objects
chunks = rag_state["chunks"]
word_vectorizer = rag_state["word_vectorizer"]
char_vectorizer = rag_state["char_vectorizer"]
word_matrix = rag_state["word_matrix"]
char_matrix = rag_state["char_matrix"]
intent_boosts = rag_state["intent_boosts"]


# Optimizer objects
channels = optimizer_state["channels"]
decision_baseline = optimizer_state["decision_baseline"]
latest_row = optimizer_state["latest_row"]
decay = optimizer_state["decay"]
saturation_K = optimizer_state["saturation_K"]
saturated_mmm = optimizer_state["saturated_mmm"]
weekly_mmm = optimizer_state["weekly_mmm"]


# ============================================================
# 3. QUESTION INTENT CLASSIFIER
# ============================================================

def classify_question(question):
    q = question.lower()

    if any(
        word in q
        for word in [
            "caus",
            "cause",
            "incremental",
            "attribution",
            "did meta",
            "did google",
            "did campaign",
        ]
    ):
        return "causality"

    if any(
        word in q
        for word in [
            "p95",
            "ceiling",
            "upper bound",
            "constraint",
            "spending limit",
            "spend limit",
        ]
    ):
        return "constraints"

    if any(
        word in q
        for word in [
            "adstock",
            "carryover",
            "decay",
        ]
    ):
        return "adstock"

    if any(
        word in q
        for word in [
            "saturation",
            "diminishing",
            "marginal return",
        ]
    ):
        return "response_curve"

    if any(
        word in q
        for word in [
            "budget",
            "allocate",
            "allocation",
            "next ₹",
            "decision engine",
            "optimizer",
        ]
    ):
        return "optimization"

    return "general"


# ============================================================
# 4. RAG RETRIEVAL
# ============================================================

def retrieve_knowledge(question, top_k=3):

    intent = classify_question(question)

    query_word = word_vectorizer.transform([question])
    query_char = char_vectorizer.transform([question])

    word_scores = cosine_similarity(
        query_word,
        word_matrix
    ).flatten()

    char_scores = cosine_similarity(
        query_char,
        char_matrix
    ).flatten()

    scores = (
        0.70 * word_scores
        + 0.30 * char_scores
    )

    relevant_sections = intent_boosts.get(intent, [])

    for i, chunk in enumerate(chunks):
        if chunk["title"] in relevant_sections:
            scores[i] += 0.10

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append(
            {
                "chunk_id": chunks[idx]["chunk_id"],
                "title": chunks[idx]["title"],
                "text": chunks[idx]["text"],
                "score": float(scores[idx]),
            }
        )

    return results


# ============================================================
# 5. LOCAL RAG + LLAMA
# ============================================================

def answer_with_local_rag(question, top_k=3):

    retrieved = retrieve_knowledge(
        question,
        top_k=top_k
    )

    context_parts = []

    for i, result in enumerate(retrieved, 1):

        context_parts.append(
            f"[SOURCE {i}]\n"
            f"Section: {result['title']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are the Aaranya Marketing Analytics AI assistant.

Answer the user's question using ONLY the supplied
Aaranya project context.

Rules:
- Do not invent Aaranya facts.
- Do not invent numbers.
- If the context does not contain enough information, say so.
- Distinguish observed data, model assumptions, and model outputs.
- Never describe observational attribution as causal.
- Keep the answer concise and business-oriented.
"""

    user_prompt = f"""
AARANYA PROJECT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return {
        "question": question,
        "intent": classify_question(question),
        "retrieved_sources": [
            {
                "title": r["title"],
                "score": r["score"],
            }
            for r in retrieved
        ],
        "answer": response.message.content,
    }


# ============================================================
# 6. NEXT-PERIOD MMM PREDICTION
# ============================================================

def next_period_prediction(spend_vector):

    scenario = {}

    for i, channel in enumerate(channels):

        previous_adstock = latest_row[
            f"{channel}_adstock_0.50"
        ]

        next_adstock = (
            spend_vector[i]
            + decay * previous_adstock
        )

        scenario[
            f"{channel}_adstock_0.50"
        ] = next_adstock

        K = saturation_K[channel]

        scenario[
            f"{channel}_sat_0.50"
        ] = (
            next_adstock /
            (next_adstock + K)
        )

    scenario_df = pd.DataFrame([scenario])

    return saturated_mmm.predict(
        scenario_df
    )[0]


# ============================================================
# 7. BUDGET EXTRACTION
# ============================================================

def extract_budget(question):

    q = question.lower()

    # ₹1 lakh / 1 lakh / rs 1 lakh
    lakh_match = re.search(
        r"(?:₹|rs\.?|inr)?\s*([\d,.]+)\s*lakh",
        q
    )

    if lakh_match:

        value = float(
            lakh_match.group(1).replace(",", "")
        )

        return value * 100000

    # ₹50k / 50k
    k_match = re.search(
        r"(?:₹|rs\.?|inr)?\s*([\d,.]+)\s*k\b",
        q
    )

    if k_match:

        value = float(
            k_match.group(1).replace(",", "")
        )

        return value * 1000

    # ₹50,000 / Rs 50,000 / 50000
    number_matches = re.findall(
        r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)",
        q
    )

    for match in number_matches:

        value = float(
            match.replace(",", "")
        )

        if value >= 1000:
            return value

    return None


# ============================================================
# 8. CLEAN OBJECT FOR RETURN / DISPLAY
# ============================================================

def clean_for_llm(obj):

    if isinstance(obj, dict):

        return {
            k: clean_for_llm(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            clean_for_llm(v)
            for v in obj
        ]

    if isinstance(obj, tuple):

        return tuple(
            clean_for_llm(v)
            for v in obj
        )

    if isinstance(obj, np.generic):

        return obj.item()

    if isinstance(obj, pd.DataFrame):

        return obj.to_dict(
            orient="records"
        )

    if isinstance(obj, pd.Series):

        return obj.to_dict()

    return obj


# ============================================================
# 9. DETERMINISTIC BUDGET DECISION ENGINE
# ============================================================

def aaranya_decision_engine(
    incremental_budget,
    upper_percentile=0.95
):

    upper_bounds = np.array(
        [
            weekly_mmm[channel].quantile(
                upper_percentile
            )
            for channel in channels
        ]
    )

    baseline = decision_baseline.copy()

    baseline_total = baseline.sum()

    maximum_increment = (
        upper_bounds.sum()
        - baseline_total
    )

    if incremental_budget < 0:

        raise ValueError(
            "Incremental budget cannot be negative."
        )

    if incremental_budget > maximum_increment:

        raise ValueError(
            f"Budget exceeds the available "
            f"P95 capacity of ₹{maximum_increment:,.0f}."
        )

    target_total = (
        baseline_total
        + incremental_budget
    )

    x0 = baseline.copy()

    def objective(x):

        return -next_period_prediction(x)

    constraint = {
        "type": "eq",
        "fun": lambda x:
            x.sum() - target_total
    }

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[
            (
                baseline[i],
                upper_bounds[i]
            )
            for i in range(len(channels))
        ],
        constraints=constraint,
        options={
            "maxiter": 2000,
            "ftol": 1e-8,
        },
    )

    if not result.success:

        raise RuntimeError(
            result.message
        )

    optimized_spend = result.x

    baseline_cm = next_period_prediction(
        baseline
    )

    optimized_cm = next_period_prediction(
        optimized_spend
    )

    incremental_cm = (
        optimized_cm
        - baseline_cm
    )

    allocation = pd.DataFrame(
        {
            "Channel": channels,
            "Current Spend": baseline,
            "Recommended Spend": optimized_spend,
            "Incremental Allocation":
                optimized_spend - baseline,
            "P95 Ceiling": upper_bounds,
        }
    )

    if incremental_budget > 0:

        allocation[
            "Allocation Share"
        ] = (
            allocation[
                "Incremental Allocation"
            ]
            / incremental_budget
        )

    else:

        allocation[
            "Allocation Share"
        ] = 0

    allocation[
        "Remaining P95 Capacity"
    ] = (
        allocation["P95 Ceiling"]
        - allocation["Recommended Spend"]
    )

    return {
        "allocation": allocation,
        "baseline_weekly_budget": float(
            baseline_total
        ),
        "recommended_weekly_budget": float(
            optimized_spend.sum()
        ),
        "incremental_predicted_cm": float(
            incremental_cm
        ),
        "incremental_cm_per_rupee": float(
            incremental_cm / incremental_budget
        )
        if incremental_budget > 0
        else 0,
        "maximum_increment": float(
            maximum_increment
        ),
    }


# ============================================================
# 10. MAIN AARANYA AGENT
# ============================================================

def run_aaranya_agent(question):

    intent = classify_question(
        question
    )

    # --------------------------------------------------------
    # Budget optimization route
    # --------------------------------------------------------

    if intent == "optimization":

        budget = extract_budget(
            question
        )

        if budget is None:

            return {
                "route": "decision_engine",
                "intent": intent,
                "answer": (
                    "I can run the Aaranya decision engine, "
                    "but I need the additional budget amount. "
                    "For example: "
                    "'I have ₹50,000 additional budget.'"
                ),
            }

        decision = aaranya_decision_engine(
            budget
        )

        allocation_df = (
            decision["allocation"].copy()
        )

        allocation_lines = []

        for _, row in allocation_df.iterrows():

            allocation_lines.append(
                f"- {row['Channel']}: "
                f"₹{row['Incremental Allocation']:,.2f} "
                f"({row['Allocation Share'] * 100:.2f}%)"
            )

        allocation_text = "\n".join(
            allocation_lines
        )

        explanation_text = (
            "This is a modeled allocation based on the Aaranya "
            "response model and subject to the historical P95 "
            "spending ceiling. The contribution-margin estimate "
            "is not guaranteed."
        )

        final_answer = (
            f"For an additional ₹{budget:,.0f} "
            f"marketing budget, the Aaranya "
            f"decision engine recommends:\n\n"
            f"{allocation_text}\n\n"
            f"**Latest observed weekly budget:** "
            f"₹{decision['baseline_weekly_budget']:,.2f}\n\n"
            f"**Recommended weekly budget:** "
            f"₹{decision['recommended_weekly_budget']:,.2f}\n\n"
            f"**Modeled incremental contribution margin:** "
            f"₹{decision['incremental_predicted_cm']:,.2f}\n\n"
            f"**Modeled incremental CM per rupee:** "
            f"₹{decision['incremental_cm_per_rupee']:,.2f}\n\n"
            f"{explanation_text}"
        )

        return {
            "route": "decision_engine",
            "intent": intent,
            "budget": budget,
            "decision": clean_for_llm(
                decision
            ),
            "answer": final_answer,
        }

    # --------------------------------------------------------
    # Knowledge / RAG route
    # --------------------------------------------------------

    result = answer_with_local_rag(
        question
    )

    return {
        "route": "rag",
        "intent": result["intent"],
        "retrieved_sources":
            result["retrieved_sources"],
        "answer": result["answer"],
    }
