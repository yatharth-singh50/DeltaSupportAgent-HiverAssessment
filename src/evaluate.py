import os
import json
import time
import random
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from agent import DemoMode, client, MODEL_ID
from baselines import TrivialBaseline, SimpleBaseline


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = "data/delta_golden_set_labeled.csv"
RESULTS_PATH = "data/evaluation_results.csv"
SUMMARY_PATH = "data/evaluation_summary.json"

MAX_RETRIES = 3
SLEEP_BETWEEN_ROWS = 0.5

INTENTS = [
    "Flight_Status_and_Upgrades",
    "Baggage_and_Amenities",
    "Refunds_and_Complaints",
    "Praise_and_Feedback",
    "Needs_Context_or_DM",
]


# ============================================================
# LOAD / INITIALIZE RESULTS
# ============================================================

print("=" * 70)
print(" DELTA AI SUPPORT AGENT - EVALUATION HARNESS")
print("=" * 70)

golden_df = pd.read_csv(GOLDEN_PATH)

print(f"\nGolden set size: {len(golden_df)}")

if os.path.exists(RESULTS_PATH):
    print(f"Existing results found: {RESULTS_PATH}")
    print("Resuming from previously completed evaluation...")
    df = pd.read_csv(RESULTS_PATH)

    # Safety check: make sure row count matches.
    if len(df) != len(golden_df):
        raise RuntimeError(
            f"Existing results contain {len(df)} rows, "
            f"but golden set contains {len(golden_df)} rows."
        )

else:
    print("No previous results found. Starting fresh.")
    df = golden_df.copy()


# Make sure all evaluation columns exist.
EVALUATION_COLUMNS = [
    "pred_intent",
    "pred_escalate",
    "pred_reason",
    "pred_reply",
    "pipeline_status",
    "pipeline_error",
    "llm_judge_score",
    "llm_judge_reasoning",
]

for col in EVALUATION_COLUMNS:
    if col not in df.columns:
        df[col] = None


# ============================================================
# HELPERS
# ============================================================

def is_missing(value):
    """
    Robustly determine whether a dataframe cell is empty.
    """
    if pd.isna(value):
        return True

    if isinstance(value, str):
        return value.strip() == ""

    return False


def save_results():
    """
    Save progress immediately so API interruptions don't destroy work.
    """
    df.to_csv(RESULTS_PATH, index=False)


def generate_judge_with_retry(messages, max_retries=MAX_RETRIES):
    """
    Call Groq for the LLM judge with retry handling.
    """

    transient_errors = (
        "429",
        "500",
        "502",
        "503",
        "504",
    )

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            return response.choices[0].message.content

        except Exception as e:

            error_text = str(e)

            is_transient = any(
                code in error_text
                for code in transient_errors
            )

            if not is_transient or attempt == max_retries - 1:
                raise

            delay = (2 ** attempt) + random.uniform(0, 1)

            print(
                f"   Judge API error "
                f"(attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {delay:.1f}s..."
            )

            time.sleep(delay)


# ============================================================
# LLM-AS-JUDGE
# ============================================================

JUDGE_PROMPT = """
You are a quality assurance judge for Delta Airlines customer service.

Review the customer message, the historical context available to
the AI agent, and the AI agent's drafted reply.

Score the reply from 1 to 5 using this rubric:

5 = Excellent
    Correct, helpful, relevant, professional, grounded in the provided
    historical context, and safe.

4 = Good
    Helpful and appropriate with only minor omissions or weaknesses.

3 = Acceptable
    Partially useful but has noticeable issues in completeness,
    relevance, grounding, or tone.

2 = Poor
    Significant problems such as weak grounding, poor usefulness,
    or inappropriate assumptions.

1 = Unacceptable
    Incorrect, misleading, unsafe, or substantially unrelated.

Evaluate these dimensions:

1. Groundedness:
   Does the reply avoid inventing policies or unsupported claims
   beyond the historical context?

2. Helpfulness:
   Does it meaningfully address the customer's request?

3. Tone:
   Is it professional, concise, and empathetic?

4. Safety:
   Does it avoid asking for unnecessary private information
   and avoid making unsafe commitments?

Return ONLY valid JSON:

{
    "score": integer from 1 to 5,
    "reasoning": "short explanation"
}
"""


def grade_reply(customer_msg, historical_context, draft_reply):

    if not draft_reply:
        return 0, "No reply drafted."

    prompt = f"""
Customer Message:
{customer_msg}

Historical Context Provided To Agent:
{historical_context}

AI Agent Reply:
{draft_reply}
"""

    messages = [
        {
            "role": "system",
            "content": JUDGE_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:

        raw_response = generate_judge_with_retry(messages)

        result = json.loads(raw_response)

        score = int(result.get("score", 0))
        reasoning = result.get(
            "reasoning",
            "No reasoning provided."
        )

        # Safety check.
        if score < 1 or score > 5:
            return 0, "Judge returned an invalid score."

        return score, reasoning

    except Exception as e:

        return 0, f"Judge API error: {str(e)}"


# ============================================================
# MAIN AGENT EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: AGENT PREDICTIONS")
print("=" * 70)

agent_completed = 0
agent_skipped = 0
agent_failed = 0

for i in range(len(df)):

    row = df.iloc[i]

    customer_msg = str(row["text_customer"])

    has_intent = not is_missing(row["pred_intent"])
    has_escalation = not is_missing(row["pred_escalate"])

    # --------------------------------------------------------
    # Skip rows that already have the agent prediction.
    # --------------------------------------------------------

    if has_intent and has_escalation:

        agent_skipped += 1

        if (i + 1) % 20 == 0:
            print(
                f"[{i + 1}/{len(df)}] "
                f"Already evaluated - skipping."
            )

        continue

    # --------------------------------------------------------
    # Run agent only for incomplete rows.
    # --------------------------------------------------------

    print(
        f"\n[{i + 1}/{len(df)}] "
        f"Evaluating: {customer_msg[:100]}"
    )

    try:

        decision = DemoMode(customer_msg)

        df.at[i, "pred_intent"] = decision.get("intent")
        df.at[i, "pred_escalate"] = decision.get(
            "escalate_boolean"
        )
        df.at[i, "pred_reason"] = decision.get(
            "escalation_reason"
        )
        df.at[i, "pred_reply"] = decision.get(
            "generated_reply"
        )
        df.at[i, "pipeline_status"] = decision.get(
            "status",
            "unknown"
        )

        df.at[i, "pipeline_error"] = decision.get(
            "error"
        )

        agent_completed += 1

        # SAVE IMMEDIATELY.
        save_results()

    except Exception as e:

        print(f"   Agent evaluation failed: {e}")

        df.at[i, "pipeline_status"] = "evaluation_error"
        df.at[i, "pipeline_error"] = str(e)

        agent_failed += 1

        save_results()

    time.sleep(SLEEP_BETWEEN_ROWS)


print("\nAgent evaluation phase complete.")
print(f"New evaluations:       {agent_completed}")
print(f"Previously completed:  {agent_skipped}")
print(f"Failed evaluations:    {agent_failed}")


# ============================================================
# PHASE 2: LLM-AS-JUDGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: LLM-AS-JUDGE")
print("=" * 70)

judge_completed = 0
judge_skipped = 0
judge_failed = 0

for i in range(len(df)):

    row = df.iloc[i]

    reply = row["pred_reply"]
    escalate = row["pred_escalate"]

    # --------------------------------------------------------
    # No reply means there is nothing for the judge to score.
    # --------------------------------------------------------

    if is_missing(reply):

        judge_skipped += 1
        continue

    # --------------------------------------------------------
    # Don't judge escalated cases.
    # --------------------------------------------------------

    if str(escalate).strip().lower() == "true":

        judge_skipped += 1
        continue

    # --------------------------------------------------------
    # If judge score already exists, preserve it.
    # --------------------------------------------------------

    if not is_missing(row["llm_judge_score"]):

        judge_skipped += 1
        continue

    customer_msg = str(row["text_customer"])

    print(
        f"\n[{i + 1}/{len(df)}] "
        f"Judging reply: {customer_msg[:100]}"
    )

    try:

        # Reconstruct the same historical context used by the
        # agent's RAG pipeline.
        try:
            historical_context = (
                __import__("agent")
                .get_historical_context(customer_msg)
            )
        except Exception as e:
            historical_context = (
                f"Historical context unavailable: {e}"
            )

        score, reasoning = grade_reply(
            customer_msg,
            historical_context,
            str(reply),
        )

        if score > 0:

            df.at[i, "llm_judge_score"] = score
            df.at[i, "llm_judge_reasoning"] = reasoning

            judge_completed += 1

        else:

            df.at[i, "llm_judge_reasoning"] = reasoning
            judge_failed += 1

        # SAVE IMMEDIATELY.
        save_results()

    except Exception as e:

        print(f"   Judge failed: {e}")

        df.at[i, "llm_judge_reasoning"] = (
            f"Judge evaluation error: {e}"
        )

        judge_failed += 1

        save_results()

    time.sleep(SLEEP_BETWEEN_ROWS)


print("\nJudge phase complete.")
print(f"New judge scores:      {judge_completed}")
print(f"Previously scored:     {judge_skipped}")
print(f"Failed judge calls:    {judge_failed}")


# ============================================================
# PHASE 3: FINAL METRICS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: FINAL METRICS")
print("=" * 70)


# ------------------------------------------------------------
# Determine valid agent predictions.
# ------------------------------------------------------------

valid_df = df[
    (~df["pred_intent"].isna()) &
    (~df["pred_escalate"].isna())
].copy()

print(
    f"\nCompleted agent predictions: "
    f"{len(valid_df)}/{len(df)}"
)


# ============================================================
# INTENT METRICS
# ============================================================

y_true_intent = valid_df["ground_truth_intent"].astype(str)
y_pred_intent = valid_df["pred_intent"].astype(str)

intent_accuracy = accuracy_score(
    y_true_intent,
    y_pred_intent,
)

intent_macro_f1 = f1_score(
    y_true_intent,
    y_pred_intent,
    average="macro",
    zero_division=0,
)

intent_weighted_f1 = f1_score(
    y_true_intent,
    y_pred_intent,
    average="weighted",
    zero_division=0,
)

print("\n--- INTENT CLASSIFICATION ---")

print(
    f"Accuracy:       {intent_accuracy:.2%}"
)

print(
    f"Macro F1:       {intent_macro_f1:.2%}"
)

print(
    f"Weighted F1:    {intent_weighted_f1:.2%}"
)


print("\nPer-class metrics:")

intent_report = classification_report(
    y_true_intent,
    y_pred_intent,
    labels=INTENTS,
    output_dict=True,
    zero_division=0,
)

for intent in INTENTS:

    metrics = intent_report.get(intent, {})

    print(
        f"  {intent:<30} "
        f"Precision={metrics.get('precision', 0):.2%} "
        f"Recall={metrics.get('recall', 0):.2%} "
        f"F1={metrics.get('f1-score', 0):.2%}"
    )


# ============================================================
# INTENT CONFUSION MATRIX
# ============================================================

intent_cm = confusion_matrix(
    y_true_intent,
    y_pred_intent,
    labels=INTENTS,
)

print("\nIntent Confusion Matrix:")
print("Rows = actual, Columns = predicted")

cm_df = pd.DataFrame(
    intent_cm,
    index=INTENTS,
    columns=INTENTS,
)

print(cm_df.to_string())


# ============================================================
# ESCALATION METRICS
# ============================================================

def normalize_bool(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() == "true"


y_true_esc = valid_df[
    "escalate_boolean"
].apply(normalize_bool)

y_pred_esc = valid_df[
    "pred_escalate"
].apply(normalize_bool)


escalation_accuracy = accuracy_score(
    y_true_esc,
    y_pred_esc,
)

escalation_precision = precision_score(
    y_true_esc,
    y_pred_esc,
    zero_division=0,
)

escalation_recall = recall_score(
    y_true_esc,
    y_pred_esc,
    zero_division=0,
)

escalation_f1 = f1_score(
    y_true_esc,
    y_pred_esc,
    zero_division=0,
)


print("\n--- ESCALATION ---")

print(
    f"Accuracy:       {escalation_accuracy:.2%}"
)

print(
    f"Precision:      {escalation_precision:.2%}"
)

print(
    f"Recall:         {escalation_recall:.2%}"
)

print(
    f"F1:             {escalation_f1:.2%}"
)


# ------------------------------------------------------------
# Escalation confusion matrix.
# ------------------------------------------------------------

esc_cm = confusion_matrix(
    y_true_esc,
    y_pred_esc,
    labels=[False, True],
)

print("\nEscalation Confusion Matrix:")
print("Rows = actual, Columns = predicted")

esc_cm_df = pd.DataFrame(
    esc_cm,
    index=["Actual False", "Actual True"],
    columns=["Pred False", "Pred True"],
)

print(esc_cm_df.to_string())


# ============================================================
# LLM JUDGE METRICS
# ============================================================

judge_df = df[
    ~df["llm_judge_score"].isna()
].copy()

if len(judge_df) > 0:

    judge_scores = pd.to_numeric(
        judge_df["llm_judge_score"],
        errors="coerce",
    ).dropna()

    judge_mean = judge_scores.mean()
    judge_median = judge_scores.median()

    print("\n--- LLM-AS-JUDGE ---")

    print(
        f"Replies scored:  {len(judge_scores)}"
    )

    print(
        f"Mean score:      {judge_mean:.2f} / 5"
    )

    print(
        f"Median score:    {judge_median:.2f} / 5"
    )

else:

    judge_mean = None
    judge_median = None

    print("\nNo LLM judge scores available.")


# ============================================================
# BASELINE EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4: BASELINE COMPARISON")
print("=" * 70)


baseline_records = []

for i, row in df.iterrows():

    customer_msg = row["text_customer"]

    # --------------------------------------------------------
    # Trivial baseline
    # --------------------------------------------------------

    trivial = TrivialBaseline(customer_msg)

    # --------------------------------------------------------
    # Simple keyword baseline
    # --------------------------------------------------------

    simple = SimpleBaseline(customer_msg)

    baseline_records.append(
        {
            "text_customer": customer_msg,

            "ground_truth_intent": row[
                "ground_truth_intent"
            ],

            "ground_truth_escalate": normalize_bool(
                row["escalate_boolean"]
            ),

            "trivial_intent": trivial["intent"],
            "trivial_escalate": trivial[
                "escalate_boolean"
            ],

            "simple_intent": simple["intent"],
            "simple_escalate": simple[
                "escalate_boolean"
            ],
        }
    )


baseline_df = pd.DataFrame(baseline_records)


# ------------------------------------------------------------
# Helper for baseline metrics.
# ------------------------------------------------------------

def calculate_baseline_metrics(
    name,
    true_intent,
    pred_intent,
    true_esc,
    pred_esc,
):

    return {
        "intent_accuracy": accuracy_score(
            true_intent,
            pred_intent,
        ),

        "intent_macro_f1": f1_score(
            true_intent,
            pred_intent,
            average="macro",
            zero_division=0,
        ),

        "escalation_accuracy": accuracy_score(
            true_esc,
            pred_esc,
        ),

        "escalation_precision": precision_score(
            true_esc,
            pred_esc,
            zero_division=0,
        ),

        "escalation_recall": recall_score(
            true_esc,
            pred_esc,
            zero_division=0,
        ),

        "escalation_f1": f1_score(
            true_esc,
            pred_esc,
            zero_division=0,
        ),
    }


true_baseline_intent = baseline_df[
    "ground_truth_intent"
].astype(str)

true_baseline_esc = baseline_df[
    "ground_truth_escalate"
].apply(normalize_bool)


trivial_metrics = calculate_baseline_metrics(
    "Trivial Baseline",
    true_baseline_intent,
    baseline_df["trivial_intent"],
    true_baseline_esc,
    baseline_df["trivial_escalate"],
)


simple_metrics = calculate_baseline_metrics(
    "Simple Keyword Baseline",
    true_baseline_intent,
    baseline_df["simple_intent"],
    true_baseline_esc,
    baseline_df["simple_escalate"],
)


print("\nBaseline comparison:")

print(
    f"\nTrivial Baseline"
    f"\n  Intent Accuracy:    "
    f"{trivial_metrics['intent_accuracy']:.2%}"
    f"\n  Intent Macro F1:    "
    f"{trivial_metrics['intent_macro_f1']:.2%}"
    f"\n  Escalation Accuracy:"
    f" {trivial_metrics['escalation_accuracy']:.2%}"
)

print(
    f"\nSimple Keyword Baseline"
    f"\n  Intent Accuracy:    "
    f"{simple_metrics['intent_accuracy']:.2%}"
    f"\n  Intent Macro F1:    "
    f"{simple_metrics['intent_macro_f1']:.2%}"
    f"\n  Escalation Accuracy:"
    f" {simple_metrics['escalation_accuracy']:.2%}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "dataset": {
        "golden_set_size": len(df),
        "completed_agent_predictions": len(valid_df),
        "completion_rate": (
            len(valid_df) / len(df)
            if len(df) > 0
            else 0
        ),
    },

    "agent": {
        "intent_accuracy": intent_accuracy,
        "intent_macro_f1": intent_macro_f1,
        "intent_weighted_f1": intent_weighted_f1,

        "escalation_accuracy": escalation_accuracy,
        "escalation_precision": escalation_precision,
        "escalation_recall": escalation_recall,
        "escalation_f1": escalation_f1,

        "llm_judge_replies_scored": (
            len(judge_df)
        ),

        "llm_judge_mean": judge_mean,
        "llm_judge_median": judge_median,
    },

    "baselines": {
        "trivial": trivial_metrics,
        "simple_keyword": simple_metrics,
    },

    "intent_per_class": {
        intent: {
            "precision": intent_report[
                intent
            ].get("precision", 0),

            "recall": intent_report[
                intent
            ].get("recall", 0),

            "f1": intent_report[
                intent
            ].get("f1-score", 0),

            "support": intent_report[
                intent
            ].get("support", 0),
        }

        for intent in INTENTS
    },

    "evaluation_run": {
        "new_agent_evaluations": agent_completed,
        "skipped_existing_agent": agent_skipped,
        "agent_failures": agent_failed,

        "new_judge_scores": judge_completed,
        "skipped_existing_judge": judge_skipped,
        "judge_failures": judge_failed,
    },
}


with open(SUMMARY_PATH, "w", encoding="utf-8") as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL SAVE
# ============================================================

save_results()


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nResults CSV:  {RESULTS_PATH}"
)

print(
    f"Summary JSON: {SUMMARY_PATH}"
)

print(
    "\nNo previously completed agent predictions "
    "or judge scores were overwritten."
)

print("=" * 70)