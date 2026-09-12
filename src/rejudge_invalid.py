import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

MODEL_ID = "openai/gpt-oss-20b"

EVALUATION_PATH = "data/evaluation_results.csv"
HUMAN_SAMPLE_PATH = "data/human_judge_samples.csv"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------

def judge_reply(customer_message, historical_response, agent_reply):
    """
    Re-evaluate one generated reply using the same 1-5 evaluation idea
    used in the main evaluation pipeline.
    """

    prompt = f"""
You are evaluating an AI customer-support reply for Delta Airlines.

Evaluate the AGENT REPLY using the CUSTOMER MESSAGE and the
HISTORICAL DELTA RESPONSE as evidence.

Consider:

1. Groundedness
   - Is the reply supported by the historical response/context?
   - Penalize invented facts, policies, timelines, promises, or actions.

2. Helpfulness
   - Does the reply meaningfully address the customer's issue?

3. Tone
   - Is the reply professional, natural, polite, and appropriate?

4. Safety
   - Does the reply avoid misleading, risky, or overconfident guidance?

5. Overall quality
   - Considering all criteria, how good is the reply?

Score the reply from 1 to 5:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Return STRICT JSON only:

{{
    "score": 1,
    "reason": "brief explanation"
}}

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL DELTA RESPONSE:
{historical_response}

AGENT REPLY:
{agent_reply}
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of customer-support AI responses. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Remove accidental markdown fences if the model returns them.
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    result = json.loads(raw)

    score = int(result["score"])
    reason = str(result.get("reason", ""))

    if score < 1 or score > 5:
        raise ValueError(
            f"Judge returned invalid score {score}. Expected 1-5."
        )

    return score, reason


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not os.path.exists(EVALUATION_PATH):
        print(f"ERROR: Could not find {EVALUATION_PATH}")
        return

    evaluation_df = pd.read_csv(EVALUATION_PATH)

    if "llm_judge_score" not in evaluation_df.columns:
        print("ERROR: evaluation_results.csv has no llm_judge_score column.")
        return

    # Find invalid zero scores.
    invalid_mask = evaluation_df["llm_judge_score"] == 0

    invalid_rows = evaluation_df[invalid_mask]

    print("=" * 80)
    print("INVALID LLM JUDGE REPAIR")
    print("=" * 80)

    print(f"\nInvalid zero-score rows found: {len(invalid_rows)}")

    if len(invalid_rows) == 0:
        print("\nNothing to repair.")
        return

    # Load human sample file if available.
    human_df = None

    if os.path.exists(HUMAN_SAMPLE_PATH):
        human_df = pd.read_csv(HUMAN_SAMPLE_PATH)

    # -----------------------------------------------------------------
    # Re-judge each invalid row
    # -----------------------------------------------------------------

    for index, row in invalid_rows.iterrows():

        customer = row["text_customer"]
        historical = row["text_delta"]
        reply = row["pred_reply"]

        print("\n" + "-" * 80)
        print(f"Evaluation row: {index}")
        print("-" * 80)

        print("\nCUSTOMER:")
        print(customer)

        print("\nHISTORICAL RESPONSE:")
        print(historical)

        print("\nAGENT REPLY:")
        print(reply)

        print("\nRe-judging...")

        try:
            new_score, reason = judge_reply(
                customer,
                historical,
                reply,
            )

        except Exception as exc:
            print(f"\nFAILED: {exc}")
            continue

        print(f"\nNew judge score: {new_score}/5")
        print(f"Reason: {reason}")

        # -------------------------------------------------------------
        # Update main evaluation results
        # -------------------------------------------------------------

        evaluation_df.at[index, "llm_judge_score"] = new_score

        # Save immediately.
        evaluation_df.to_csv(
            EVALUATION_PATH,
            index=False
        )

        print("\nUpdated evaluation_results.csv")

        # -------------------------------------------------------------
        # Update human judge sample file too
        # -------------------------------------------------------------

        if human_df is not None:

            # Match using customer message + generated reply.
            match = (
                (human_df["text_customer"] == customer)
                &
                (human_df["pred_reply"] == reply)
            )

            matches = int(match.sum())

            if matches == 1:

                matched_index = human_df.index[match][0]

                human_df.at[
                    matched_index,
                    "llm_judge_score"
                ] = new_score

                human_df.to_csv(
                    HUMAN_SAMPLE_PATH,
                    index=False
                )

                print("Updated human_judge_samples.csv")

            elif matches == 0:

                print(
                    "No matching row found in human_judge_samples.csv."
                )

            else:

                print(
                    "WARNING: Multiple matching rows found in "
                    "human_judge_samples.csv; no automatic update performed."
                )

    print("\n" + "=" * 80)
    print("RE-JUDGE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()