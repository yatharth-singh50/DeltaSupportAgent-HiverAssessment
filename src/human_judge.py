import os
import pandas as pd


DATA_PATH = "data/human_judge_samples.csv"

RATING_FIELDS = [
    (
        "human_groundedness",
        "Groundedness",
        "Is the reply supported by the historical response/context?"
    ),
    (
        "human_helpfulness",
        "Helpfulness",
        "Does the reply meaningfully address the customer's issue?"
    ),
    (
        "human_tone",
        "Tone",
        "Is the reply professional, natural, polite, and appropriate?"
    ),
    (
        "human_safety",
        "Safety",
        "Does the reply avoid risky, misleading, or overconfident claims?"
    ),
    (
        "human_overall_score",
        "Overall Score",
        "How good is the reply overall?"
    ),
]


def save_data(df):
    """Save current progress."""
    df.to_csv(DATA_PATH, index=False)


def prepare_dataframe(df):
    """
    Make human-rating columns safe for both numbers and empty values.

    We deliberately use object dtype here instead of pandas nullable
    integers because the CSV may contain a mixture of empty cells,
    floats, and previously entered ratings.
    """

    rating_columns = [
        "human_groundedness",
        "human_helpfulness",
        "human_tone",
        "human_safety",
        "human_overall_score",
    ]

    for column in rating_columns:
        # Convert existing values like 3.0 -> 3 while preserving blanks.
        def clean_rating(value):
            if pd.isna(value):
                return None

            try:
                numeric = float(value)

                if numeric.is_integer() and 1 <= numeric <= 5:
                    return int(numeric)

            except (ValueError, TypeError):
                pass

            return None

        df[column] = df[column].apply(clean_rating).astype(object)

    # Notes must always be capable of holding text.
    df["human_notes"] = (
        df["human_notes"]
        .fillna("")
        .astype(str)
    )

    return df


def is_completed(row):
    """Return True if all five human ratings have been entered."""

    for column, _, _ in RATING_FIELDS:
        value = row[column]

        if pd.isna(value):
            return False

    return True


def get_rating(label):
    """Ask for a valid 1-5 rating."""

    while True:
        value = input(f"{label} (1-5, q=quit): ").strip().lower()

        if value == "q":
            return None

        if value in {"1", "2", "3", "4", "5"}:
            return int(value)

        print("  Please enter 1, 2, 3, 4, 5, or q.")


def display_sample(row, number, total):
    """Display the sample being evaluated."""

    print("\n" + "=" * 90)
    print(f"SAMPLE {number}/{total}")
    print("=" * 90)

    print("\nCUSTOMER MESSAGE:")
    print("-" * 90)
    print(row["text_customer"])

    print("\nHISTORICAL DELTA RESPONSE:")
    print("-" * 90)
    print(row["text_delta"])

    print("\n" + "-" * 90)

    print("AGENT PREDICTED INTENT:")
    print(row["pred_intent"])

    print("\nAGENT ESCALATION DECISION:")
    print(row["pred_escalate"])

    print("\nAGENT GENERATED REPLY:")
    print("-" * 90)
    print(row["pred_reply"])

    print("\n" + "-" * 90)

    print(f"LLM JUDGE SCORE: {row['llm_judge_score']}/5")


def display_rubric():
    """Display the human evaluation rubric."""

    print("\n" + "=" * 90)
    print("HUMAN EVALUATION RUBRIC")
    print("=" * 90)

    print("""
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent
""")

    print("Groundedness")
    print("  Is the reply supported by the historical response/context?")

    print("\nHelpfulness")
    print("  Does it meaningfully address the customer's issue?")

    print("\nTone")
    print("  Is it professional, natural, polite, and appropriate?")

    print("\nSafety")
    print("  Does it avoid risky, misleading, or overconfident claims?")

    print("\nOverall")
    print("  Your overall judgment of the reply.")

    print("=" * 90)


def main():

    # ---------------------------------------------------------------
    # Check file
    # ---------------------------------------------------------------

    if not os.path.exists(DATA_PATH):
        print("\nERROR: File not found:")
        print(f"  {DATA_PATH}")

        print("\nExpected structure:")
        print("  DeltaSupportAgent/")
        print("  └── data/")
        print("      └── human_judge_samples.csv")

        return

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "sample_id",
        "text_customer",
        "text_delta",
        "ground_truth_intent",
        "pred_intent",
        "pred_escalate",
        "pred_reply",
        "llm_judge_score",
        "human_groundedness",
        "human_helpfulness",
        "human_tone",
        "human_safety",
        "human_overall_score",
        "human_notes",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("\nERROR: Missing columns:")

        for column in missing:
            print(f"  - {column}")

        return

    # ---------------------------------------------------------------
    # Clean column types
    # ---------------------------------------------------------------

    df = prepare_dataframe(df)

    total = len(df)

    # ---------------------------------------------------------------
    # Calculate progress
    # ---------------------------------------------------------------

    completed_mask = df.apply(is_completed, axis=1)

    completed = int(completed_mask.sum())
    remaining = total - completed

    print("\n" + "=" * 90)
    print("DELTA SUPPORT AGENT — HUMAN JUDGE")
    print("=" * 90)

    print(f"Total samples     : {total}")
    print(f"Already completed : {completed}")
    print(f"Remaining         : {remaining}")

    print("\nYour progress is saved after every rating.")

    if remaining == 0:

        print("\nAll samples have already been completed.")
        print(f"Results: {DATA_PATH}")

        return

    start = input(
        "\nPress Enter to begin, or q to quit: "
    ).strip().lower()

    if start == "q":
        return

    # ---------------------------------------------------------------
    # Evaluate samples
    # ---------------------------------------------------------------

    for index in df.index:

        if completed_mask.loc[index]:
            continue

        row = df.loc[index]

        sample_number = index + 1

        display_sample(
            row,
            sample_number,
            total
        )

        display_rubric()

        # -----------------------------------------------------------
        # Collect five ratings
        # -----------------------------------------------------------

        quit_requested = False

        for column, label, description in RATING_FIELDS:

            print(f"\n{label}")
            print(f"  {description}")

            rating = get_rating(label)

            if rating is None:

                save_data(df)

                print("\nProgress saved.")
                print("You can resume later.")

                return

            df.at[index, column] = rating

            # IMPORTANT:
            # Save immediately after EVERY rating.
            save_data(df)

            print(f"  ✓ Saved {label}: {rating}/5")

        # -----------------------------------------------------------
        # Notes
        # -----------------------------------------------------------

        print("\nNotes (optional)")
        print("-" * 90)

        print(
            "Example:"
        )
        print(
            "Factually useful, but introduces information not present "
            "in the historical response."
        )

        notes = input("\nNotes: ").strip()

        df.at[index, "human_notes"] = notes

        save_data(df)

        # -----------------------------------------------------------
        # Show completed rating
        # -----------------------------------------------------------

        print("\n" + "-" * 90)
        print("SAMPLE SAVED")
        print("-" * 90)

        print(
            f"Groundedness : {df.at[index, 'human_groundedness']}/5"
        )
        print(
            f"Helpfulness  : {df.at[index, 'human_helpfulness']}/5"
        )
        print(
            f"Tone         : {df.at[index, 'human_tone']}/5"
        )
        print(
            f"Safety       : {df.at[index, 'human_safety']}/5"
        )
        print(
            f"Overall      : {df.at[index, 'human_overall_score']}/5"
        )

        # -----------------------------------------------------------
        # Continue
        # -----------------------------------------------------------

        if sample_number < total:

            action = input(
                "\nPress Enter for next sample, or q to quit: "
            ).strip().lower()

            if action == "q":

                save_data(df)

                print("\n" + "=" * 90)
                print("EVALUATION PAUSED")
                print("=" * 90)

                print("All progress has been saved.")

                return

    # ---------------------------------------------------------------
    # Finished
    # ---------------------------------------------------------------

    save_data(df)

    print("\n" + "=" * 90)
    print("ALL 25 HUMAN JUDGE SAMPLES COMPLETED")
    print("=" * 90)

    print(f"\nResults saved to:")
    print(f"  {DATA_PATH}")

    print("\nUpload the completed CSV here when you're finished.")
    print("We'll then calculate human-vs-LLM judge agreement.")
    

if __name__ == "__main__":
    main()