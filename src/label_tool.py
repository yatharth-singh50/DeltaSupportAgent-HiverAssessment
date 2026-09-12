import os
import pandas as pd

input_path = "data/delta_golden_set_unlabeled.csv"
output_path = "data/delta_golden_set_labeled.csv"

source = output_path if os.path.exists(output_path) else input_path
df = pd.read_csv(source)

for col in ["ground_truth_intent", "escalation_reason"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(object).fillna("")

if "escalate_boolean" not in df.columns:
    df["escalate_boolean"] = ""
df["escalate_boolean"] = df["escalate_boolean"].astype(object).fillna("")

intents = {
    "1": "Flight_Status_and_Upgrades",
    "2": "Baggage_and_Amenities",
    "3": "Refunds_and_Complaints",
    "4": "Praise_and_Feedback",
    "5": "Needs_Context_or_DM",
}

# Find the first index that isn't labeled yet
i = 0
while i < len(df) and str(df.at[i, "ground_truth_intent"]).strip() != "":
    i += 1

print(f"Total rows: {len(df)}. Resuming from row [{i + 1}].")
print("Controls: [1-5] Choose intent | [b] Back to previous | [s] Skip | [q] Save & Quit\n")

while 0 <= i < len(df):
    row = df.iloc[i]
    print(f"\n--- [{i + 1}/{len(df)}] ---")
    print(f"Customer: {row['text_customer']}")
    if str(row["ground_truth_intent"]).strip():
        print(f"(Current: Intent={row['ground_truth_intent']}, Escalate={row['escalate_boolean']}, Reason={row['escalation_reason']})")

    choice = input("Select [1-5, b, s, q]: ").strip().lower()

    if choice == "q":
        break
    elif choice == "b":
        if i > 0:
            i -= 1
        else:
            print("Already at the first row.")
        continue
    elif choice == "s":
        i += 1
        continue
    elif choice in intents:
        intent = intents[choice]
        esc_in = input("Escalate? (y/n, default n): ").strip().lower()
        esc = esc_in == "y"
        
        if esc:
            reason = input("Escalation Reason: ").strip()
            if not reason:
                reason = "General escalation required"
        else:
            reason = "None"

        df.at[i, "ground_truth_intent"] = intent
        df.at[i, "escalate_boolean"] = esc
        df.at[i, "escalation_reason"] = reason

        df.to_csv(output_path, index=False)
        i += 1
    else:
        print("Invalid input. Try again.")

print(f"\nProgress saved to {output_path}")