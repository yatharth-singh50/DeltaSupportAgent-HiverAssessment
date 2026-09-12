import pandas as pd
import os

os.makedirs('data', exist_ok=True)
print("Loading twcs.csv...")
df = pd.read_csv('data/twcs.csv')

# Isolate Delta's tweets
delta_tweets = df[df['author_id'] == 'Delta']

# Match Delta's replies back to the inbound customer tweets
conversations = pd.merge(
    df[df['inbound'] == True], 
    delta_tweets, 
    left_on='tweet_id', 
    right_on='in_response_to_tweet_id', 
    suffixes=('_customer', '_delta')
)

clean_conversations = conversations[['text_customer', 'text_delta']].dropna()
print(f"Total Delta conversations: {len(clean_conversations)}")

# Sample 200 rows for the Golden Evaluation Set
golden_set = clean_conversations.sample(n=200, random_state=42).copy()
golden_set['ground_truth_intent'] = ""
golden_set['escalate_boolean'] = ""
golden_set['escalation_reason'] = ""

golden_set.to_csv('data/delta_golden_set_unlabeled.csv', index=False)
print("Exported 200 rows to data/delta_golden_set_unlabeled.csv")