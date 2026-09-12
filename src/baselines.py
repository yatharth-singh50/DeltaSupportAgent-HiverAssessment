import pandas as pd
import json

def TrivialBaseline(customer_text):
    """
    Trivial Baseline: Always predicts the majority class based on the golden set distribution.
    Intent: Praise_and_Feedback
    Escalate: True
    """
    return {
        "intent": "Praise_and_Feedback",
        "escalate_boolean": True,
        "escalation_reason": "Trivial baseline default escalation.",
        "generated_reply": None,
        "status": "success"
    }

def SimpleBaseline(customer_text):
    """
    Simple Baseline: Uses basic keyword heuristics for routing and escalation.
    Falls back to Praise_and_Feedback if no keywords match.
    """
    text_lower = str(customer_text).lower()
    
    # 1. Needs Context / DM
    if any(word in text_lower for word in ["dm", "sent you a message", "check inbox", "confirmation"]):
        return {
            "intent": "Needs_Context_or_DM",
            "escalate_boolean": True,
            "escalation_reason": "Customer mentions private message or confirmation code.",
            "generated_reply": None,
            "status": "success"
        }
        
    # 2. Refunds and Complaints (High friction)
    if any(word in text_lower for word in ["delay", "cancel", "refund", "rude", "stuck", "missed", "wait"]):
        return {
            "intent": "Refunds_and_Complaints",
            "escalate_boolean": True,
            "escalation_reason": "High-friction keywords detected.",
            "generated_reply": None,
            "status": "success"
        }
        
    # 3. Baggage and Amenities
    if any(word in text_lower for word in ["bag", "luggage", "carry on", "wi-fi", "lounge", "club"]):
        return {
            "intent": "Baggage_and_Amenities",
            "escalate_boolean": False,
            "escalation_reason": "None",
            "generated_reply": "For details on baggage sizes and lounge access, please review the policies on Delta.com.",
            "status": "success"
        }
        
    # 4. Flight Status and Upgrades
    if any(word in text_lower for word in ["upgrade", "seat", "first class", "boarding", "gate"]):
        return {
            "intent": "Flight_Status_and_Upgrades",
            "escalate_boolean": False,
            "escalation_reason": "None",
            "generated_reply": "Upgrades and seating are subject to availability. Check your Delta app for the latest status.",
            "status": "success"
        }
        
    # 5. Default Fallback
    return {
        "intent": "Praise_and_Feedback",
        "escalate_boolean": False,
        "escalation_reason": "None",
        "generated_reply": "Thank you for reaching out to Delta! We appreciate your feedback.",
        "status": "success"
    }

if __name__ == "__main__":
    test_message_1 = "My flight is delayed and I want a refund!"
    test_message_2 = "What is the weight limit for a carry on?"

    print("--- TRIVIAL BASELINE ---")
    print(json.dumps(TrivialBaseline(test_message_1), indent=2))
    
    print("\n--- SIMPLE BASELINE (Test 1) ---")
    print(json.dumps(SimpleBaseline(test_message_1), indent=2))
    
    print("\n--- SIMPLE BASELINE (Test 2) ---")
    print(json.dumps(SimpleBaseline(test_message_2), indent=2))