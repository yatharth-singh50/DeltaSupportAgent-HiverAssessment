import json
import sys
from agent import DemoMode, LocalMode

def interactive_session():
    print("=" * 60)
    print(" Delta Airlines AI Support Agent - Interactive Test Mode")
    print("=" * 60)
    
    # Engine Selection
    print("Select Inference Engine:")
    print(" [1] Fast Demo Mode (Groq API - openai/gpt-oss-20b)")
    print(" [2] Edge AI Mode (Local PyTorch - Qwen 2.5)")
    
    engine_choice = ""
    while engine_choice not in ["1", "2"]:
        engine_choice = input("Enter 1 or 2 > ").strip()
        
    engine_name = "Groq API" if engine_choice == "1" else "Local Edge Model"
    print(f"\n[SYSTEM] Initialized using {engine_name}.\n")
    
    print("Type any customer query/tweet below to see the agent's decision.")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            query = input("\nCustomer Tweet > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting interactive test mode. Goodbye!")
                break
                
            print(f"Processing via {engine_name}...")
            
            # Route to the selected engine
            if engine_choice == "1":
                decision = DemoMode(query)
            else:
                decision = LocalMode(query)
            
            print("\n--- AGENT RESPONSE ---")
            print(f"Intent Classified  : {decision.get('intent')}")
            print(f"Escalate to Human? : {'YES' if decision.get('escalate_boolean') else 'NO'}")
            print(f"Escalation Reason  : {decision.get('escalation_reason')}")
            
            if decision.get("escalate_boolean"):
                print("\nAction Taken       : [ESCALATED] Ticket flagged for human agent dispatch. Auto-reply suppressed for safety.")
            else:
                print(f"\nDrafted Reply      :\n\"{decision.get('generated_reply')}\"")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    interactive_session()