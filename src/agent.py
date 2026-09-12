import os
import json
import time
import random
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# ============================================================
# LOCAL MODEL INITIALIZATION (Qwen 2.5 0.5B)
# ============================================================
print("Loading local Qwen model into GPU memory...")
# Using 0.5B to ensure it fits comfortably in consumer VRAM
LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 

local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
local_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)

def parse_local_json(raw_text):
    """Helper to extract JSON from raw model output."""
    try:
        # Strip markdown blocks if the model wrapped the JSON
        match = re.search(r'\{.*\}', raw_text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception as e:
        print(f"Failed to parse JSON: {raw_text}")
        return None

# ============================================================
# 1. INITIALIZATION
# ============================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Please add it to your .env file.")

client = Groq(api_key=GROQ_API_KEY)
MODEL_ID = "openai/gpt-oss-20b"

# --- Load the RAG Components (Only runs once at startup) ---
print("Loading vectorizer and reference corpus...")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
nn_model = joblib.load("models/nn_model.pkl")
reference_corpus = pd.read_pickle("models/reference_corpus.pkl")

# ============================================================
# 2. SYSTEM PROMPTS
# ============================================================
ROUTER_PROMPT = """
You are an AI support routing agent for Delta Airlines.

Analyze the customer message and output a strict JSON object
with exactly these three keys:
- "intent": Must be exactly one of: [Flight_Status_and_Upgrades, Baggage_and_Amenities, Refunds_and_Complaints, Praise_and_Feedback, Needs_Context_or_DM]
- "escalate_boolean": true if human intervention, private data, PII, booking-specific information, or case-specific investigation is needed. Otherwise false.
- "escalation_reason": A short string explaining why it must be escalated, or "None".
"""

LOCAL_ROUTER_PROMPT = """You are a Delta Airlines support routing agent.
Classify intent and decide escalation.

Allowed intents:
[Flight_Status_and_Upgrades, Baggage_and_Amenities, Refunds_and_Complaints, Praise_and_Feedback, Needs_Context_or_DM]

Rules:
- Delays, missed flights, rebooking, anger, gate issues -> escalate_boolean: true
- Vague requests needing booking details -> escalate_boolean: true
- General policy questions (baggage dimensions, wifi fees) -> escalate_boolean: false
- Compliments and thanks -> escalate_boolean: false

Example 1:
Customer: "Flight 202 is delayed 3 hours, I am going to miss my connection!"
Output: {"intent": "Refunds_and_Complaints", "escalate_boolean": true, "escalation_reason": "Severe flight delay with connection risk"}

Example 2:
Customer: "What are the carry-on dimensions for domestic flights?"
Output: {"intent": "Baggage_and_Amenities", "escalate_boolean": false, "escalation_reason": "None"}

Example 3:
Customer: "Always pleasant to fly with Delta! Very friendly crew!"
Output: {"intent": "Praise_and_Feedback", "escalate_boolean": false, "escalation_reason": "None"}

Example 4:
Customer: "I need a little assistance please"
Output: {"intent": "Needs_Context_or_DM", "escalate_boolean": true, "escalation_reason": "Vague request requiring booking details"}
"""

DRAFTER_PROMPT = """
You are a customer support agent for Delta Airlines.
Write a short, polite, and helpful reply to the customer's message.
Use the provided Historical Context to match the brand's tone and standard policies.
Do NOT ask for PII (like booking reference or phone numbers) since this ticket was cleared for auto-handling.
Do not invent policies or information that is not supported by the Historical Context.
"""

# ============================================================
# 3. GROQ HELPER WITH RETRY
# ============================================================
def generate_with_retry(messages, response_format=None, max_retries=3):
    """Calls Groq with retry handling for temporary API failures."""
    transient_errors = ("429", "500", "502", "503", "504")

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": MODEL_ID,
                "messages": messages,
                "temperature": 0.1
            }
            if response_format:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            error_text = str(e)
            is_transient = any(error_code in error_text for error_code in transient_errors)

            if not is_transient or attempt == max_retries - 1:
                raise

            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Temporary Groq API error (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
            time.sleep(delay)

# ============================================================
# 4. RETRIEVAL FUNCTION
# ============================================================
def get_historical_context(customer_text):
    vec = vectorizer.transform([customer_text])
    distances, indices = nn_model.kneighbors(vec)
    context = ""
    for idx in indices[0]:
        row = reference_corpus.iloc[idx]
        context += f"PAST CUSTOMER: {row['text_customer']}\nPAST DELTA REPLY: {row['text_delta']}\n---\n"
    return context

# ============================================================
# 5. DEMO MODE
# ============================================================
def DemoMode(customer_text):
    # --------------------------------------------------------
    # STEP 1: ROUTING
    # --------------------------------------------------------
    try:
        router_messages = [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": f"Customer Message: {customer_text}"}
        ]
        raw_response = generate_with_retry(router_messages, response_format="json")
        decision = json.loads(raw_response)
    except Exception as e:
        return {"intent": None, "escalate_boolean": True, "escalation_reason": "Routing error.", "generated_reply": None, "status": "routing_error", "error": str(e)}

    valid_intents = {"Flight_Status_and_Upgrades", "Baggage_and_Amenities", "Refunds_and_Complaints", "Praise_and_Feedback", "Needs_Context_or_DM"}
    intent = decision.get("intent")
    if intent not in valid_intents:
        return {"intent": None, "escalate_boolean": True, "escalation_reason": "Invalid routing result.", "generated_reply": None, "status": "invalid_router_output"}

    escalate = decision.get("escalate_boolean")
    if not isinstance(escalate, bool):
        decision["escalate_boolean"] = True
        decision["escalation_reason"] = "Invalid escalation decision; routed to human for safety."

    # --------------------------------------------------------
    # STEP 2: GENERATION
    # --------------------------------------------------------
    reply = None
    if not decision["escalate_boolean"]:
        try:
            context = get_historical_context(customer_text)
            draft_messages = [
                {"role": "system", "content": DRAFTER_PROMPT},
                {"role": "user", "content": f"Historical Context:\n{context}\n\nCustomer Message: {customer_text}"}
            ]
            reply = generate_with_retry(draft_messages).strip()
        except Exception as e:
            return {"intent": intent, "escalate_boolean": False, "escalation_reason": "None", "generated_reply": None, "status": "generation_api_error", "error": str(e)}

    decision["generated_reply"] = reply
    decision["status"] = "success"
    return decision

def LocalMode(customer_text):
    """
    Privacy-first local execution using Qwen 2.5 0.5B via PyTorch
    with deterministic post-processing guardrails.
    """
    print(f"\n[LOCAL MODE] Processing: {customer_text}")

    # --------------------------------------------------------
    # STEP 1: ROUTING INFERENCE
    # --------------------------------------------------------
    print("   -> Routing agent thinking...")
    router_messages = [
        {"role": "system", "content": LOCAL_ROUTER_PROMPT},
        {"role": "user", "content": f"Customer Message: {customer_text}\nOutput strictly valid JSON:"}
    ]

    text_input = local_tokenizer.apply_chat_template(
        router_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = local_tokenizer([text_input], return_tensors="pt").to(local_model.device)

    generated_ids = local_model.generate(
        **model_inputs,
        max_new_tokens=150,
        temperature=0.1
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    router_response = local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    decision = parse_local_json(router_response)

    # Fallback if parsing fails completely
    if not decision:
        return {
            "intent": "Needs_Context_or_DM",
            "escalate_boolean": True,
            "escalation_reason": "Local model failed to output valid JSON.",
            "generated_reply": None,
            "status": "local_parsing_error"
        }

    # --------------------------------------------------------
    # STEP 1.5: DETERMINISTIC POST-PROCESSING GUARDRAILS
    # --------------------------------------------------------
    intent = decision.get("intent")
    esc_reason = str(decision.get("escalation_reason", "")).strip().lower()

    # Guardrail A: Praise & Feedback should never escalate
    if intent == "Praise_and_Feedback":
        decision["escalate_boolean"] = False
        decision["escalation_reason"] = "None"

    # Guardrail B: Smart Escalation Trigger
    # Catches cases where the SLM wrote an escalation-worthy reason but defaulted the boolean to False
    risk_keywords = [
        "delay", "cancel", "urgent", "rebook", "human",
        "frustrat", "miss", "lost", "assist", "stuck"
    ]
    if any(k in esc_reason for k in risk_keywords) and not decision.get("escalate_boolean"):
        decision["escalate_boolean"] = True

    # Guardrail C: Strict boolean type check
    if not isinstance(decision.get("escalate_boolean"), bool):
        decision["escalate_boolean"] = True
        decision["escalation_reason"] = "Invalid escalation decision type; routed to human for safety."

    # --------------------------------------------------------
    # STEP 2: RAG-GROUNDED DRAFTING (If Auto-Handled)
    # --------------------------------------------------------
    reply = None
    if not decision.get("escalate_boolean"):
        print("   -> Fetching historical context and drafting reply...")

        try:
            context = get_historical_context(customer_text)
        except Exception as e:
            decision["escalate_boolean"] = True
            decision["escalation_reason"] = "RAG context retrieval failed."
            decision["status"] = "retrieval_error"
            return decision

        draft_messages = [
            {"role": "system", "content": DRAFTER_PROMPT},
            {"role": "user", "content": f"Historical Context:\n{context}\n\nCustomer Message: {customer_text}"}
        ]

        draft_input = local_tokenizer.apply_chat_template(
            draft_messages, tokenize=False, add_generation_prompt=True
        )
        draft_tensors = local_tokenizer([draft_input], return_tensors="pt").to(local_model.device)

        draft_ids = local_model.generate(
            **draft_tensors,
            max_new_tokens=200,
            temperature=0.7
        )

        draft_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(draft_tensors.input_ids, draft_ids)
        ]

        reply = local_tokenizer.batch_decode(draft_ids, skip_special_tokens=True)[0].strip()

    decision["generated_reply"] = reply
    decision["status"] = "success"

    return decision