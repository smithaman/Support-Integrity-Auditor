"""
adversarial_tests.py
Adversarial Robustness Testing
Contains 10 hand-crafted adversarial tickets designed to fool
keyword-based systems (negated urgency, hidden critical cases,
false urgency language on trivial issues).
Evaluates model on these cases. Score ≥7/10 → 10% bonus.
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/adversarial/adversarial_tests.py
# ─────────────────────────────────────────

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config.constants import (
    COL_TICKET_ID,
    COL_SUBJECT,
    COL_DESCRIPTION,
    COL_PRIORITY,
    COL_CHANNEL,
    COL_CATEGORY,
    COL_RT,
    COL_CUSTOMER_TIER,
    COL_COMBINED_TEXT,
    COL_MODEL_INPUT,
    COL_PREDICTION,
    COL_CONFIDENCE,
    LABEL_MISMATCH,
    LABEL_CONSISTENT,
    LABEL_NAMES,
    ADVERSARIAL_PASS_COUNT,
)
from src.utils.helpers import load_config, save_json, ensure_dir
from src.utils.logger import (
    get_sia_logger,
    log_step,
    log_success,
    log_warning,
    log_metrics,
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  ADVERSARIAL TEST CASES
# ══════════════════════════════════════════════════════════════

# 10 hand-crafted adversarial tickets designed to fool
# keyword-based systems. Each has:
#   - A deceptive surface form (tricks keyword matching)
#   - A true label (what the correct prediction should be)
#   - A description of the adversarial strategy used

ADVERSARIAL_TICKETS = [
    {
        "ticket_id":          "ADV-001",
        "Ticket_Subject":     "Quick question about my account",
        "Ticket_Description": (
            "Hi support, I have a quick question. "
            "Everything seems completely fine on the surface but "
            "I noticed that three unauthorized transactions totaling "
            "$4,200 appeared on my account this morning. "
            "My login credentials may have been compromised. "
            "No urgency intended, just flagging this."
        ),
        "Priority_Level":     "Low",
        "Ticket_Channel":     "Email",
        "Issue_Category":     "Account",
        "Resolution_Time_Hours": 96.0,
        "Customer_Tier":      "standard",
        "true_label":         LABEL_MISMATCH,       # Actually Critical (fraud)
        "mismatch_type":      "Hidden Crisis",
        "adversarial_strategy": (
            "Polite framing + 'no urgency' disclaimer disguises "
            "a clear fraud/security incident. "
            "Keyword system misses due to softened language."
        ),
    },
    {
        "ticket_id":          "ADV-002",
        "Ticket_Subject":     "URGENT!!! ASAP!!! CRITICAL ISSUE!!!",
        "Ticket_Description": (
            "This is extremely urgent and needs immediate attention ASAP. "
            "Critical problem: I cannot remember which email address "
            "I used to sign up. Please help me find it. "
            "This is blocking everything for me right now."
        ),
        "Priority_Level":     "Critical",
        "Ticket_Channel":     "Chat",
        "Issue_Category":     "Account",
        "Resolution_Time_Hours": 1.5,
        "Customer_Tier":      "standard",
        "true_label":         LABEL_MISMATCH,       # Actually Low (trivial)
        "mismatch_type":      "False Alarm",
        "adversarial_strategy": (
            "Keyword injection — stuffs urgency words into subject/body "
            "for a completely trivial issue (forgot signup email). "
            "Keyword system would incorrectly validate Critical priority."
        ),
    },
    {
        "ticket_id":          "ADV-003",
        "Ticket_Subject":     "Minor performance observation",
        "Ticket_Description": (
            "I wanted to share a minor observation about system performance. "
            "Since yesterday our entire payment processing pipeline "
            "has been returning timeout errors for all transactions. "
            "We process approximately 50,000 transactions per hour. "
            "I understand you are busy so no rush."
        ),
        "Priority_Level":     "Low",
        "Ticket_Channel":     "Email",
        "Issue_Category":     "Technical",
        "Resolution_Time_Hours": 84.0,
        "Customer_Tier":      "enterprise",
        "true_label":         LABEL_MISMATCH,       # Actually Critical
        "mismatch_type":      "Hidden Crisis",
        "adversarial_strategy": (
            "Minimizing language ('minor observation', 'no rush') "
            "hides a complete payment system failure at enterprise scale. "
            "Semantic model should detect payment + timeout + enterprise context."
        ),
    },
    {
        "ticket_id":          "ADV-004",
        "Ticket_Subject":     "System completely down outage breach failure",
        "Ticket_Description": (
            "Just wanted to ask how I can update the font size "
            "in my profile settings page. "
            "I prefer a slightly larger text for readability. "
            "Thank you for your assistance."
        ),
        "Priority_Level":     "Critical",
        "Ticket_Channel":     "Web Form",
        "Issue_Category":     "Account",
        "Resolution_Time_Hours": 2.0,
        "Customer_Tier":      "standard",
        "true_label":         LABEL_MISMATCH,       # Actually Low
        "mismatch_type":      "False Alarm",
        "adversarial_strategy": (
            "Urgency keywords stuffed into subject only. "
            "Description reveals a completely trivial request. "
            "Model must weigh description content over subject keywords."
        ),
    },
    {
        "ticket_id":          "ADV-005",
        "Ticket_Subject":     "Everything is working perfectly fine",
        "Ticket_Description": (
            "Everything is working perfectly fine from what I can see. "
            "However I should mention that our database has not been "
            "accessible since the maintenance window 18 hours ago. "
            "All 200 of our engineers are unable to deploy any code. "
            "We have a product launch in 6 hours. "
            "But I am sure it will resolve itself."
        ),
        "Priority_Level":     "Medium",
        "Ticket_Channel":     "Email",
        "Issue_Category":     "Technical",
        "Resolution_Time_Hours": 72.0,
        "Customer_Tier":      "enterprise",
        "true_label":         LABEL_MISMATCH,       # Actually Critical
        "mismatch_type":      "Hidden Crisis",
        "adversarial_strategy": (
            "Contradictory framing — opens with 'everything is fine' "
            "but describes a 18-hour database outage blocking 200 engineers "
            "before a product launch. Passive dismissal at the end. "
            "Semantic model must understand the contradiction."
        ),
    },
    {
        "ticket_id":          "ADV-006",
        "Ticket_Subject":     "Feedback on recent experience",
        "Ticket_Description": (
            "I wanted to provide some feedback on my recent experience. "
            "While interacting with your platform I observed that "
            "a third party appears to have gained access to my account "
            "and has been exfiltrating customer data for the past 72 hours. "
            "I have attached logs. This may be worth looking into."
        ),
        "Priority_Level":     "Low",
        "Ticket_Channel":     "Email",
        "Issue_Category":     "General Inquiry",
        "Resolution_Time_Hours": 120.0,
        "Customer_Tier":      "business",
        "true_label":         LABEL_MISMATCH,       # Actually Critical
        "mismatch_type":      "Hidden Crisis",
        "adversarial_strategy": (
            "Formal feedback framing with extreme understatement "
            "('may be worth looking into') conceals active data exfiltration. "
            "Category is 'General Inquiry' — wrong categorization. "
            "Model must detect security breach from description semantics."
        ),
    },
    {
        "ticket_id":          "ADV-007",
        "Ticket_Subject":     "This is not urgent and not an emergency",
        "Ticket_Description": (
            "This is not urgent and I want to make clear this is not "
            "an emergency. I am not saying there is an outage. "
            "I am not reporting a breach. "
            "With that said, our entire API has been returning 500 errors "
            "for all clients since 3am and our SLA expires in 2 hours."
        ),
        "Priority_Level":     "Low",
        "Ticket_Channel":     "Chat",
        "Issue_Category":     "Technical",
        "Resolution_Time_Hours": 88.0,
        "Customer_Tier":      "enterprise",
        "true_label":         LABEL_MISMATCH,       # Actually Critical
        "mismatch_type":      "Hidden Crisis",
        "adversarial_strategy": (
            "Explicit negation of urgency keywords ('not urgent', "
            "'not an emergency', 'not an outage') while describing "
            "a genuine API outage with imminent SLA breach. "
            "Tests negation handling in semantic model."
        ),
    },
    {
        "ticket_id":          "ADV-008",
        "Ticket_Subject":     "Password reset help needed",
        "Ticket_Description": (
            "I need help resetting my password. "
            "I tried the reset link but it did not arrive in my inbox. "
            "I have checked spam. I have waited 30 minutes. "
            "I need to access the platform to join a call."
        ),
        "Priority_Level":     "High",
        "Ticket_Channel":     "Chat",
        "Issue_Category":     "Account",
        "Resolution_Time_Hours": 3.0,
        "Customer_Tier":      "standard",
        "true_label":         LABEL_MISMATCH,       # Actually Low/Medium
        "mismatch_type":      "False Alarm",
        "adversarial_strategy": (
            "Standard password reset framed as High priority. "
            "No business-critical impact. Fast resolution time. "
            "Model should detect mismatch — routine account issue "
            "cannot justify High priority."
        ),
    },
    {
        "ticket_id":          "ADV-009",
        "Ticket_Subject":     "Routine check in",
        "Ticket_Description": (
            "Just doing a routine check in to see if you received "
            "my previous message from last week. "
            "As mentioned, our production database has been corrupted "
            "and we have lost approximately 3 weeks of customer records. "
            "We serve 80,000 active users. "
            "Please respond when you get a chance."
        ),
        "Priority_Level":     "Low",
        "Ticket_Channel":     "Email",
        "Issue_Category":     "Technical",
        "Resolution_Time_Hours": 110.0,
        "Customer_Tier":      "enterprise",
        "true_label":         LABEL_MISMATCH,       # Actually Critical
        "mismatch_type":      "Hidden Crisis",
        "adversarial_strategy": (
            "Casual 'routine check-in' framing hides catastrophic "
            "data loss (3 weeks of records, 80K users). "
            "'When you get a chance' minimizes urgency. "
            "Model must identify data corruption severity from context."
        ),
    },
    {
        "ticket_id":          "ADV-010",
        "Ticket_Subject":     "Critical urgent emergency outage help needed now",
        "Ticket_Description": (
            "Critical emergency! Urgent outage! System down! "
            "I urgently need help with something extremely critical. "
            "My profile picture is not displaying correctly. "
            "It shows a blurry version instead of the clear one I uploaded. "
            "Please fix this critical emergency immediately."
        ),
        "Priority_Level":     "Critical",
        "Ticket_Channel":     "Chat",
        "Issue_Category":     "Account",
        "Resolution_Time_Hours": 1.0,
        "Customer_Tier":      "standard",
        "true_label":         LABEL_MISMATCH,       # Actually Low
        "mismatch_type":      "False Alarm",
        "adversarial_strategy": (
            "Maximum keyword stuffing (critical, urgent, emergency, "
            "outage) throughout subject and description for a completely "
            "cosmetic issue (blurry profile picture). "
            "Strongest False Alarm test case."
        ),
    },
]


# ══════════════════════════════════════════════════════════════
#  DATAFRAME BUILDER
# ══════════════════════════════════════════════════════════════

def build_adversarial_df() -> pd.DataFrame:
    """
    Converts ADVERSARIAL_TICKETS list into a DataFrame
    ready for the full SIA pipeline.
    """
    records = []
    for ticket in ADVERSARIAL_TICKETS:
        record = {
            COL_TICKET_ID:        ticket["ticket_id"],
            "Ticket_Subject":     ticket["Ticket_Subject"],
            "Ticket_Description": ticket["Ticket_Description"],
            COL_PRIORITY:         ticket["Priority_Level"],
            COL_CHANNEL:          ticket["Ticket_Channel"],
            COL_CATEGORY:         ticket["Issue_Category"],
            COL_RT:               ticket["Resolution_Time_Hours"],
            COL_CUSTOMER_TIER:    ticket["Customer_Tier"],
            # Add Customer_Email so encode_metadata doesn't crash
            "Customer_Email":     "user@example.com",
            "true_label":         ticket["true_label"],
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Build combined text
    df[COL_COMBINED_TEXT] = (
        df["Ticket_Subject"] + " [SEP] " + df["Ticket_Description"]
    )

    return df


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_adversarial(
    predictions: np.ndarray,
    confidences: np.ndarray,
    df: pd.DataFrame,
) -> Tuple[int, Dict]:
    """
    Evaluates model predictions on adversarial test cases.

    Scoring:
        Correct prediction (matches true_label) = 1 point
        Score >= 7/10 → 10% bonus

    Args:
        predictions : np.ndarray (10,) predicted labels
        confidences : np.ndarray (10,) mismatch probabilities
        df          : Adversarial DataFrame with true_label column

    Returns:
        (score, detailed_results)
        score            : int 0–10 correct predictions
        detailed_results : Per-ticket result dict
    """
    true_labels = df["true_label"].values
    results     = []
    score       = 0

    for i, (pred, conf, true) in enumerate(
        zip(predictions, confidences, true_labels)
    ):
        ticket    = ADVERSARIAL_TICKETS[i]
        correct   = int(pred) == int(true)

        if correct:
            score += 1

        result = {
            "ticket_id":           ticket["ticket_id"],
            "subject":             ticket["Ticket_Subject"],
            "assigned_priority":   ticket["Priority_Level"],
            "expected_label":      LABEL_NAMES[int(true)],
            "predicted_label":     LABEL_NAMES[int(pred)],
            "confidence":          round(float(conf), 4),
            "correct":             correct,
            "adversarial_strategy": ticket["adversarial_strategy"],
            "mismatch_type":       ticket["mismatch_type"],
        }
        results.append(result)

        status = "✔ CORRECT" if correct else "✘ WRONG"
        logger.info(
            f"  ADV-{i+1:03d} | "
            f"Expected={LABEL_NAMES[int(true)]:<12} | "
            f"Predicted={LABEL_NAMES[int(pred)]:<12} | "
            f"Conf={conf:.3f} | {status}"
        )

    return score, results


# ══════════════════════════════════════════════════════════════
#  MAIN ADVERSARIAL TEST PIPELINE
# ══════════════════════════════════════════════════════════════

def run_adversarial_tests(
    model,
    tokenizer,
    config_path: str    = "config/config.yaml",
    device: torch.device = None,
) -> Tuple[int, bool, Dict]:
    """
    Full adversarial robustness testing pipeline.

    Steps:
        1. Build adversarial DataFrame
        2. Run preprocessing + feature engineering
        3. Build model inputs
        4. Run classifier inference
        5. Evaluate predictions vs true labels
        6. Check bonus threshold (≥7/10)
        7. Save results

    Args:
        model       : Trained DeBERTa model
        tokenizer   : Loaded tokenizer
        config_path : Path to config.yaml
        device      : Target device

    Returns:
        (score, bonus_earned, detailed_results)
        score          : int 0–10 correct predictions
        bonus_earned   : True if score >= 7
        detailed_results: Dict with per-ticket results
    """
    from src.preprocessing.preprocess import (
        clean_texts, merge_text, encode_metadata
    )
    from src.preprocessing.feature_engineering import build_all_model_inputs
    from src.classifier.predict import predict_batch
    from src.classifier.dataset import build_dataloader

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    logger.info("=" * 55)
    logger.info("  ADVERSARIAL ROBUSTNESS TEST")
    logger.info(f"  {len(ADVERSARIAL_TICKETS)} test cases")
    logger.info("=" * 55)

    cfg = load_config(config_path)

    # ── Step 1: Build adversarial DataFrame ───────────────────
    log_step(logger, "Building adversarial test DataFrame")
    adv_df = build_adversarial_df()

    # ── Step 2: Preprocess ────────────────────────────────────
    log_step(logger, "Preprocessing adversarial tickets")
    adv_df = clean_texts(adv_df)
    adv_df = encode_metadata(adv_df)

    # ── Step 3: Build model inputs ────────────────────────────
    log_step(logger, "Building model inputs")
    adv_df = build_all_model_inputs(adv_df)

    # ── Step 4: Run inference ─────────────────────────────────
    log_step(logger, "Running inference on adversarial tickets")

    loader = build_dataloader(
        df         = adv_df,
        tokenizer  = tokenizer,
        batch_size = 10,
        max_length = cfg["classifier"]["max_length"],
        shuffle    = False,
        is_test    = True,
    )

    predictions, confidences = predict_batch(model, loader, device)

    # ── Step 5: Evaluate ──────────────────────────────────────
    log_step(logger, "Evaluating adversarial predictions")
    score, detailed_results = evaluate_adversarial(
        predictions = predictions,
        confidences = confidences,
        df          = adv_df,
    )

    # ── Step 6: Check bonus threshold ─────────────────────────
    bonus_earned = score >= ADVERSARIAL_PASS_COUNT

    logger.info("=" * 55)
    logger.info(f"  ADVERSARIAL SCORE: {score}/{len(ADVERSARIAL_TICKETS)}")
    logger.info(
        f"  BONUS (≥{ADVERSARIAL_PASS_COUNT}/10): "
        f"{'EARNED (+10%) ✔' if bonus_earned else 'NOT EARNED ✘'}"
    )
    logger.info("=" * 55)

    if bonus_earned:
        log_success(logger, f"Adversarial robustness bonus EARNED! Score: {score}/10")
    else:
        log_warning(
            logger,
            f"Adversarial score {score}/10 below threshold "
            f"{ADVERSARIAL_PASS_COUNT}/10. No bonus."
        )

    # ── Step 7: Save results ──────────────────────────────────
    report = {
        "score":              score,
        "total":              len(ADVERSARIAL_TICKETS),
        "pass_threshold":     ADVERSARIAL_PASS_COUNT,
        "bonus_earned":       bonus_earned,
        "per_ticket_results": detailed_results,
        "summary": {
            "correct":   score,
            "incorrect": len(ADVERSARIAL_TICKETS) - score,
            "accuracy":  round(score / len(ADVERSARIAL_TICKETS), 4),
        }
    }

    ensure_dir("outputs/metrics")
    save_json(report, "outputs/metrics/adversarial_results.json")
    log_success(logger, "Adversarial results saved → outputs/metrics/adversarial_results.json")

    return score, bonus_earned, report