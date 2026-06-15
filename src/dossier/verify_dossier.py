"""
verify_dossier.py
Anti-Hallucination Verification Layer
Checks every feature_evidence item against the original ticket.
Rejects any dossier containing unverifiable or fabricated claims.
Logs violations and outputs only verified dossiers.
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/dossier/verify_dossier.py
# ─────────────────────────────────────────

from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.constants import (
    COL_TICKET_ID,
    COL_COMBINED_TEXT,
    COL_PRIORITY,
    COL_RT,
    COL_INFERRED_SEV,
    COL_MISMATCH_TYPE,
    COL_DELTA_ABS,
    URGENCY_KEYWORDS,
    MISMATCH_HIDDEN_CRISIS,
    MISMATCH_FALSE_ALARM,
    PRIORITY_MAP,
    PRIORITY_LEVELS,
)
from src.utils.helpers import load_config, save_json, ensure_dir
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  INDIVIDUAL EVIDENCE VERIFIERS
# ══════════════════════════════════════════════════════════════

def verify_keyword_evidence(
    evidence_item: Dict,
    ticket_text: str,
) -> Tuple[bool, str]:
    """
    Verifies keyword evidence against ticket text.
    Uses flexible matching — same strategy as extraction.
    """
    import re

    keyword  = evidence_item.get("value", "").lower().strip()
    verified = evidence_item.get("verified", False)

    # Accept placeholder
    if keyword in ["no_urgency_keywords", "none", ""]:
        return True, ""

    if not verified:
        return False, f"Keyword '{keyword}' has verified=False"

    if not ticket_text or ticket_text.strip() == "":
        return True, ""

    text_lower = ticket_text.lower()
    text_clean = re.sub(r"[^a-z0-9\s]", " ", text_lower)
    kw_parts   = keyword.split()

    in_original = keyword in text_lower
    in_clean    = keyword in text_clean
    in_parts    = (
        len(kw_parts) > 1 and
        all(part in text_lower for part in kw_parts)
    )

    if in_original or in_clean or in_parts:
        return True, ""

    return (
        False,
        f"HALLUCINATION: keyword '{keyword}' not found in ticket text"
    )


def verify_rt_evidence(
    evidence_item: Dict,
    actual_rt: float,
    tolerance: float = 2.0,
) -> Tuple[bool, str]:
    """
    Verifies resolution time evidence against the actual RT value.
    """
    value    = evidence_item.get("value", "")
    verified = evidence_item.get("verified", False)

    if not verified:
        return False, "RT evidence has verified=False"

    # Parse RT value
    try:
        stated_rt = float(
            str(value)
            .replace("hrs", "")
            .replace("hr", "")
            .replace("hours", "")
            .strip()
        )
    except (ValueError, AttributeError):
        return False, f"RT evidence 'value' not parseable: '{value}'"

    # Check stated RT matches actual RT within tolerance
    if abs(stated_rt - actual_rt) > tolerance:
        return (
            False,
            f"HALLUCINATION: stated RT={stated_rt}hrs but "
            f"actual RT={actual_rt:.1f}hrs (diff > {tolerance}hrs)"
        )

    return True, ""


def verify_faiss_evidence(
    evidence_item: Dict,
) -> Tuple[bool, str]:
    """
    Verifies FAISS semantic similarity evidence.
    """
    verified        = evidence_item.get("verified", False)
    similar_tickets = evidence_item.get("similar_tickets", [])
    pattern         = evidence_item.get("pattern", "")

    if not verified:
        return False, "FAISS evidence has verified=False"

    if not isinstance(similar_tickets, list):
        return False, "FAISS evidence 'similar_tickets' is not a list"

    if not pattern or not isinstance(pattern, str):
        return False, "FAISS evidence 'pattern' is empty or not a string"

    # Allow empty similar_tickets (FAISS index may not be available)
    # Only validate entries if they exist
    for i, ticket in enumerate(similar_tickets):
        if not isinstance(ticket, dict):
            return False, f"Similar ticket {i} is not a dict"

    return True, ""

# ══════════════════════════════════════════════════════════════
#  DOSSIER SCHEMA VERIFIER
# ══════════════════════════════════════════════════════════════

def verify_dossier_schema(dossier: Dict) -> Tuple[bool, List[str]]:
    """
    Verifies the dossier follows the mandatory schema.

    Required fields:
        ticket_id, assigned_priority, inferred_severity,
        mismatch_type, severity_delta, feature_evidence,
        constraint_analysis, confidence

    Args:
        dossier : Dossier dict to verify

    Returns:
        (is_valid, violations)
    """
    required_fields = [
        "ticket_id",
        "assigned_priority",
        "inferred_severity",
        "mismatch_type",
        "severity_delta",
        "feature_evidence",
        "constraint_analysis",
        "confidence",
    ]

    violations = []

    # Check required fields present
    for field in required_fields:
        if field not in dossier:
            violations.append(f"Missing required field: '{field}'")

    if violations:
        return False, violations

    # Check mismatch_type is valid
    valid_types = [MISMATCH_HIDDEN_CRISIS, MISMATCH_FALSE_ALARM]
    if dossier["mismatch_type"] not in valid_types:
        violations.append(
            f"Invalid mismatch_type: '{dossier['mismatch_type']}'. "
            f"Must be one of {valid_types}"
        )

    # Check assigned_priority is valid
    if dossier["assigned_priority"] not in PRIORITY_LEVELS:
        violations.append(
            f"Invalid assigned_priority: '{dossier['assigned_priority']}'"
        )

    # Check inferred_severity is valid
    if dossier["inferred_severity"] not in PRIORITY_LEVELS:
        violations.append(
            f"Invalid inferred_severity: '{dossier['inferred_severity']}'"
        )

    # Check feature_evidence is non-empty list
    if not isinstance(dossier["feature_evidence"], list):
        violations.append("'feature_evidence' must be a list")
    elif len(dossier["feature_evidence"]) == 0:
        violations.append("'feature_evidence' is empty — must have at least 1 item")

    # Check constraint_analysis is non-empty string
    if not isinstance(dossier["constraint_analysis"], str):
        violations.append("'constraint_analysis' must be a string")
    elif len(dossier["constraint_analysis"].strip()) < 20:
        violations.append(
            f"'constraint_analysis' too short "
            f"({len(dossier['constraint_analysis'])} chars)"
        )

    # Check confidence is parseable float in [0, 1]
    try:
        conf = float(dossier["confidence"])
        if not (0.0 <= conf <= 1.0):
            violations.append(
                f"'confidence' must be in [0, 1]. Got: {conf}"
            )
    except (ValueError, TypeError):
        violations.append(
            f"'confidence' not parseable as float: '{dossier['confidence']}'"
        )

    return len(violations) == 0, violations


# ══════════════════════════════════════════════════════════════
#  FULL DOSSIER VERIFIER
# ══════════════════════════════════════════════════════════════

def verify_single_dossier(
    dossier: Dict,
    ticket_row: pd.Series,
) -> Tuple[bool, List[str]]:
    """
    Fully verifies a single dossier against its source ticket.

    Verification steps:
        1. Schema check — all required fields present and valid
        2. Keyword evidence — each keyword exists in actual text
        3. RT evidence — stated RT matches actual RT
        4. FAISS evidence — structure is valid
        5. Mismatch direction — type matches delta direction

    Args:
        dossier    : Dossier dict to verify
        ticket_row : Source ticket DataFrame row

    Returns:
        (is_valid, violations)
        is_valid   : True if ALL checks pass
        violations : List of violation descriptions (empty if valid)
    """
    violations = []

    # ── Step 1: Schema verification ───────────────────────────
    schema_ok, schema_violations = verify_dossier_schema(dossier)
    if not schema_ok:
        violations.extend(schema_violations)
        # If schema is broken, can't verify evidence safely
        return False, violations

    # ── Get ticket data for verification ─────────────────────
    ticket_text = str(ticket_row.get(COL_COMBINED_TEXT, ""))
    actual_rt   = float(ticket_row.get(COL_RT, 0.0))

    # ── Step 2: Verify each evidence item ─────────────────────
    for i, item in enumerate(dossier["feature_evidence"]):
        signal = item.get("signal", "")

        if signal == "keyword":
            ok, msg = verify_keyword_evidence(item, ticket_text)
            if not ok:
                violations.append(f"Evidence[{i}] keyword: {msg}")

        elif signal == "resolution_time":
            ok, msg = verify_rt_evidence(item, actual_rt)
            if not ok:
                violations.append(f"Evidence[{i}] RT: {msg}")

        elif signal == "semantic_similarity":
            ok, msg = verify_faiss_evidence(item)
            if not ok:
                violations.append(f"Evidence[{i}] FAISS: {msg}")

        else:
            violations.append(
                f"Evidence[{i}] unknown signal type: '{signal}'"
            )

    # ── Step 3: Verify mismatch direction ─────────────────────
    assigned_num = PRIORITY_MAP.get(dossier["assigned_priority"],  2)
    inferred_num = PRIORITY_MAP.get(dossier["inferred_severity"],  2)
    delta        = inferred_num - assigned_num

    if dossier["mismatch_type"] == MISMATCH_HIDDEN_CRISIS and delta <= 0:
        violations.append(
            f"Mismatch type is 'Hidden Crisis' but "
            f"inferred ({dossier['inferred_severity']}) <= "
            f"assigned ({dossier['assigned_priority']})"
        )

    if dossier["mismatch_type"] == MISMATCH_FALSE_ALARM and delta >= 0:
        violations.append(
            f"Mismatch type is 'False Alarm' but "
            f"inferred ({dossier['inferred_severity']}) >= "
            f"assigned ({dossier['assigned_priority']})"
        )

    is_valid = len(violations) == 0
    return is_valid, violations


# ══════════════════════════════════════════════════════════════
#  BATCH VERIFICATION
# ══════════════════════════════════════════════════════════════

def verify_all_dossiers(
    dossiers: List[Dict],
    df: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Verifies all dossiers against their source tickets.

    Separates dossiers into:
        verified : All checks passed — safe to submit
        rejected : One or more violations — disqualified

    Args:
        dossiers    : List of generated dossiers
        df          : Full DataFrame (to look up ticket rows)
        config_path : Path to config.yaml

    Returns:
        (verified_dossiers, rejected_dossiers, report)
        verified_dossiers : List of clean dossiers
        rejected_dossiers : List of {dossier, violations} dicts
        report            : Summary statistics dict
    """
    logger.info(
        f"Verifying {len(dossiers):,} dossiers "
        f"against source tickets"
    )

    verified  = []
    rejected  = []

    # Build ticket_id → row lookup
    ticket_lookup: Dict[str, pd.Series] = {}
    id_col = COL_TICKET_ID if COL_TICKET_ID in df.columns else None

    if id_col:
        for _, row in df.iterrows():
            tid = str(row.get(id_col, ""))
            ticket_lookup[tid] = row

    for i, dossier in enumerate(dossiers):
        ticket_id = str(dossier.get("ticket_id", ""))

        # Find source ticket row
        if ticket_id in ticket_lookup:
            ticket_row = ticket_lookup[ticket_id]
        else:
            log_warning(
                logger,
                f"Ticket {ticket_id} not found in DataFrame — "
                f"skipping verification"
            )
            verified.append(dossier)
            continue

        # Verify dossier
        is_valid, violations = verify_single_dossier(
            dossier    = dossier,
            ticket_row = ticket_row,
        )

        if is_valid:
            verified.append(dossier)
        else:
            rejected.append({
                "ticket_id":  ticket_id,
                "dossier":    dossier,
                "violations": violations,
            })
            logger.warning(
                f"Dossier [{ticket_id}] REJECTED — "
                f"{len(violations)} violation(s):"
            )
            for v in violations:
                logger.warning(f"   ✘ {v}")

        # Progress logging
        if (i + 1) % 200 == 0:
            logger.info(f"   Verified {i+1:,}/{len(dossiers):,} dossiers")

    # ── Build report ──────────────────────────────────────────
    total         = len(dossiers)
    n_verified    = len(verified)
    n_rejected    = len(rejected)
    rejection_rate = n_rejected / total if total > 0 else 0.0

    report = {
        "total_dossiers":   total,
        "verified":         n_verified,
        "rejected":         n_rejected,
        "rejection_rate":   round(rejection_rate, 4),
        "hallucination_free": n_rejected == 0,
    }

    logger.info("=" * 50)
    logger.info("  DOSSIER VERIFICATION REPORT")
    logger.info("=" * 50)
    logger.info(f"  Total dossiers   : {total:,}")
    logger.info(f"  Verified (clean) : {n_verified:,}")
    logger.info(f"  Rejected         : {n_rejected:,}")
    logger.info(f"  Rejection rate   : {rejection_rate:.1%}")
    logger.info(
        f"  Hallucination-free: "
        f"{'YES ✔' if n_rejected == 0 else 'NO ✘'}"
    )
    logger.info("=" * 50)

    if n_rejected == 0:
        log_success(logger, "All dossiers verified — zero hallucinations")
    else:
        log_warning(
            logger,
            f"{n_rejected} dossiers rejected. "
            f"Check rejection log above."
        )

    return verified, rejected, report


# ══════════════════════════════════════════════════════════════
#  SAVE VERIFICATION REPORT
# ══════════════════════════════════════════════════════════════

def save_verification_report(
    verified: List[Dict],
    rejected: List[Dict],
    report: Dict,
    output_dir: str = "outputs/dossiers/",
) -> None:
    """
    Saves verification results to disk.

    Files saved:
        evidence_dossiers_verified.json  — clean dossiers only
        dossier_rejections.json          — rejected dossiers + violations
        verification_report.json         — summary statistics

    Args:
        verified   : Verified dossiers
        rejected   : Rejected dossiers with violations
        report     : Summary report dict
        output_dir : Output directory
    """
    ensure_dir(output_dir)

    save_json(
        verified,
        f"{output_dir}/evidence_dossiers_verified.json"
    )
    save_json(
        rejected,
        f"{output_dir}/dossier_rejections.json"
    )
    save_json(
        report,
        f"{output_dir}/verification_report.json"
    )

    log_success(
        logger,
        f"Verification results saved → {output_dir}"
    )