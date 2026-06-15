# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  Constants — sourced from dataset analysis
# ─────────────────────────────────────────

# ── Dataset Column Names (exact names from crm CSV) ──────────
COL_TICKET_ID    = "Ticket_ID"
COL_NAME         = "Customer_Name"
COL_EMAIL        = "Customer_Email"
COL_SUBJECT      = "Ticket_Subject"
COL_DESCRIPTION  = "Ticket_Description"
COL_CATEGORY     = "Issue_Category"
COL_PRIORITY     = "Priority_Level"
COL_CHANNEL      = "Ticket_Channel"
COL_DATE         = "Submission_Date"
COL_RT           = "Resolution_Time_Hours"
COL_AGENT        = "Assigned_Agent"
COL_SATISFACTION = "Satisfaction_Score"

# Columns not useful for modeling — dropped during preprocessing
COLS_TO_DROP = [COL_NAME, COL_AGENT, COL_DATE]

# Derived column names (added during preprocessing)
COL_COMBINED_TEXT  = "combined_text"
COL_PRIORITY_NUM   = "Priority_Numeric"
COL_CUSTOMER_TIER  = "Customer_Tier"
COL_SEM_SCORE      = "Severity_Semantic"
COL_RT_SCORE       = "Severity_RT"
COL_FUSED_SCORE    = "Fused_Score"
COL_INFERRED_SEV   = "Inferred_Severity"
COL_INFERRED_NUM   = "Inferred_Numeric"
COL_DELTA          = "Severity_Delta"
COL_DELTA_ABS      = "Severity_Delta_Abs"
COL_MISMATCH_LABEL = "Mismatch_Label"
COL_MISMATCH_TYPE  = "Mismatch_Type"
COL_MODEL_INPUT    = "Model_Input"
COL_PREDICTION     = "Prediction"
COL_CONFIDENCE     = "Confidence"

# ── Priority Mapping ─────────────────────────────────────────
PRIORITY_MAP    = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
PRIORITY_INV    = {v: k for k, v in PRIORITY_MAP.items()}
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]

# ── Label Indices ────────────────────────────────────────────
LABEL_CONSISTENT = 0
LABEL_MISMATCH   = 1
LABEL_NAMES      = {LABEL_CONSISTENT: "Consistent", LABEL_MISMATCH: "Mismatch"}

# ── Mismatch Types ───────────────────────────────────────────
MISMATCH_HIDDEN_CRISIS = "Hidden Crisis"   # Inferred > Assigned (under-triaged)
MISMATCH_FALSE_ALARM   = "False Alarm"     # Inferred < Assigned (over-triaged)
CONSISTENT             = "Consistent"

# ── Customer Tier Derivation (from email domain) ─────────────
# Discovered from dataset: 6 unique domains
TIER_MAP = {
    "enterprise.org": "enterprise",
    "company.com":    "business",
    "tech.io":        "tech",
    "example.com":    "standard",
    "example.net":    "standard",
    "example.org":    "standard",
}
TIER_DEFAULT = "standard"

# ── Issue Categories (from dataset) ──────────────────────────
# Key insight: Fraud → only Critical/High; Account/Billing → only Low/Medium
ISSUE_CATEGORIES = ["Technical", "Billing", "Account", "General Inquiry", "Fraud"]

# Category severity bias — used to weight Signal 1
# Based on cross-tab: Fraud=Critical/High only, Account/Billing=Low/Medium only
CATEGORY_SEVERITY_BIAS = {
    "Fraud":           4,
    "Technical":       3,
    "Billing":         2,
    "General Inquiry": 1,
    "Account":         1,
}

# ── Ticket Channels (from dataset) ───────────────────────────
TICKET_CHANNELS = ["Chat", "Email", "Web Form"]

# ── Signal Fusion Weights ────────────────────────────────────
WEIGHT_SEMANTIC    = 0.7
WEIGHT_RESOLUTION  = 0.3
MISMATCH_THRESHOLD = 1.5   # |inferred - assigned| >= this → Mismatch

# ── Severity Anchor Texts for Signal 1 ───────────────────────
# Rich multi-sentence anchors for stable cosine similarity scoring
SEVERITY_ANCHORS = {
    1: [
        "general information request about the product",
        "feature request for future consideration",
        "account settings question how to update profile",
        "documentation clarification about usage",
        "billing inquiry about invoice amount",
        "how to change email address or password",
        "cosmetic ui issue with font or color",
        "low urgency single user not blocking work",
        "everything is working fine minor question",
        "export data question no business impact",
    ],
    2: [
        "minor bug affecting one user with workaround",
        "slow page loading intermittent performance issue",
        "occasional error message that resolves itself",
        "feature not working as expected sometimes",
        "payment method update failing for one user",
        "data sync is delayed but not lost",
        "moderate impact some users affected",
        "workaround is available team can continue",
        "login slow but eventually succeeds",
        "report generation taking too long",
    ],
    3: [
        "production issue service degraded for all users",
        "important feature completely broken no workaround",
        "entire team is blocked cannot complete work",
        "multiple users affected by critical bug",
        "deadline at serious risk due to system issue",
        "application crashes frequently for many users",
        "login failing for entire organization",
        "we cannot process customer orders today",
        "significant business impact needs urgent attention",
        "escalating to management due to severity",
    ],
    4: [
        "complete system outage platform totally down",
        "security breach unauthorized access detected",
        "fraud detected financial transactions compromised",
        "data loss customer records corrupted or missing",
        "payment processing failure all transactions failing",
        "cannot access platform at all since morning",
        "all users affected business operations stopped",
        "SLA breach imminent emergency escalation required",
        "hacked stolen credentials identity theft",
        "critical infrastructure failure immediate action needed",
    ],
}

# ── Urgency Keywords for Dossier Evidence Extraction ─────────
URGENCY_KEYWORDS = {
    "critical": [
        "outage", "breach", "data loss", "system down", "cannot access",
        "complete failure", "corrupted", "security incident", "fraud",
        "unauthorized", "hacked", "stolen", "emergency",
    ],
    "high": [
        "urgent", "asap", "broken", "blocked", "not working",
        "failed", "error", "critical", "crashing", "crashes",
        "not loading", "spinning wheel", "cannot login", "login failed",
    ],
    "escalation": [
        "unacceptable", "immediately", "escalate", "manager",
        "legal action", "refund", "cancel subscription", "lawsuit",
        "compensation", "this is ridiculous",
    ],
    "negation": [
        "not working", "cannot", "won't work", "doesn't work",
        "unable to", "failed to", "never resolved", "still broken",
        "no response", "keeps failing",
    ],
}

# ── Expected Resolution Time Windows (hours) per Priority ────
# Derived from dataset analysis:
#   Critical → mean 12hrs  | High → mean 24hrs
#   Medium   → mean 44hrs  | Low  → mean 45hrs
EXPECTED_RT = {
    "Low":      (24, 72),
    "Medium":   (8,  24),
    "High":     (2,   8),
    "Critical": (0,   4),
}

# RT percentile thresholds (from dataset stats)
# 25th=11hrs, 50th=27hrs, 75th=58hrs, max=120hrs
RT_PERCENTILES = {
    "p25": 11,
    "p50": 27,
    "p75": 58,
    "max": 120,
}

# ── Verification Thresholds ───────────────────────────────────
THRESHOLD_ACCURACY         = 0.83
THRESHOLD_MACRO_F1         = 0.82
THRESHOLD_PER_CLASS_RECALL = 0.78
ADVERSARIAL_PASS_COUNT     = 7    # ≥7/10 → 10% score bonus

# ── Noise Pattern in Descriptions ────────────────────────────
# Dataset has filler sentences appended to descriptions
# e.g. "Lay soon message show know main." "Study talk teach."
# These are random word sequences — stripped during preprocessing
NOISE_PATTERN = r'\b([A-Z][a-z]+ ){2,5}[a-z]+\.\s*$'