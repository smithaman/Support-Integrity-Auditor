# Support-Integrity-Auditor
# 🛡️ Support Integrity Auditor (SIA)

An AI-powered system that audits CRM support tickets and detects priority mismatches between the ticket's actual severity and the priority assigned by support agents.

##  Problem Statement

In enterprise-scale CRM ecosystems, manual ticket triage is often affected by:

- Agent fatigue bias
- Customer favoritism
- Keyword anchoring
- Inconsistent severity assessment

This can lead to:

- Critical issues being marked as Low Priority
- Non-critical tickets being escalated unnecessarily
- SLA violations
- Increased customer dissatisfaction

Existing rule-based systems fail to understand the true context and severity of support requests.

##  Our Solution

Support Integrity Auditor automatically identifies priority mismatches by combining:

- Semantic severity understanding using NLP
- Historical resolution-time analysis
- Pseudo-label generation without manual annotations
- Transformer-based classification
- Evidence-backed verification through semantic retrieval

The system not only flags suspicious tickets but also provides supporting evidence to justify its decision.

##  Key Features

- Semantic Severity Scoring
- Resolution-Time Severity Analysis
- Automatic Pseudo-Label Generation
- DeBERTa-v3 Based Mismatch Detection
- FAISS-Powered Evidence Retrieval
- Explainable Audit Reports
- Interactive Streamlit Dashboard

## Architecture

Raw CRM Tickets
↓
Data Preprocessing
↓
Semantic Severity Signal
+
Resolution-Time Signal
↓
Weighted Signal Fusion
↓
Inferred True Severity
↓
Priority Mismatch Detection
↓
DeBERTa-v3 Classifier
↓
FAISS Evidence Retrieval
↓
Evidence Dossier Generation
↓
Streamlit Dashboard

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- PyTorch
- Hugging Face Transformers
- DeBERTa-v3-small
- BGE Embeddings
- FAISS
- Scikit-Learn
- Streamlit

##  Expected Impact

- Improve ticket prioritization accuracy
- Reduce SLA violations
- Minimize manual auditing efforts
- Increase fairness in support operations
- Enhance customer satisfaction

##  Team

Support Integrity Auditor

Built for intelligent and explainable CRM ticket auditing.

##  License

MIT License
