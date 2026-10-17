"""Deterministic prompts for local LLM document extraction."""


def build_garage_quote_prompt(extraction) -> str:
    return f"""
Extract ALL explicitly stated information from this repair quote.

Return structured JSON matching the provided schema.

Rules:
- "Garage:" maps to garage.
- "Claim ID:" maps to claim_id.
- EVERY repair or replacement line before "Labor:" must be included in parts.
- For each part:
  - description = the text before ":"
  - price.amount = numeric amount only
  - price.currency = three-letter currency code
- "Labor:" maps ONLY to labor.
- "Total:" maps ONLY to total.
- Labor must NOT be included in parts.
- Total must NOT be included in parts.
- DO NOT omit repair lines.
- If the document contains:
    Front bumper: 780 EUR
    Left headlight: 620 EUR
    Hood: 840 EUR
  then parts MUST contain exactly 3 entries.
- Monetary amount must contain only the numeric value.
- Correct: {{"amount": 780, "currency": "EUR"}}
- Incorrect: {{"amount": "780 EUR", "currency": "EUR"}}
- Return null only when a scalar field is genuinely absent.
- Never invent missing information.

DOCUMENT:

{extraction.text}
""".strip()


def build_garage_quote_retry_prompt(extraction, missing_fields: list[str]) -> str:
    missing = ", ".join(missing_fields)
    return f"""
The previous extraction missed these fields:

{missing}

Re-read the repair quote carefully and extract the complete structured result.

Important:
- Look at EVERY line of the document.
- "Garage:" maps to garage.
- "Claim ID:" maps to claim_id.
- "Labor:" maps to labor.
- "Total:" maps to total.
- Repair/replacement lines map to parts.
- Labor must NOT be included in parts.
- Total must NOT be included in parts.
- Monetary amount must contain ONLY the numeric value.
- Example: {{"amount": 920, "currency": "EUR"}}
- Never include "EUR" inside amount.
- If a requested field is explicitly present, you MUST extract it.
- Return null ONLY if the information is genuinely absent.
- Do not calculate or invent missing values.
- Return valid JSON matching the provided schema.

DOCUMENT:

{extraction.text}
""".strip()


def build_claim_form_prompt(extraction) -> str:
    return f"""
Extract ALL explicitly stated information from this insurance claim form.

Rules:
- Use only information present in the document.
- Never invent missing information.
- Return null when a requested scalar field is absent.
- Use YYYY-MM-DD for incident_date.
- Return vehicle make and model separately.
- declared_damage must contain only damage types supported by the schema.
- injuries_declared must be true or false only when explicitly stated.
- Preserve the claim ID exactly.

DOCUMENT:

{extraction.text}
""".strip()


def build_accident_report_prompt(extraction) -> str:
    return f"""
Extract ALL explicitly stated information from this accident report.

Rules:
- Use only information present in the document.
- Never invent missing information.
- Return null when a requested scalar field is absent.
- Use YYYY-MM-DD for incident_date.
- Return vehicle make and model separately.
- reported_damage must contain only damage types supported by the schema.
- passengers must be the explicitly stated number of passengers.
- injuries_declared must be true or false only when explicitly stated.
- Preserve the claim ID exactly.

DOCUMENT:

{extraction.text}
""".strip()
