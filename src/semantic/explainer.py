# src/semantic/explainer.py
#
# V3 Semantic layer — rule-based anomaly explanation generator.
#
# Strategy: scan evidence_window fields using lightweight heuristics.
# No ML models required; always safe to import and call.

"""V3 Semantic layer — rule-based anomaly explanation generator."""
from __future__ import annotations

_KNOWN_ERROR_KEYWORDS: frozenset[str] = frozenset(
    {"error", "exception", "fail", "failed", "timeout", "refused", "critical", "fatal"}
)
_LOW_DIVERSITY_THRESHOLD: int = 3   # unique templates below this → flag low diversity
_HIGH_DENSITY_THRESHOLD: int = 40   # token_count above this → flag high density


class RuleBasedExplainer:
    """
    Generates lightweight text explanations for anomalous windows.

    Rules applied (in priority order)
    ----------------------------------
    1. Error-keyword scan — look for error/exception/fail keywords in templates_preview
    2. Template diversity — flag low variety of event types in the window
    3. Window density — flag unusually large (or empty) token counts

    Falls back to a generic sentence when no rule fires.

    This class has no internal state and is safe to instantiate once and reuse.
    """

    def explain(self, evidence_window: dict) -> dict:
        """
        Generate a rule-based explanation for an anomaly.

        Parameters
        ----------
        evidence_window : dict from Alert.evidence_window or RiskResult.evidence_window.
            Recognised keys (all optional):
                templates_preview : list[str]  — template text snippets
                token_count       : int         — number of tokens in the window

        Returns
        -------
        dict with keys:
            explanation    : str       — human-readable explanation sentence
            evidence_tokens : list[str] — template snippets that triggered rules (up to 5)
        """
        templates: list = evidence_window.get("templates_preview", [])
        token_count: int = evidence_window.get("token_count", 0)

        reasons: list[str] = []
        evidence_tokens: list[str] = []

        # Rule 1: error-keyword scan
        error_matches: list[str] = []
        for tmpl in templates:
            tmpl_lower = str(tmpl).lower()
            if any(kw in tmpl_lower for kw in _KNOWN_ERROR_KEYWORDS):
                error_matches.append(str(tmpl))
        if error_matches:
            reasons.append(
                f"Detected {len(error_matches)} error-indicative log template(s)"
            )
            evidence_tokens.extend(error_matches[:5])

        # Rule 2: template diversity
        unique_count = len({str(t) for t in templates})
        if templates and unique_count < _LOW_DIVERSITY_THRESHOLD:
            reasons.append(
                f"Low template diversity: {unique_count} distinct template(s) in window"
            )

        # Rule 3: window density
        if token_count > _HIGH_DENSITY_THRESHOLD:
            reasons.append(f"High event density: {token_count} tokens in window")
        elif token_count == 0:
            reasons.append("Empty event window")

        explanation = (
            "; ".join(reasons) + "."
            if reasons
            else "Anomaly score exceeded detection threshold; no specific rule triggered."
        )

        return {
            "explanation": explanation,
            "evidence_tokens": evidence_tokens,
        }
