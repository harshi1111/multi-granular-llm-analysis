# src/factual_checker.py
from typing import Dict, Any, List, Tuple
import re
import json
import os

class FactualChecker:
    """
    Very lightweight factual checker for demo purposes.
    Uses a small JSON knowledge base of facts and simple patterns
    like 'Australia is in Europe', 'president of X in YEAR is Y'.
    """

    def __init__(self, kb_path: str = "config/knowledge_base.json"):
        self.kb_path = kb_path
        self.kb = self._load_kb()

    def _load_kb(self) -> Dict[str, Any]:
        if os.path.exists(self.kb_path):
            with open(self.kb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # default small KB if file not present
        return {
            "us_president_2023": "Joe Biden",
            "continent_Australia": "Oceania",
            "continent_Europe": [
                "France", "Germany", "Italy", "Spain", "Portugal",
                "Poland", "Netherlands", "Belgium", "Greece"
            ],
        }

    def check(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Returns:
            {
              "factual_score": float in [0,1],
              "issues": [str, ...],
              "checked_claims": [str, ...]
            }
        """
        issues: List[str] = []
        checked: List[str] = []

        text = (prompt or "") + " " + (response or "")

        # 1) Check president of US in 2023 style claims
        pres_pattern = re.compile(
            r"(president of (the )?united states.*?(2023|in 2023).*?)(?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)*)",
            re.IGNORECASE,
        )
        for m in pres_pattern.finditer(text):
            name = m.group("name")
            checked.append(f"US president 2023 claimed: {name}")
            true_name = self.kb.get("us_president_2023")
            if true_name and name.lower() != true_name.lower():
                issues.append(
                    f"Claimed US president in 2023 is '{name}', "
                    f"but knowledge base says '{true_name}'."
                )

        # 2) Check continent of Australia
        aus_pattern = re.compile(
            r"Australia .*?in (?P<place>[A-Z][a-zA-Z]+)", re.IGNORECASE
        )
        for m in aus_pattern.finditer(text):
            place = m.group("place")
            checked.append(f"Australia location claimed: {place}")
            true_continent = self.kb.get("continent_Australia")
            if place.lower() != true_continent.lower():
                issues.append(
                    f"Claimed Australia is in '{place}', "
                    f"but knowledge base says it belongs to '{true_continent}'."
                )

        # Simple scoring: 1.0 if no issues, else penalty
        if not checked:
            factual_score = 1.0  # nothing to check
        else:
            factual_score = max(0.0, 1.0 - 0.4 * len(issues))

        return {
            "factual_score": factual_score,
            "issues": issues,
            "checked_claims": checked,
        }
