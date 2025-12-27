"""
Pattern matching for common inference forms and logical fallacies.
"""
from typing import List, Dict, Any, Optional, Tuple
import re
from enum import Enum

class InferencePattern(Enum):
    """Types of inference patterns."""
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    CONSTRUCTIVE_DILEMMA = "constructive_dilemma"
    CATEGORICAL_SYLLOGISM = "categorical_syllogism"
    
class FallacyPattern(Enum):
    """Types of logical fallacies."""
    AFFIRMING_CONSEQUENT = "affirming_consequent"
    DENYING_ANTECEDENT = "denying_antecedent"
    FALSE_DILEMMA = "false_dilemma"
    HASTY_GENERALIZATION = "hasty_generalization"
    SLIPPERY_SLOPE = "slippery_slope"
    CIRCULAR_REASONING = "circular_reasoning"
    STRAW_MAN = "straw_man"
    AD_HOMINEM = "ad_hominem"

class InferencePatternMatcher:
    """
    Matches common inference patterns and logical fallacies in reasoning.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Inference patterns
        self.inference_patterns = {
            InferencePattern.MODUS_PONENS: {
                "pattern": ["if P then Q", "P", "therefore Q"],
                "description": "Valid: If P implies Q, and P is true, then Q is true"
            },
            InferencePattern.MODUS_TOLLENS: {
                "pattern": ["if P then Q", "not Q", "therefore not P"],
                "description": "Valid: If P implies Q, and Q is false, then P is false"
            },
            InferencePattern.HYPOTHETICAL_SYLLOGISM: {
                "pattern": ["if P then Q", "if Q then R", "therefore if P then R"],
                "description": "Valid: Chain of implications"
            },
            InferencePattern.DISJUNCTIVE_SYLLOGISM: {
                "pattern": ["P or Q", "not P", "therefore Q"],
                "description": "Valid: If P or Q is true, and P is false, then Q is true"
            },
        }
        
        # Fallacy patterns
        self.fallacy_patterns = {
            FallacyPattern.AFFIRMING_CONSEQUENT: {
                "pattern": ["if P then Q", "Q", "therefore P"],
                "description": "Fallacy: Assuming the converse is true"
            },
            FallacyPattern.DENYING_ANTECEDENT: {
                "pattern": ["if P then Q", "not P", "therefore not Q"],
                "description": "Fallacy: Assuming the inverse is true"
            },
            FallacyPattern.FALSE_DILEMMA: {
                "indicators": ["either", "or", "only two options", "must choose"],
                "description": "Presenting only two options when more exist"
            },
            FallacyPattern.HASTY_GENERALIZATION: {
                "indicators": ["all", "every", "always", "never", "based on one example"],
                "description": "Making broad generalizations from limited evidence"
            },
            FallacyPattern.CIRCULAR_REASONING: {
                "indicators": ["because", "since", "as evidenced by", "proves that"],
                "description": "Using the conclusion as a premise"
            },
        }
        
        # Categorical syllogism patterns
        self.syllogism_patterns = self._initialize_syllogism_patterns()
    
    def _initialize_syllogism_patterns(self) -> Dict:
        """Initialize patterns for categorical syllogisms."""
        return {
            "AAA-1": ("All M are P", "All S are M", "Therefore all S are P"),
            "EAE-1": ("No M are P", "All S are M", "Therefore no S are P"),
            "AII-1": ("All M are P", "Some S are M", "Therefore some S are P"),
            "EIO-1": ("No M are P", "Some S are M", "Therefore some S are not P"),
        }
    
    def match_inference_patterns(self, reasoning_steps: List[Dict]) -> Dict:
        """
        Match inference patterns in reasoning steps.
        
        Args:
            reasoning_steps: List of reasoning steps
            
        Returns:
            Dictionary of matched patterns
        """
        results = {
            "valid_inferences": [],
            "fallacies": [],
            "syllogisms": [],
            "pattern_coverage": 0.0
        }
        
        if len(reasoning_steps) < 2:
            return results
        
        # Extract step texts
        step_texts = [step.get("text", "").lower() for step in reasoning_steps]
        
        # Check for valid inference patterns
        valid_matches = self._check_valid_patterns(step_texts, reasoning_steps)
        results["valid_inferences"] = valid_matches
        
        # Check for fallacies
        fallacy_matches = self._check_fallacies(step_texts, reasoning_steps)
        results["fallacies"] = fallacy_matches
        
        # Check for categorical syllogisms
        syllogism_matches = self._check_syllogisms(step_texts, reasoning_steps)
        results["syllogisms"] = syllogism_matches
        
        # Calculate pattern coverage
        total_checks = len(valid_matches) + len(fallacy_matches) + len(syllogism_matches)
        possible_patterns = len(self.inference_patterns) + len(self.fallacy_patterns)
        
        if possible_patterns > 0:
            results["pattern_coverage"] = total_checks / possible_patterns
        
        return results
    
    def _check_valid_patterns(self, step_texts: List[str], 
                            reasoning_steps: List[Dict]) -> List[Dict]:
        """Check for valid inference patterns."""
        matches = []
        
        # Need at least 3 steps for most patterns
        if len(step_texts) < 3:
            return matches
        
        # Check each pattern
        for pattern_type, pattern_info in self.inference_patterns.items():
            pattern = pattern_info["pattern"]
            
            # Try to match pattern against consecutive steps
            for i in range(len(step_texts) - len(pattern) + 1):
                window = step_texts[i:i + len(pattern)]
                
                if self._matches_pattern(window, pattern):
                    matches.append({
                        "type": "valid_inference",
                        "pattern": pattern_type.value,
                        "description": pattern_info["description"],
                        "steps": reasoning_steps[i:i + len(pattern)],
                        "start_index": i,
                        "confidence": 0.8
                    })
        
        return matches
    
    def _check_fallacies(self, step_texts: List[str], 
                        reasoning_steps: List[Dict]) -> List[Dict]:
        """Check for logical fallacies."""
        matches = []
        
        # Check each fallacy pattern
        for fallacy_type, fallacy_info in self.fallacy_patterns.items():
            
            if "pattern" in fallacy_info:
                # Pattern-based fallacies (like affirming consequent)
                pattern = fallacy_info["pattern"]
                
                for i in range(len(step_texts) - len(pattern) + 1):
                    window = step_texts[i:i + len(pattern)]
                    
                    if self._matches_pattern(window, pattern):
                        matches.append({
                            "type": "fallacy",
                            "fallacy": fallacy_type.value,
                            "description": fallacy_info["description"],
                            "steps": reasoning_steps[i:i + len(pattern)],
                            "start_index": i,
                            "confidence": 0.7
                        })
            
            elif "indicators" in fallacy_info:
                # Indicator-based fallacies
                indicators = fallacy_info["indicators"]
                
                for i, step in enumerate(reasoning_steps):
                    step_text = step.get("text", "").lower()
                    
                    # Check for multiple indicators in same step
                    found_indicators = []
                    for indicator in indicators:
                        if indicator in step_text:
                            found_indicators.append(indicator)
                    
                    if len(found_indicators) >= 2:  # Require at least 2 indicators
                        matches.append({
                            "type": "fallacy",
                            "fallacy": fallacy_type.value,
                            "description": fallacy_info["description"],
                            "step": step,
                            "step_index": i,
                            "indicators_found": found_indicators,
                            "confidence": 0.6
                        })
        
        return matches
    
    def _check_syllogisms(self, step_texts: List[str], 
                         reasoning_steps: List[Dict]) -> List[Dict]:
        """Check for categorical syllogisms."""
        matches = []
        
        if len(step_texts) < 3:
            return matches
        
        # Extract categorical statements
        categorical_steps = []
        for i, text in enumerate(step_texts):
            statement_type = self._classify_categorical_statement(text)
            if statement_type:
                categorical_steps.append({
                    "index": i,
                    "text": text,
                    "type": statement_type,
                    "step": reasoning_steps[i]
                })
        
        # Check for syllogism patterns
        if len(categorical_steps) >= 3:
            for i in range(len(categorical_steps) - 2):
                triple = categorical_steps[i:i + 3]
                
                # Check if this forms a syllogism
                syllogism_type = self._identify_syllogism_type(triple)
                if syllogism_type:
                    matches.append({
                        "type": "syllogism",
                        "syllogism_type": syllogism_type,
                        "steps": [s["step"] for s in triple],
                        "indices": [s["index"] for s in triple],
                        "confidence": 0.7
                    })
        
        return matches
    
    def _matches_pattern(self, window: List[str], pattern: List[str]) -> bool:
        """
        Check if window matches a pattern.
        Simplified matching for demonstration.
        """
        if len(window) != len(pattern):
            return False
        
        # Basic pattern matching
        for w, p in zip(window, pattern):
            # Check for pattern keywords
            if "if" in p and "then" in p:
                if "if" not in w or "then" not in w:
                    return False
            elif "or" in p:
                if "or" not in w:
                    return False
            elif "not" in p:
                if "not" not in w:
                    return False
        
        return True
    
    def _classify_categorical_statement(self, text: str) -> Optional[str]:
        """Classify categorical statement type (A, E, I, O)."""
        text_lower = text.lower()
        
        # Universal affirmative (All S are P)
        if text_lower.startswith("all ") or text_lower.startswith("every "):
            return "A"
        
        # Universal negative (No S are P)
        elif text_lower.startswith("no ") or text_lower.startswith("none "):
            return "E"
        
        # Particular affirmative (Some S are P)
        elif text_lower.startswith("some ") or "there exists" in text_lower:
            return "I"
        
        # Particular negative (Some S are not P)
        elif "are not" in text_lower or "is not" in text_lower:
            return "O"
        
        return None
    
    def _identify_syllogism_type(self, triple: List[Dict]) -> Optional[str]:
        """Identify if triple matches a known syllogism form."""
        # Get statement types
        types = [t["type"] for t in triple]
        type_str = ''.join(types)
        
        # Check against known syllogism forms
        for form, _ in self.syllogism_patterns.items():
            if form.startswith(type_str):
                return form
        
        return None
    
    def generate_pattern_report(self, pattern_results: Dict) -> str:
        """Generate human-readable pattern matching report."""
        report = []
        report.append("=" * 60)
        report.append("INFERENCE PATTERN ANALYSIS")
        report.append("=" * 60)
        
        # Valid inferences
        valid_inferences = pattern_results.get("valid_inferences", [])
        if valid_inferences:
            report.append(f"\nVALID INFERENCE PATTERNS FOUND ({len(valid_inferences)}):")
            for i, inf in enumerate(valid_inferences[:3], 1):
                report.append(f"  {i}. {inf.get('pattern', 'N/A').replace('_', ' ').title()}")
                report.append(f"     Description: {inf.get('description', '')}")
                report.append(f"     Steps: {inf.get('start_index', 0)+1}-{inf.get('start_index', 0)+len(inf.get('steps', []))}")
        
        # Fallacies
        fallacies = pattern_results.get("fallacies", [])
        if fallacies:
            report.append(f"\nLOGICAL FALLACIES DETECTED ({len(fallacies)}):")
            for i, fall in enumerate(fallacies[:3], 1):
                report.append(f"  {i}. {fall.get('fallacy', 'N/A').replace('_', ' ').title()}")
                report.append(f"     Description: {fall.get('description', '')}")
                if 'indicators_found' in fall:
                    report.append(f"     Indicators: {', '.join(fall['indicators_found'])}")
        
        # Syllogisms
        syllogisms = pattern_results.get("syllogisms", [])
        if syllogisms:
            report.append(f"\nCATEGORICAL SYLLOGISMS FOUND ({len(syllogisms)}):")
            for i, syll in enumerate(syllogisms[:2], 1):
                report.append(f"  {i}. Type: {syll.get('syllogism_type', 'N/A')}")
                report.append(f"     Valid: {'Yes' if syll.get('syllogism_type') in self.syllogism_patterns else 'No'}")
        
        # Pattern coverage
        coverage = pattern_results.get("pattern_coverage", 0)
        report.append(f"\nPATTERN COVERAGE: {coverage:.1%}")
        
        # Overall assessment
        if fallacies:
            assessment = "CONTAINS LOGICAL FALLACIES"
        elif valid_inferences:
            assessment = "CONTAINS VALID INFERENCE PATTERNS"
        else:
            assessment = "NO CLEAR INFERENCE PATTERNS DETECTED"
        
        report.append(f"\nOVERALL ASSESSMENT: {assessment}")
        
        return "\n".join(report)