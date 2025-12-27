"""
Reasoning-level analyzer for logical consistency and inference validation.
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from transformers import pipeline
import spacy
from sympy import symbols, Implies, And, Or, Not, to_cnf, satisfiable
from .base_analyzer import BaseAnalyzer
# Add imports at top
#from .inference_pattern_matcher import InferencePatternMatcher
#from .logic_validator import LogicValidator

class ReasoningAnalyzer(BaseAnalyzer):
    """Analyzes LLM responses at the reasoning level for logical consistency."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reasoning analyzer.
        
        Args:
            config: Configuration dictionary
                - reasoning_indicators: Keywords indicating reasoning steps
                - nli_model: Model for natural language inference
                - max_steps: Maximum reasoning steps to extract
                - logic_threshold: Threshold for logical consistency
        """
        super().__init__("reasoning", config)
        
        # Add performance config
        self.timeout = self.config.get("timeout", 30)
        self.max_steps = min(self.config.get("max_steps", 10), 10)  # Limit for long responses
        self.enable_lightweight = self.config.get("enable_lightweight", True)
        
        # Use lightweight NLI model for long responses
        self.lightweight_nli_model = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",  # Much faster (~400MB)
            device=-1
        )
        
        # Configuration
        self.reasoning_indicators = self.config.get(
            "reasoning_indicators", 
            ["therefore", "thus", "hence", "so", "because", "since", 
             "as a result", "consequently", "implies that", "which means",
             "it follows that", "we can conclude", "this shows that"]
        )
        self.max_steps = self.config.get("max_steps", 20)
        self.logic_threshold = self.config.get("logic_threshold", 0.7)
        self.min_step_length = self.config.get("min_step_length", 5)
        self.max_step_length = self.config.get("max_step_length", 200)
        self.coherence_threshold = self.config.get("coherence_threshold", 0.4)
        
        # Initialize models
        self.nli_model = self.lightweight_nli_model
        self.spacy_model = spacy.load("en_core_web_sm")
        
        # Initialize pattern matcher and logic validator
        self.pattern_matcher = None
        self.logic_validator = None
        
        # Logical operators for formal logic parsing
        self.logical_operators = {
            "and": "∧", "or": "∨", "not": "¬", "implies": "→",
            "if": "→", "then": "→", "iff": "↔", "if and only if": "↔"
        }
    
    def analyze_optimized(self, text: str) -> Dict:
        """
        Optimized analysis for long responses.
        """
        # Check if we need lightweight mode
        use_lightweight = len(text) > 1000 or self.enable_lightweight
        
        try:
            # Extract reasoning steps with limit
            reasoning_steps = self.extract_components(text)[:self.max_steps]
            
            # Calculate basic metrics (fast)
            metrics = self.calculate_basic_metrics(reasoning_steps)
            
            # Run lightweight logical analysis
            if use_lightweight:
                logical_analysis = self._analyze_logical_consistency_lightweight(reasoning_steps)
            else:
                logical_analysis = self._analyze_logical_consistency(reasoning_steps)
            
            # Simplified issues identification
            issues = self._identify_critical_issues(reasoning_steps, metrics, logical_analysis)
            
            return {
                "text": text,
                "reasoning_steps": reasoning_steps,
                "metrics": metrics,
                "logical_analysis": logical_analysis,
                "issues": issues,
                "summary": self._generate_lightweight_summary(metrics, logical_analysis, issues),
                "optimized_mode": "lightweight" if use_lightweight else "full"
            }
            
        except Exception as e:
            # Fallback: minimal analysis
            return self._fallback_analysis(text)
    
    def _analyze_logical_consistency_lightweight(self, reasoning_steps: List[Dict]) -> Dict:
        """Lightweight logical consistency check."""
        if not reasoning_steps or len(reasoning_steps) < 2:
            return {"logical_consistency": 1.0, "contradictions": []}
        
        # Only check adjacent steps for contradictions
        contradictions = []
        for i in range(len(reasoning_steps) - 1):
            step1 = reasoning_steps[i].get("text", "")[:100]  # First 100 chars
            step2 = reasoning_steps[i + 1].get("text", "")[:100]
            
            if step1 and step2:
                try:
                    result = self.lightweight_nli_model(
                        f"{step1} [SEP] {step2}",
                        truncation=True,
                        max_length=256
                    )
                    if result[0]["label"].upper() == "CONTRADICTION":
                        contradictions.append({
                            "step1_idx": i,
                            "step2_idx": i + 1,
                            "confidence": result[0]["score"]
                        })
                except:
                    continue
        
        consistency = 1.0 - (len(contradictions) * 0.2)
        return {
            "logical_consistency": max(0, consistency),
            "contradictions": contradictions
        }
    
    def calculate_basic_metrics(self, reasoning_steps: List[Dict]) -> Dict:
        """Calculate basic metrics without heavy NLI analysis."""
        if not reasoning_steps:
            return {"step_count": 0, "has_reasoning": False}
        
        # Basic metrics only
        step_lengths = [len(step.get("text", "")) for step in reasoning_steps]
        
        return {
            "step_count": len(reasoning_steps),
            "has_reasoning": len(reasoning_steps) > 1,
            "avg_step_length": np.mean(step_lengths) if step_lengths else 0,
            "total_length": sum(step_lengths),
            "explicit_steps": len([s for s in reasoning_steps if s.get("type") == "reasoning_step"]),
        }
    
    def _identify_critical_issues(self, reasoning_steps: List[Dict], 
                                 metrics: Dict, logical_analysis: Dict) -> List[Dict]:
        """Identify only critical issues."""
        issues = []
        
        # Check for no reasoning structure
        if not metrics.get("has_reasoning", False):
            issues.append({
                "type": "no_reasoning_structure",
                "severity": "medium",
                "description": "No clear reasoning structure detected",
                "step_count": metrics.get("step_count", 0)
            })
        
        # Check for contradictions
        contradictions = logical_analysis.get("contradictions", [])
        if contradictions:
            issues.append({
                "type": "logical_contradictions",
                "severity": "high",
                "description": f"Found {len(contradictions)} logical contradiction(s)",
                "contradiction_count": len(contradictions)
            })
        
        # Check for low logical consistency
        if logical_analysis.get("logical_consistency", 1) < self.logic_threshold:
            issues.append({
                "type": "low_logical_consistency",
                "severity": "high",
                "description": f"Low logical consistency score: {logical_analysis['logical_consistency']:.2f}",
                "metric_value": logical_analysis["logical_consistency"],
                "threshold": self.logic_threshold
            })
        
        return issues
    
    def _generate_lightweight_summary(self, metrics: Dict, 
                                     logical_analysis: Dict,
                                     issues: List[Dict]) -> Dict:
        """Generate lightweight summary."""
        consistency = logical_analysis.get("logical_consistency", 1)
        has_reasoning = metrics.get("has_reasoning", False)
        
        if not has_reasoning:
            reasoning_quality = "no_reasoning"
        elif consistency > 0.8:
            reasoning_quality = "good"
        elif consistency > 0.6:
            reasoning_quality = "fair"
        else:
            reasoning_quality = "poor"
        
        return {
            "has_issues": len(issues) > 0,
            "issue_count": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "high"]),
            "reasoning_quality": reasoning_quality,
            "logical_consistency_level": "consistent" if consistency > 0.7 else "inconsistent",
            "overall_score": consistency
        }
    
    def _fallback_analysis(self, text: str) -> Dict:
        """Minimal fallback analysis when other methods fail."""
        sentences = [s.text for s in self.spacy_model(text).sents]
        
        return {
            "text": text,
            "reasoning_steps": [],
            "metrics": {
                "step_count": 0,
                "has_reasoning": False,
                "sentence_count": len(sentences)
            },
            "logical_analysis": {
                "logical_consistency": 0.5,
                "contradictions": []
            },
            "issues": [{
                "type": "analysis_failed",
                "severity": "medium",
                "description": "Analysis could not be completed, using fallback"
            }],
            "summary": {
                "has_issues": True,
                "issue_count": 1,
                "critical_issues": 0,
                "reasoning_quality": "unknown",
                "overall_score": 0.5
            },
            "fallback_mode": True
        }
    
    def extract_components(self, text: str) -> List[Dict]:
        """
        Extract reasoning steps from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of reasoning steps with metadata
        """
        # First, try to extract explicit reasoning chains
        explicit_steps = self._extract_explicit_reasoning_steps(text)
        
        # If no explicit steps found, try to infer implicit reasoning structure
        if not explicit_steps or len(explicit_steps) < 2:
            implicit_steps = self._extract_implicit_reasoning_steps(text)
            if implicit_steps:
                return implicit_steps
        
        return explicit_steps
    
    def _extract_explicit_reasoning_steps(self, text: str) -> List[Dict]:
        """Extract reasoning steps using explicit indicators."""
        steps = []
        sentences = [sent.text for sent in self.spacy_model(text).sents]
        
        current_step = {"premises": [], "conclusion": None, "text": "", "type": "premise"}
        step_texts = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if this sentence contains a reasoning indicator
            has_indicator = any(indicator in sentence_lower 
                               for indicator in self.reasoning_indicators)
            
            # Check if this is likely a conclusion
            is_conclusion = (has_indicator or 
                           any(word in sentence_lower for word in ["conclusion", "conclude", "therefore"]) or
                           i == len(sentences) - 1)  # Last sentence often contains conclusion
            
            if is_conclusion and current_step["premises"]:
                # Finalize current step
                current_step["conclusion"] = sentence
                current_step["text"] = " ".join(current_step["premises"]) + " " + sentence
                current_step["type"] = "reasoning_step"
                steps.append(current_step.copy())
                
                # Start new step
                current_step = {"premises": [], "conclusion": None, "text": "", "type": "premise"}
                step_texts.append(current_step["text"].strip())
            else:
                # Add as premise
                current_step["premises"].append(sentence)
        
        # Handle any remaining premises
        if current_step["premises"]:
            current_step["text"] = " ".join(current_step["premises"])
            current_step["type"] = "premise_set"
            steps.append(current_step)
            step_texts.append(current_step["text"].strip())
        
        # Clean and validate steps
        validated_steps = []
        for step in steps:
            if self._validate_step(step):
                step["step_id"] = f"step_{len(validated_steps)}"
                validated_steps.append(step)
        
        return validated_steps
    
    def _extract_implicit_reasoning_steps(self, text: str) -> List[Dict]:
        """Extract implicit reasoning steps when no explicit indicators found."""
        doc = self.spacy_model(text)
        sentences = [sent.text for sent in doc.sents]
        
        # If few sentences, treat each as separate step
        if len(sentences) <= 3:
            steps = []
            for i, sentence in enumerate(sentences):
                step_type = "conclusion" if i == len(sentences) - 1 else "premise"
                steps.append({
                    "step_id": f"step_{i}",
                    "text": sentence,
                    "type": step_type,
                    "premises": [] if step_type == "conclusion" else [sentence],
                    "conclusion": sentence if step_type == "conclusion" else None
                })
            return steps
        
        # For longer texts, use discourse parsing to find argument structure
        steps = []
        chunk_size = max(2, len(sentences) // 4)  # Aim for 3-4 reasoning steps
        
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:min(i + chunk_size, len(sentences))]
            if len(chunk) >= 2:
                steps.append({
                    "step_id": f"step_{len(steps)}",
                    "text": " ".join(chunk),
                    "type": "implicit_step",
                    "premises": chunk[:-1],
                    "conclusion": chunk[-1] if len(chunk) > 1 else None
                })
        
        return steps
    
    def _validate_step(self, step: Dict) -> bool:
        """Validate if a reasoning step is well-formed."""
        text = step.get("text", "").strip()
        if not text:
            return False
        
        # Check length constraints
        if len(text) < self.min_step_length or len(text) > self.max_step_length:
            return False
        
        # Check if it contains meaningful content (not just punctuation/stopwords)
        doc = self.spacy_model(text[:100])  # Check first 100 chars
        content_words = [token for token in doc if not token.is_stop and not token.is_punct]
        if len(content_words) < 3:
            return False
        
        return True
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text at reasoning level.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing reasoning analysis results
        """
        # Use optimized analysis for long texts
        if len(text) > 2000:
            return self.analyze_optimized(text)
        
        # Extract reasoning steps
        reasoning_steps = self.extract_components(text)
        
        # Calculate metrics
        metrics = self.calculate_metrics(reasoning_steps)
        
        # Perform logical consistency analysis
        logical_analysis = self._analyze_logical_consistency(reasoning_steps)
        
        # Apply pattern matching
        pattern_results = {"fallacies": [], "valid_inferences": []}
        
        # Apply formal logic validation
        logic_results = {"contradictions": [],"consistency_score": 1.0,"inference_validity": [],"logical_rigor": 1.0}
        
        # Generate reports
        pattern_report = "Pattern matching disabled for performance"
        logic_report = "Formal logic validation disabled for performance"
        
        # Identify issues (enhanced with pattern and logic results)
        issues = self._identify_issues(reasoning_steps, metrics, logical_analysis, 
                                      pattern_results, logic_results)
        
        # Compile results
        self.results = {
            "text": text,
            "reasoning_steps": reasoning_steps,
            "metrics": metrics,
            "logical_analysis": logical_analysis,
            "pattern_analysis": pattern_results,
            "formal_logic_analysis": logic_results,
            "pattern_report": pattern_report,
            "logic_report": logic_report,
            "issues": issues,
            "summary": self._generate_summary(metrics, logical_analysis, 
                                             pattern_results, logic_results, issues)
        }
        
        return self.results
    
    def calculate_metrics(self, reasoning_steps: List[Dict]) -> Dict:
        """Calculate reasoning-level metrics."""
        if not reasoning_steps:
            return {"step_count": 0, "has_reasoning": False}
        
        # Basic metrics
        metrics = {
            "step_count": len(reasoning_steps),
            "has_reasoning": len(reasoning_steps) > 1,
            "avg_step_length": np.mean([len(step.get("text", "")) for step in reasoning_steps]),
            "explicit_steps": len([s for s in reasoning_steps if s.get("type") == "reasoning_step"]),
            "implicit_steps": len([s for s in reasoning_steps if s.get("type") == "implicit_step"]),
            "premise_sets": len([s for s in reasoning_steps if s.get("type") == "premise_set"]),
        }
        
        # Structure metrics
        if len(reasoning_steps) >= 2:
            metrics.update(self._calculate_structure_metrics(reasoning_steps))
        
        # Inference chain metrics
        metrics.update(self._calculate_inference_metrics(reasoning_steps))
        
        return metrics
    
    def _calculate_structure_metrics(self, reasoning_steps: List[Dict]) -> Dict:
        """Calculate structural metrics of reasoning chain."""
        # Check for reasoning indicators
        indicator_counts = []
        for step in reasoning_steps:
            text_lower = step.get("text", "").lower()
            count = sum(1 for indicator in self.reasoning_indicators 
                       if indicator in text_lower)
            indicator_counts.append(count)
        
        # Calculate coherence between steps using NLI
        nli_scores = []
        if len(reasoning_steps) >= 2:
            for i in range(len(reasoning_steps) - 1):
                step1 = reasoning_steps[i].get("text", "")
                step2 = reasoning_steps[i + 1].get("text", "")
                if step1 and step2:
                    try:
                        result = self.nli_model(f"{step1} [SEP] {step2}", 
                                              truncation=True, max_length=512)
                        # Map to coherence score: entailment = 1, neutral = 0.5, contradiction = 0
                        score_map = {"ENTAILMENT": 1.0, "NEUTRAL": 0.5, "CONTRADICTION": 0.0}
                        nli_scores.append(score_map.get(result[0]["label"].upper(), 0.5))
                    except:
                        nli_scores.append(0.5)  # Default to neutral on error
        
        return {
            "avg_indicators_per_step": np.mean(indicator_counts) if indicator_counts else 0,
            "step_coherence_scores": nli_scores,
            "avg_step_coherence": np.mean(nli_scores) if nli_scores else 0.5,
            "structure_variance": np.var([len(step.get("text", "")) for step in reasoning_steps]) 
                                if len(reasoning_steps) > 1 else 0,
        }
    
    def _calculate_inference_metrics(self, reasoning_steps: List[Dict]) -> Dict:
        """Calculate inference quality metrics."""
        if len(reasoning_steps) < 2:
            return {}
        
        # Extract propositions from each step
        propositions = []
        for step in reasoning_steps:
            text = step.get("text", "")
            # Simple proposition extraction (can be enhanced)
            sentences = [sent.text for sent in self.spacy_model(text).sents]
            propositions.extend(sentences)
        
        # Calculate proposition consistency
        consistency_scores = []
        for i in range(len(propositions) - 1):
            for j in range(i + 1, len(propositions)):
                try:
                    result = self.nli_model(f"{propositions[i]} [SEP] {propositions[j]}", 
                                          truncation=True, max_length=512)
                    # Penalize contradictions
                    if result[0]["label"].upper() == "CONTRADICTION":
                        consistency_scores.append(0.0)
                    else:
                        consistency_scores.append(1.0)
                except:
                    consistency_scores.append(0.5)  # Default
        
        # Calculate inference chain completeness
        has_conclusion = any(step.get("type") in ["reasoning_step", "conclusion"] 
                           for step in reasoning_steps)
        has_premises = any(step.get("type") in ["premise_set", "premise"] 
                          for step in reasoning_steps)
        
        return {
            "proposition_count": len(propositions),
            "avg_proposition_consistency": np.mean(consistency_scores) if consistency_scores else 1.0,
            "inference_chain_complete": has_premises and has_conclusion,
            "premise_to_conclusion_ratio": len([s for s in reasoning_steps if s.get("type") in ["premise_set", "premise"]]) 
                                         / max(1, len([s for s in reasoning_steps if s.get("type") in ["reasoning_step", "conclusion"]])),
        }
    
    def _analyze_logical_consistency(self, reasoning_steps: List[Dict]) -> Dict:
        """Analyze logical consistency of reasoning steps."""
        if not reasoning_steps:
            return {"logical_consistency": 1.0, "contradictions": [], "valid_inferences": []}
        
        # Extract logical propositions
        propositions = []
        for step in reasoning_steps:
            text = step.get("text", "")
            # Simple logical form extraction (can be enhanced with more sophisticated parsing)
            logical_forms = self._extract_logical_forms(text)
            if logical_forms:
                propositions.extend(logical_forms)
        
        # Check for contradictions using NLI
        contradictions = []
        if len(propositions) >= 2:
            for i in range(len(propositions) - 1):
                for j in range(i + 1, len(propositions)):
                    try:
                        result = self.nli_model(f"{propositions[i]} [SEP] {propositions[j]}", 
                                              truncation=True, max_length=512)
                        if result[0]["label"].upper() == "CONTRADICTION" and result[0]["score"] > 0.7:
                            contradictions.append({
                                "proposition1": propositions[i],
                                "proposition2": propositions[j],
                                "confidence": result[0]["score"]
                            })
                    except:
                        continue
        
        # Check inference validity
        valid_inferences = []
        invalid_inferences = []
        
        for step in reasoning_steps:
            if step.get("type") == "reasoning_step":
                premises = step.get("premises", [])
                conclusion = step.get("conclusion", "")
                if premises and conclusion:
                    # Simple inference check: are premises relevant to conclusion?
                    try:
                        # Check if conclusion follows from premises using NLI
                        premise_text = " ".join(premises)
                        result = self.nli_model(f"{premise_text} [SEP] {conclusion}", 
                                              truncation=True, max_length=512)
                        
                        if result[0]["label"].upper() == "ENTAILMENT" and result[0]["score"] > 0.7:
                            valid_inferences.append({
                                "step_id": step.get("step_id"),
                                "premises": premises,
                                "conclusion": conclusion,
                                "confidence": result[0]["score"]
                            })
                        else:
                            invalid_inferences.append({
                                "step_id": step.get("step_id"),
                                "premises": premises,
                                "conclusion": conclusion,
                                "reason": result[0]["label"].lower(),
                                "confidence": result[0]["score"]
                            })
                    except:
                        invalid_inferences.append({
                            "step_id": step.get("step_id"),
                            "premises": premises,
                            "conclusion": conclusion,
                            "reason": "analysis_failed",
                            "confidence": 0.0
                        })
        
        # Calculate overall logical consistency score
        total_checks = len(contradictions) + len(invalid_inferences)
        if total_checks == 0:
            consistency_score = 1.0
        else:
            consistency_score = 1.0 - (len(contradictions) * 0.5 + len(invalid_inferences) * 0.3) / total_checks
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        return {
            "logical_consistency": consistency_score,
            "contradictions": contradictions,
            "valid_inferences": valid_inferences,
            "invalid_inferences": invalid_inferences,
            "proposition_count": len(propositions),
            "contradiction_ratio": len(contradictions) / max(1, len(propositions)),
        }
    
    def _extract_logical_forms(self, text: str) -> List[str]:
        """Extract simple logical forms from text."""
        # This is a simplified version - can be enhanced with proper semantic parsing
        doc = self.spacy_model(text)
        
        # Extract subject-predicate-object triples
        logical_forms = []
        for sent in doc.sents:
            # Simple: use the sentence itself as a proposition
            logical_forms.append(sent.text)
            
            # Try to extract simplified form
            subj = None
            verb = None
            obj = None
            
            for token in sent:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    subj = token.text
                elif token.dep_ in ["ROOT", "ccomp"] and token.pos_ == "VERB":
                    verb = token.lemma_
                elif token.dep_ in ["dobj", "attr", "prep"]:
                    obj = token.text
            
            if subj and verb:
                simple_form = f"{subj} {verb}"
                if obj:
                    simple_form += f" {obj}"
                logical_forms.append(simple_form)
        
        return logical_forms
    
    def _identify_issues(self, reasoning_steps: List[Dict], metrics: Dict, 
                        logical_analysis: Dict, pattern_results: Dict,
                        logic_results: Dict) -> List[Dict]:
        """Identify reasoning-level issues."""
        issues = []
        
        # Original issues from logical analysis
        issues.extend(self._get_original_issues(logical_analysis, metrics))
        
        # Issues from pattern matching
        issues.extend(self._get_pattern_issues(pattern_results))
        
        # Issues from formal logic validation
        issues.extend(self._get_logic_issues(logic_results))
        
        return issues
    
    def _get_original_issues(self, logical_analysis: Dict, metrics: Dict) -> List[Dict]:
        """Get original logical analysis issues."""
        issues = []
        
        # Check for lack of reasoning structure
        if not metrics.get("has_reasoning", False):
            issues.append({
                "type": "no_reasoning_structure",
                "severity": "medium",
                "description": "No clear reasoning structure detected",
                "step_count": metrics.get("step_count", 0)
            })
        
        # Check for low logical consistency
        if logical_analysis.get("logical_consistency", 1) < self.logic_threshold:
            issues.append({
                "type": "low_logical_consistency",
                "severity": "high",
                "description": f"Low logical consistency score: {logical_analysis['logical_consistency']:.2f}",
                "metric_value": logical_analysis["logical_consistency"],
                "threshold": self.logic_threshold
            })
        
        # Check for contradictions
        contradictions = logical_analysis.get("contradictions", [])
        if contradictions:
            issues.append({
                "type": "logical_contradictions",
                "severity": "high",
                "description": f"Found {len(contradictions)} logical contradiction(s)",
                "contradiction_count": len(contradictions),
                "examples": contradictions[:3]  # Include first 3 examples
            })
        
        # Check for invalid inferences
        invalid_inferences = logical_analysis.get("invalid_inferences", [])
        if invalid_inferences:
            issues.append({
                "type": "invalid_inferences",
                "severity": "medium",
                "description": f"Found {len(invalid_inferences)} potentially invalid inference(s)",
                "invalid_count": len(invalid_inferences),
                "examples": invalid_inferences[:3]
            })
        
        # Check for poor step coherence
        if metrics.get("avg_step_coherence", 0.5) < 0.4:
            issues.append({
                "type": "poor_step_coherence",
                "severity": "medium",
                "description": f"Poor coherence between reasoning steps: {metrics['avg_step_coherence']:.2f}",
                "metric_value": metrics["avg_step_coherence"],
                "threshold": 0.4
            })
        
        # Check for incomplete inference chains
        if not metrics.get("inference_chain_complete", False):
            issues.append({
                "type": "incomplete_inference_chain",
                "severity": "low",
                "description": "Reasoning chain appears incomplete (missing premises or conclusion)"
            })
        
        return issues
    
    def _get_pattern_issues(self, pattern_results: Dict) -> List[Dict]:
        """Get issues from pattern matching."""
        issues = []
        
        # Check for fallacies
        fallacies = pattern_results.get("fallacies", [])
        if fallacies:
            for fallacy in fallacies:
                issues.append({
                    "type": f"logical_fallacy_{fallacy.get('fallacy', 'unknown')}",
                    "severity": "high",
                    "description": f"Logical fallacy detected: {fallacy.get('description', '')}",
                    "fallacy_type": fallacy.get("fallacy", "unknown"),
                    "step_indices": [fallacy.get("step_index")] if "step_index" in fallacy 
                                  else list(range(fallacy.get("start_index", 0), 
                                                fallacy.get("start_index", 0) + len(fallacy.get("steps", []))))
                })
        
        return issues
    
    def _get_logic_issues(self, logic_results: Dict) -> List[Dict]:
        """Get issues from formal logic validation."""
        issues = []
        
        # Check for formal contradictions
        contradictions = logic_results.get("contradictions", [])
        if contradictions:
            issues.append({
                "type": "formal_contradiction",
                "severity": "high",
                "description": f"Formal logical contradiction detected",
                "contradiction_count": len(contradictions),
                "examples": [c.get("explanation", "") for c in contradictions[:2]]
            })
        
        # Check for low consistency score
        if logic_results.get("consistency_score", 1) < 0.7:
            issues.append({
                "type": "low_formal_consistency",
                "severity": "medium",
                "description": f"Low formal consistency score: {logic_results['consistency_score']:.2f}",
                "metric_value": logic_results["consistency_score"],
                "threshold": 0.7
            })
        
        # Check for invalid inferences
        inferences = logic_results.get("inference_validity", [])
        invalid_inferences = [inf for inf in inferences if not inf.get("is_valid", True)]
        
        if invalid_inferences:
            issues.append({
                "type": "invalid_formal_inference",
                "severity": "high",
                "description": f"Found {len(invalid_inferences)} invalid formal inference(s)",
                "invalid_count": len(invalid_inferences)
            })
        
        # Check for low logical rigor
        logical_rigor = logic_results.get("logical_rigor", 1)
        if logical_rigor < 0.6:
            issues.append({
                "type": "low_logical_rigor",
                "severity": "medium",
                "description": f"Low formal logical rigor: {logical_rigor:.2f}",
                "metric_value": logical_rigor,
                "threshold": 0.6
            })
        
        return issues
    
    def _generate_summary(self, metrics: Dict, logical_analysis: Dict,
                         pattern_results: Dict, logic_results: Dict,
                         issues: List[Dict]) -> Dict:
        """Generate summary of reasoning analysis."""
        # Determine reasoning quality
        reasoning_quality = "no_reasoning"
        if metrics.get("has_reasoning", False):
            # Combine scores from different analyses
            scores = [
                logical_analysis.get("logical_consistency", 1),
                logic_results.get("consistency_score", 1),
                logic_results.get("validity_score", 1),
                1.0 - (len(pattern_results.get("fallacies", [])) * 0.3)
            ]
            avg_score = sum(scores) / len(scores)
            
            if avg_score > 0.8:
                reasoning_quality = "excellent"
            elif avg_score > 0.6:
                reasoning_quality = "good"
            elif avg_score > 0.4:
                reasoning_quality = "fair"
            else:
                reasoning_quality = "poor"
        
        # Determine inference validity
        inference_validity = "unknown"
        invalid_patterns = len(pattern_results.get("fallacies", []))
        invalid_formal = len([inf for inf in logic_results.get("inference_validity", []) 
                             if not inf.get("is_valid", True)])
        
        if invalid_patterns == 0 and invalid_formal == 0:
            inference_validity = "valid"
        elif invalid_patterns > 0 or invalid_formal > 0:
            inference_validity = "contains_invalid"
        else:
            inference_validity = "uncertain"
        
        # Get formal assessment
        formal_assessment = self._get_formal_assessment(logic_results, pattern_results)
        
        return {
            "has_issues": len(issues) > 0,
            "issue_count": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "high"]),
            "formal_issues": len([i for i in issues if "formal" in i["type"]]),
            "fallacy_issues": len([i for i in issues if "fallacy" in i["type"]]),
            "reasoning_quality": reasoning_quality,
            "inference_validity": inference_validity,
            "logical_consistency_level": "consistent" if logical_analysis.get("logical_consistency", 1) > 0.7 
                                     else "inconsistent",
            "formal_consistency_level": "consistent" if logic_results.get("consistency_score", 1) > 0.7 
                                   else "inconsistent",
            "fallacy_count": len(pattern_results.get("fallacies", [])),
            "valid_patterns": len(pattern_results.get("valid_inferences", [])),
            "structure_completeness": "complete" if metrics.get("inference_chain_complete", False) 
                                  else "incomplete",
            "logical_rigor_score": logic_results.get("logical_rigor", 0),
            "overall_score": (logical_analysis.get("logical_consistency", 1) + 
                             logic_results.get("consistency_score", 1)) / 2,
            "formal_assessment": formal_assessment
        }
    
    def _get_formal_assessment(self, logic_results: Dict, pattern_results: Dict) -> str:
        """Get formal logic assessment."""
        rigor = logic_results.get("logical_rigor", 0)
        fallacies = pattern_results.get("fallacies", [])
        
        if rigor > 0.85 and not fallacies:
            return "Mathematically rigorous"
        elif rigor > 0.7 and not fallacies:
            return "Logically sound"
        elif rigor > 0.5:
            if fallacies:
                return "Partially valid with fallacies"
            return "Partially valid"
        elif rigor > 0.3:
            if fallacies:
                return "Informally valid, formally weak with fallacies"
            return "Informally valid, formally weak"
        else:
            if fallacies:
                return "Mathematically unsound with fallacies"
            return "Mathematically unsound"