"""
Cross-granular analysis: Error propagation tracking and granularity gap analysis.
"""
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
import networkx as nx
from collections import defaultdict
import json
from enum import Enum
from .factual_checker import FactualChecker

class ErrorPropagationType(Enum):
    TOKEN_TO_SENTENCE = "token_to_sentence"
    SENTENCE_TO_REASONING = "sentence_to_reasoning"
    DIRECT_TOKEN_TO_REASONING = "direct_token_to_reasoning"
    GRANULARITY_SPECIFIC = "granularity_specific"

class GranularityGapType(Enum):
    ONLY_TOKEN_DETECTABLE = "only_token_detectable"
    ONLY_SENTENCE_DETECTABLE = "only_sentence_detectable"
    ONLY_REASONING_DETECTABLE = "only_reasoning_detectable"
    CROSS_LEVEL = "cross_level"

class CrossGranularComparator:
    """
    Analyzes error propagation across granularity levels and identifies granularity gaps.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.propagation_graph = nx.DiGraph()
        self.error_trace = {}
        self.granularity_gaps = []
        self.factual_checker = FactualChecker(
            self.config.get("knowledge_base_path", "config/knowledge_base.json")
        )
        
    def compare(self, token_results: Dict, 
                sentence_results: Dict, 
                reasoning_results: Dict) -> Dict:
        """
        Perform comprehensive cross-granular comparison.
        
        Args:
            token_results: Token-level analysis results
            sentence_results: Sentence-level analysis results  
            reasoning_results: Reasoning-level analysis results
            
        Returns:
            Cross-granular analysis results
        """
        # Initialize results
        comparison_results = {
            "error_propagation": [],
            "granularity_gaps": [],
            "level_correlations": {},
            "root_cause_analysis": [],
            "propagation_graph": {},
            "summary": {}
        }
        
        # Extract issues from each level
        token_issues = self._extract_issues(token_results, "token")
        sentence_issues = self._extract_issues(sentence_results, "sentence")
        reasoning_issues = self._extract_issues(reasoning_results, "reasoning")
        
        # Track all issues
        all_issues = {
            "token": token_issues,
            "sentence": sentence_issues,
            "reasoning": reasoning_issues
        }
        
        # 1. Analyze error propagation
        propagation_patterns = self._analyze_error_propagation(
            token_issues, sentence_issues, reasoning_issues
        )
        comparison_results["error_propagation"] = propagation_patterns
        
        # 2. Identify granularity gaps
        gaps = self._identify_granularity_gaps(
            token_issues, sentence_issues, reasoning_issues
        )
        comparison_results["granularity_gaps"] = gaps
        
        # 3. Calculate level correlations
        correlations = self._calculate_level_correlations(
            token_results, sentence_results, reasoning_results
        )
        comparison_results["level_correlations"] = correlations
        
        # 4. Perform root cause analysis
        root_causes = self._analyze_root_causes(all_issues)
        comparison_results["root_cause_analysis"] = root_causes
        
        # 5. Build propagation graph
        graph_data = self._build_propagation_graph(all_issues, propagation_patterns)
        comparison_results["propagation_graph"] = graph_data
        
        # 6. Generate summary
        summary = self._generate_comparison_summary(
            all_issues, propagation_patterns, gaps, correlations
        )
        comparison_results["summary"] = summary
        
        # Store for later use
        self.error_trace = {
            "issues": all_issues,
            "propagation": propagation_patterns,
            "gaps": gaps
        }
        
        # --- NEW: factual consistency check ---
        prompt_text = sentence_results.get("prompt", "") or token_results.get("prompt", "")
        response_text = reasoning_results.get("text", "") or sentence_results.get("text", "")
        
        if prompt_text and response_text:
            factual_result = self.factual_checker.check(prompt_text, response_text)
            factual_score = factual_result.get("factual_score", 0.0)
            factual_issues = factual_result.get("issues", [])
            
            # Treat factual issues as reasoning-level issues for now
            if factual_issues:
                summary["critical_issues"] = summary.get("critical_issues", 0) + len(factual_issues)
                all_issues["reasoning"].extend([{
                    "type": "factual_inaccuracy",
                    "description": issue.get("description", ""),
                    "severity": "high" if issue.get("severity") == "high" else "medium",
                    "granularity_level": "reasoning",
                    "issue_id": f"factual_{i}"
                } for i, issue in enumerate(factual_issues)])
            
            # Add into summary and comparison results
            summary["factual_score"] = factual_score
            comparison_results["factual_score"] = factual_score
            comparison_results["factual_issues"] = factual_issues
            comparison_results["factual_checked_claims"] = factual_result.get("checked_claims", [])
            
            # Update summary with factual information
            if "issue_distribution" in summary:
                summary["issue_distribution"]["factual"] = len(factual_issues)
            else:
                summary["issue_distribution"] = {"factual": len(factual_issues)}
        
        return {
            "summary": summary,
            "issue_distribution": summary.get("issue_distribution", {}),
            "level_correlations": correlations,
            "error_propagation": propagation_patterns,
            "granularity_gaps": gaps,
            "root_cause_analysis": root_causes,
            # NEW:
            "factual_score": comparison_results.get("factual_score", 0.0),
            "factual_issues": comparison_results.get("factual_issues", []),
            "factual_checked_claims": comparison_results.get("factual_checked_claims", []),
        }
    
    def _extract_issues(self, results: Dict, level: str) -> List[Dict]:
        """Extract issues from analysis results at a specific level."""
        issues = results.get("issues", [])
        
        # Add level information to each issue
        for issue in issues:
            issue["granularity_level"] = level
            issue["issue_id"] = f"{level}_{hash(str(issue)) % 10000:04d}"
        
        return issues
    
    def _analyze_error_propagation(self, token_issues: List[Dict],
                                  sentence_issues: List[Dict],
                                  reasoning_issues: List[Dict]) -> List[Dict]:
        """
        Analyze how errors propagate across granularity levels.
        
        Returns propagation patterns with evidence.
        """
        propagation_patterns = []
        
        # Pattern 1: Token → Sentence propagation
        token_sentence_prop = self._check_token_to_sentence_propagation(
            token_issues, sentence_issues
        )
        if token_sentence_prop["has_propagation"]:
            propagation_patterns.append({
                "type": ErrorPropagationType.TOKEN_TO_SENTENCE.value,
                "description": "Token-level issues lead to sentence-level issues",
                "evidence": token_sentence_prop["evidence"],
                "confidence": token_sentence_prop["confidence"],
                "affected_issues": token_sentence_prop["affected_issues"]
            })
        
        # Pattern 2: Sentence → Reasoning propagation
        sentence_reasoning_prop = self._check_sentence_to_reasoning_propagation(
            sentence_issues, reasoning_issues
        )
        if sentence_reasoning_prop["has_propagation"]:
            propagation_patterns.append({
                "type": ErrorPropagationType.SENTENCE_TO_REASONING.value,
                "description": "Sentence-level issues lead to reasoning-level issues",
                "evidence": sentence_reasoning_prop["evidence"],
                "confidence": sentence_reasoning_prop["confidence"],
                "affected_issues": sentence_reasoning_prop["affected_issues"]
            })
        
        # Pattern 3: Direct Token → Reasoning propagation
        direct_prop = self._check_direct_token_to_reasoning_propagation(
            token_issues, reasoning_issues
        )
        if direct_prop["has_propagation"]:
            propagation_patterns.append({
                "type": ErrorPropagationType.DIRECT_TOKEN_TO_REASONING.value,
                "description": "Token-level issues directly lead to reasoning-level issues",
                "evidence": direct_prop["evidence"],
                "confidence": direct_prop["confidence"],
                "affected_issues": direct_prop["affected_issues"]
            })
        
        # Pattern 4: Granularity-specific issues
        specific_issues = self._identify_granularity_specific_issues(
            token_issues, sentence_issues, reasoning_issues
        )
        if specific_issues:
            propagation_patterns.append({
                "type": ErrorPropagationType.GRANULARITY_SPECIFIC.value,
                "description": "Issues that only appear at specific granularity levels",
                "specific_issues": specific_issues,
                "confidence": 0.8
            })
        
        return propagation_patterns
    
    def _check_token_to_sentence_propagation(self, token_issues: List[Dict],
                                            sentence_issues: List[Dict]) -> Dict:
        """Check if token issues propagate to sentence issues."""
        result = {
            "has_propagation": False,
            "evidence": [],
            "confidence": 0.0,
            "affected_issues": []
        }
        
        if not token_issues or not sentence_issues:
            return result
        
        # Look for token issues that could cause sentence issues
        evidence = []
        affected = []
        
        for t_issue in token_issues:
            # Token repetition might cause sentence coherence issues
            if t_issue.get("type") == "excessive_repetition":
                for s_issue in sentence_issues:
                    if s_issue.get("type") in ["low_coherence", "low_semantic_coherence"]:
                        evidence.append(f"Token repetition ({t_issue['type']}) → Sentence coherence issue ({s_issue['type']})")
                        affected.append({
                            "token_issue": t_issue.get("description", "")[:50],
                            "sentence_issue": s_issue.get("description", "")[:50]
                        })
            
            # Token instability might cause semantic drift
            elif t_issue.get("type") == "low_token_stability":
                for s_issue in sentence_issues:
                    if s_issue.get("type") in ["low_start_end_similarity", "high_coherence_drift"]:
                        evidence.append(f"Token instability ({t_issue['type']}) → Semantic drift ({s_issue['type']})")
                        affected.append({
                            "token_issue": t_issue.get("description", "")[:50],
                            "sentence_issue": s_issue.get("description", "")[:50]
                        })
        
        if evidence:
            result.update({
                "has_propagation": True,
                "evidence": evidence,
                "confidence": min(0.9, len(evidence) * 0.3),  # More evidence = higher confidence
                "affected_issues": affected
            })
        
        return result
    
    def _check_sentence_to_reasoning_propagation(self, sentence_issues: List[Dict],
                                                reasoning_issues: List[Dict]) -> Dict:
        """Check if sentence issues propagate to reasoning issues."""
        result = {
            "has_propagation": False,
            "evidence": [],
            "confidence": 0.0,
            "affected_issues": []
        }
        
        if not sentence_issues or not reasoning_issues:
            return result
        
        evidence = []
        affected = []
        
        for s_issue in sentence_issues:
            # Sentence contradictions might lead to logical contradictions
            if s_issue.get("type") == "semantic_contradictions":
                for r_issue in reasoning_issues:
                    if r_issue.get("type") in ["logical_contradictions", "formal_contradiction"]:
                        evidence.append(f"Sentence contradiction ({s_issue['type']}) → Logical contradiction ({r_issue['type']})")
                        affected.append({
                            "sentence_issue": s_issue.get("description", "")[:50],
                            "reasoning_issue": r_issue.get("description", "")[:50]
                        })
            
            # Low semantic coherence might lead to invalid inferences
            elif s_issue.get("type") == "low_semantic_coherence":
                for r_issue in reasoning_issues:
                    if r_issue.get("type") in ["invalid_inferences", "invalid_formal_inference"]:
                        evidence.append(f"Low semantic coherence ({s_issue['type']}) → Invalid inference ({r_issue['type']})")
                        affected.append({
                            "sentence_issue": s_issue.get("description", "")[:50],
                            "reasoning_issue": r_issue.get("description", "")[:50]
                        })
            
            # Factual errors might lead to logical errors
            elif s_issue.get("type") == "contradicted_facts":
                for r_issue in reasoning_issues:
                    if "fallacy" in r_issue.get("type", ""):
                        evidence.append(f"Factual error ({s_issue['type']}) → Logical fallacy ({r_issue['type']})")
                        affected.append({
                            "sentence_issue": s_issue.get("description", "")[:50],
                            "reasoning_issue": r_issue.get("description", "")[:50]
                        })
        
        if evidence:
            result.update({
                "has_propagation": True,
                "evidence": evidence,
                "confidence": min(0.9, len(evidence) * 0.3),
                "affected_issues": affected
            })
        
        return result
    
    def _check_direct_token_to_reasoning_propagation(self, token_issues: List[Dict],
                                                    reasoning_issues: List[Dict]) -> Dict:
        """Check for direct token-to-reasoning propagation (bypassing sentence level)."""
        result = {
            "has_propagation": False,
            "evidence": [],
            "confidence": 0.0,
            "affected_issues": []
        }
        
        if not token_issues or not reasoning_issues:
            return result
        
        evidence = []
        affected = []
        
        # Extreme token repetition might directly cause reasoning breakdown
        for t_issue in token_issues:
            if t_issue.get("type") == "excessive_repetition":
                repetition_score = t_issue.get("metric_value", 0)
                if repetition_score > 0.5:  # High repetition
                    for r_issue in reasoning_issues:
                        if r_issue.get("type") == "no_reasoning_structure":
                            evidence.append(f"High token repetition → No reasoning structure")
                            affected.append({
                                "token_issue": t_issue.get("description", "")[:50],
                                "reasoning_issue": r_issue.get("description", "")[:50]
                            })
        
        if evidence:
            result.update({
                "has_propagation": True,
                "evidence": evidence,
                "confidence": 0.7,
                "affected_issues": affected
            })
        
        return result
    
    def _identify_granularity_specific_issues(self, token_issues: List[Dict],
                                             sentence_issues: List[Dict],
                                             reasoning_issues: List[Dict]) -> Dict:
        """Identify issues that only appear at specific granularity levels."""
        specific_issues = {
            "token_only": [],
            "sentence_only": [],
            "reasoning_only": []
        }
        
        # Token-only issues
        for issue in token_issues:
            if not self._has_corresponding_issue(issue, sentence_issues + reasoning_issues):
                specific_issues["token_only"].append({
                    "issue": issue.get("description", "")[:50],
                    "type": issue.get("type"),
                    "severity": issue.get("severity", "unknown")
                })
        
        # Sentence-only issues
        for issue in sentence_issues:
            if not self._has_corresponding_issue(issue, token_issues + reasoning_issues):
                specific_issues["sentence_only"].append({
                    "issue": issue.get("description", "")[:50],
                    "type": issue.get("type"),
                    "severity": issue.get("severity", "unknown")
                })
        
        # Reasoning-only issues
        for issue in reasoning_issues:
            if not self._has_corresponding_issue(issue, token_issues + sentence_issues):
                specific_issues["reasoning_only"].append({
                    "issue": issue.get("description", "")[:50],
                    "type": issue.get("type"),
                    "severity": issue.get("severity", "unknown")
                })
        
        return specific_issues
    
    def _has_corresponding_issue(self, issue: Dict, other_issues: List[Dict]) -> bool:
        """Check if an issue has corresponding issues at other levels."""
        issue_text = issue.get("description", "").lower()
        issue_type = issue.get("type", "")
        
        for other_issue in other_issues:
            other_text = other_issue.get("description", "").lower()
            other_type = other_issue.get("type", "")
            
            # Check for similar keywords or related types
            keywords = ["contradiction", "inconsistency", "error", "invalid", "low", "poor"]
            if any(keyword in issue_text and keyword in other_text for keyword in keywords):
                return True
            
            # Check for related issue types
            type_pairs = [
                ("excessive_repetition", "low_coherence"),
                ("low_token_stability", "semantic_drift"),
                ("semantic_contradictions", "logical_contradictions"),
                ("contradicted_facts", "logical_fallacy")
            ]
            
            if (issue_type, other_type) in type_pairs or (other_type, issue_type) in type_pairs:
                return True
        
        return False
    
    def _identify_granularity_gaps(self, token_issues: List[Dict],
                                  sentence_issues: List[Dict],
                                  reasoning_issues: List[Dict]) -> List[Dict]:
        """
        Identify granularity gaps - issues detectable only at specific levels.
        """
        gaps = []
        
        # Check for reasoning-level issues not detectable at token/sentence level
        for r_issue in reasoning_issues:
            if r_issue.get("type") in ["logical_fallacy", "invalid_formal_inference", "circular_reasoning"]:
                # These are typically only detectable at reasoning level
                if not self._has_corresponding_issue(r_issue, token_issues + sentence_issues):
                    gaps.append({
                        "type": GranularityGapType.ONLY_REASONING_DETECTABLE.value,
                        "issue": r_issue.get("description", "")[:50],
                        "issue_type": r_issue.get("type"),
                        "description": "This logical issue is only detectable at reasoning level",
                        "evidence": f"Token/sentence analysis missed: {r_issue.get('type')}"
                    })
        
        # Check for token-level patterns not visible at higher levels
        for t_issue in token_issues:
            if t_issue.get("type") in ["excessive_repetition", "low_token_stability"]:
                if not self._has_corresponding_issue(t_issue, sentence_issues + reasoning_issues):
                    gaps.append({
                        "type": GranularityGapType.ONLY_TOKEN_DETECTABLE.value,
                        "issue": t_issue.get("description", "")[:50],
                        "issue_type": t_issue.get("type"),
                        "description": "This lexical pattern is only detectable at token level",
                        "evidence": f"Sentence/reasoning analysis missed: {t_issue.get('type')}"
                    })
        
        # Check for cross-level issues (detectable at multiple levels)
        cross_level = []
        all_issues = token_issues + sentence_issues + reasoning_issues
        for i, issue1 in enumerate(all_issues):
            for issue2 in all_issues[i+1:]:
                if issue1["granularity_level"] != issue2["granularity_level"]:
                    if self._are_related_issues(issue1, issue2):
                        cross_level.append({
                            "issue1": issue1.get("description", "")[:50],
                            "level1": issue1["granularity_level"],
                            "issue2": issue2.get("description", "")[:50],
                            "level2": issue2["granularity_level"],
                            "relationship": "related_cross_level"
                        })
        
        if cross_level:
            gaps.append({
                "type": GranularityGapType.CROSS_LEVEL.value,
                "cross_level_issues": cross_level[:3],  # Show first 3
                "description": "Issues detected at multiple granularity levels",
                "count": len(cross_level)
            })
        
        return gaps
    
    def _are_related_issues(self, issue1: Dict, issue2: Dict) -> bool:
        """Check if two issues from different levels are related."""
        text1 = issue1.get("description", "").lower()
        text2 = issue2.get("description", "").lower()
        
        # Check for overlapping concepts
        overlapping_words = set(text1.split()) & set(text2.split())
        if len(overlapping_words) >= 2:  # At least 2 overlapping words
            return True
        
        # Check issue type relationships
        type_relationships = {
            ("excessive_repetition", "low_coherence"): True,
            ("low_token_stability", "semantic_drift"): True,
            ("semantic_contradictions", "logical_contradictions"): True,
            ("contradicted_facts", "logical_fallacy"): True,
        }
        
        key = (issue1.get("type"), issue2.get("type"))
        return key in type_relationships or key[::-1] in type_relationships
    
    def _calculate_level_correlations(self, token_results: Dict,
                                     sentence_results: Dict,
                                     reasoning_results: Dict) -> Dict:
        """Calculate correlations between metrics at different levels."""
        correlations = {}
        
        # Extract key metrics
        token_metrics = token_results.get("metrics", {})
        sentence_metrics = sentence_results.get("metrics", {})
        reasoning_logical = reasoning_results.get("logical_analysis", {})
        reasoning_metrics = reasoning_results.get("metrics", {})
        
        # Token stability ↔ Sentence coherence correlation
        token_stability = token_metrics.get("token_stability", 0.5)
        sentence_coherence = sentence_metrics.get("enhanced_coherence", 
                                                 sentence_metrics.get("avg_coherence", 0.5))
        
        stability_corr = "positive" if abs(token_stability - sentence_coherence) < 0.2 else "negative"
        correlations["token_stability_to_sentence_coherence"] = {
            "token_stability": token_stability,
            "sentence_coherence": sentence_coherence,
            "correlation": stability_corr,
            "strength": 1 - abs(token_stability - sentence_coherence)
        }
        
        # Sentence coherence ↔ Logical consistency correlation
        logical_consistency = reasoning_logical.get("logical_consistency", 0.5)
        coherence_consistency_corr = "positive" if abs(sentence_coherence - logical_consistency) < 0.2 else "negative"
        correlations["sentence_coherence_to_logical_consistency"] = {
            "sentence_coherence": sentence_coherence,
            "logical_consistency": logical_consistency,
            "correlation": coherence_consistency_corr,
            "strength": 1 - abs(sentence_coherence - logical_consistency)
        }
        
        # Issue count correlation
        token_issue_count = len(token_results.get("issues", []))
        sentence_issue_count = len(sentence_results.get("issues", []))
        reasoning_issue_count = len(reasoning_results.get("issues", []))
        
        total_issues = token_issue_count + sentence_issue_count + reasoning_issue_count
        if total_issues > 0:
            correlations["issue_distribution"] = {
                "token_percentage": token_issue_count / total_issues,
                "sentence_percentage": sentence_issue_count / total_issues,
                "reasoning_percentage": reasoning_issue_count / total_issues,
                "dominant_level": max(["token", "sentence", "reasoning"], 
                                     key=lambda x: [token_issue_count, sentence_issue_count, 
                                                   reasoning_issue_count][["token", "sentence", "reasoning"].index(x)])
            }
        
        return correlations
    
    def _analyze_root_causes(self, all_issues: Dict) -> List[Dict]:
        """Analyze root causes of issues across granularity levels."""
        root_causes = []
        
        # Group issues by likely root cause
        token_issues = all_issues.get("token", [])
        sentence_issues = all_issues.get("sentence", [])
        reasoning_issues = all_issues.get("reasoning", [])
        
        # Check for lexical instability as root cause
        lexical_instability_issues = [i for i in token_issues 
                                     if i.get("type") in ["low_token_stability", "excessive_repetition"]]
        if lexical_instability_issues:
            downstream_issues = []
            for si in sentence_issues:
                if si.get("type") in ["low_coherence", "semantic_drift"]:
                    downstream_issues.append(si.get("description", "")[:50])
            
            if downstream_issues:
                root_causes.append({
                    "root_cause": "lexical_instability",
                    "description": "Unstable token patterns lead to coherence issues",
                    "primary_issues": [i.get("description", "")[:50] for i in lexical_instability_issues[:2]],
                    "downstream_effects": downstream_issues[:3],
                    "confidence": 0.7
                })
        
        # Check for factual errors as root cause
        factual_error_issues = [i for i in sentence_issues 
                               if i.get("type") in ["contradicted_facts", "low_fact_verification"]]
        if factual_error_issues:
            logical_issues = []
            for ri in reasoning_issues:
                if "fallacy" in ri.get("type", "") or "contradiction" in ri.get("type", ""):
                    logical_issues.append(ri.get("description", "")[:50])
            
            if logical_issues:
                root_causes.append({
                    "root_cause": "factual_inaccuracy",
                    "description": "Factual errors lead to logical inconsistencies",
                    "primary_issues": [i.get("description", "")[:50] for i in factual_error_issues[:2]],
                    "downstream_effects": logical_issues[:3],
                    "confidence": 0.8
                })
        
        # Check for structural issues as root cause
        structural_issues = [i for i in reasoning_issues 
                           if i.get("type") in ["no_reasoning_structure", "incomplete_inference_chain"]]
        if structural_issues:
            root_causes.append({
                "root_cause": "reasoning_structure_deficit",
                "description": "Lack of proper reasoning structure",
                "primary_issues": [i.get("description", "")[:50] for i in structural_issues[:2]],
                "downstream_effects": ["Affects overall argument validity"],
                "confidence": 0.6
            })
        
        return root_causes
    
    def _build_propagation_graph(self, all_issues: Dict, 
                                propagation_patterns: List[Dict]) -> Dict:
        """Build a graph representation of error propagation."""
        graph = nx.DiGraph()
        
        # Add nodes for each granularity level
        graph.add_node("token_level", type="granularity", issues=len(all_issues.get("token", [])))
        graph.add_node("sentence_level", type="granularity", issues=len(all_issues.get("sentence", [])))
        graph.add_node("reasoning_level", type="granularity", issues=len(all_issues.get("reasoning", [])))
        
        # Add edges based on propagation patterns
        for pattern in propagation_patterns:
            if pattern["type"] == ErrorPropagationType.TOKEN_TO_SENTENCE.value:
                graph.add_edge("token_level", "sentence_level", 
                             type="propagation", confidence=pattern.get("confidence", 0.5))
            elif pattern["type"] == ErrorPropagationType.SENTENCE_TO_REASONING.value:
                graph.add_edge("sentence_level", "reasoning_level",
                             type="propagation", confidence=pattern.get("confidence", 0.5))
            elif pattern["type"] == ErrorPropagationType.DIRECT_TOKEN_TO_REASONING.value:
                graph.add_edge("token_level", "reasoning_level",
                             type="direct_propagation", confidence=pattern.get("confidence", 0.5))
        
        # Convert to JSON-serializable format
        graph_data = {
            "nodes": [],
            "edges": [],
            "propagation_paths": []
        }
        
        for node in graph.nodes():
            graph_data["nodes"].append({
                "id": node,
                "type": graph.nodes[node].get("type", "unknown"),
                "issue_count": graph.nodes[node].get("issues", 0)
            })
        
        for edge in graph.edges():
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1],
                "type": graph.edges[edge].get("type", "unknown"),
                "confidence": graph.edges[edge].get("confidence", 0.5)
            })
        
        # Identify propagation paths
        if nx.has_path(graph, "token_level", "reasoning_level"):
            graph_data["propagation_paths"].append({
                "path": ["token_level", "sentence_level", "reasoning_level"],
                "description": "Full propagation path"
            })
        
        return graph_data
    
    def _generate_comparison_summary(self, all_issues: Dict,
                                    propagation_patterns: List[Dict],
                                    gaps: List[Dict],
                                    correlations: Dict) -> Dict:
        """Generate comprehensive summary of cross-granular analysis."""
        token_issues = all_issues.get("token", [])
        sentence_issues = all_issues.get("sentence", [])
        reasoning_issues = all_issues.get("reasoning", [])
        
        # Calculate overall metrics
        total_issues = len(token_issues) + len(sentence_issues) + len(reasoning_issues)
        critical_issues = len([i for i in token_issues + sentence_issues + reasoning_issues 
                              if i.get("severity") == "high"])
        
        # Determine if there's error propagation
        has_propagation = any(pattern["type"] != ErrorPropagationType.GRANULARITY_SPECIFIC.value 
                             for pattern in propagation_patterns)
        
        # Determine dominant issue level
        issue_counts = {
            "token": len(token_issues),
            "sentence": len(sentence_issues),
            "reasoning": len(reasoning_issues)
        }
        dominant_level = max(issue_counts, key=issue_counts.get)
        
        # Check for granularity gaps
        has_granularity_gaps = len(gaps) > 0
        
        # Overall assessment
        if has_propagation and critical_issues > 0:
            overall_assessment = "significant_error_propagation"
        elif has_granularity_gaps:
            overall_assessment = "granularity_gaps_present"
        elif total_issues == 0:
            overall_assessment = "no_issues_detected"
        else:
            overall_assessment = "isolated_issues"
        
        return {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "has_error_propagation": has_propagation,
            "propagation_pattern_count": len([p for p in propagation_patterns 
                                            if p["type"] != ErrorPropagationType.GRANULARITY_SPECIFIC.value]),
            "has_granularity_gaps": has_granularity_gaps,
            "gap_count": len(gaps),
            "dominant_issue_level": dominant_level,
            "issue_distribution": issue_counts,
            "overall_assessment": overall_assessment,
            "recommendations": self._generate_recommendations(
                has_propagation, has_granularity_gaps, dominant_level, critical_issues
            )
        }
    
    def _generate_recommendations(self, has_propagation: bool,
                                 has_granularity_gaps: bool,
                                 dominant_level: str,
                                 critical_issues: int) -> List[str]:
        """Generate recommendations based on cross-granular analysis."""
        recommendations = []
        
        if has_propagation:
            recommendations.append("Address root causes at lower granularity levels to prevent propagation.")
        
        if has_granularity_gaps:
            recommendations.append("Use multi-granular analysis to detect issues that single-level analysis would miss.")
        
        if dominant_level == "token" and critical_issues > 0:
            recommendations.append("Focus on improving lexical stability and token patterns.")
        elif dominant_level == "sentence" and critical_issues > 0:
            recommendations.append("Improve sentence coherence and factual accuracy.")
        elif dominant_level == "reasoning" and critical_issues > 0:
            recommendations.append("Strengthen logical reasoning and inference validity.")
        
        if not recommendations:
            recommendations.append("Response appears stable across all granularity levels.")
        
        return recommendations
    
    def generate_comparison_report(self, comparison_results: Dict) -> str:
        """Generate human-readable comparison report."""
        report = []
        report.append("=" * 70)
        report.append("CROSS-GRANULAR ANALYSIS REPORT")
        report.append("=" * 70)
        
        summary = comparison_results.get("summary", {})
        
        # Overall summary
        report.append(f"\nOVERALL SUMMARY:")
        report.append(f"  Total Issues: {summary.get('total_issues', 0)}")
        report.append(f"  Critical Issues: {summary.get('critical_issues', 0)}")
        report.append(f"  Error Propagation: {'Yes' if summary.get('has_error_propagation') else 'No'}")
        report.append(f"  Granularity Gaps: {'Yes' if summary.get('has_granularity_gaps') else 'No'}")
        report.append(f"  Dominant Issue Level: {summary.get('dominant_issue_level', 'none').title()}")
        
        # Error propagation
        propagation = comparison_results.get("error_propagation", [])
        if propagation:
            report.append(f"\nERROR PROPAGATION PATTERNS ({len(propagation)}):")
            for i, pattern in enumerate(propagation[:3], 1):  # Show first 3
                report.append(f"  {i}. {pattern.get('description', '')}")
                if pattern.get("evidence"):
                    report.append(f"     Evidence: {pattern['evidence'][0]}")
        
        # Granularity gaps
        gaps = comparison_results.get("granularity_gaps", [])
        if gaps:
            report.append(f"\nGRANULARITY GAPS ({len(gaps)}):")
            for i, gap in enumerate(gaps[:3], 1):
                report.append(f"  {i}. {gap.get('description', '')}")
                if "issue" in gap:
                    report.append(f"     Issue: {gap.get('issue', '')}")
        
        # Level correlations
        correlations = comparison_results.get("level_correlations", {})
        if correlations:
            report.append(f"\nLEVEL CORRELATIONS:")
            for key, corr in correlations.items():
                if isinstance(corr, dict):
                    report.append(f"  {key.replace('_', ' ').title()}: {corr.get('correlation', 'unknown')}")
        
        # Root cause analysis
        root_causes = comparison_results.get("root_cause_analysis", [])
        if root_causes:
            report.append(f"\nROOT CAUSE ANALYSIS ({len(root_causes)}):")
            for i, cause in enumerate(root_causes[:2], 1):
                report.append(f"  {i}. {cause.get('description', '')}")
                report.append(f"     Confidence: {cause.get('confidence', 0):.1f}")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            report.append(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                report.append(f"  {i}. {rec}")
        
        # Final assessment
        assessment_map = {
            "significant_error_propagation": "❌ SIGNIFICANT ERROR PROPAGATION DETECTED",
            "granularity_gaps_present": "⚠ GRANULARITY GAPS IDENTIFIED",
            "no_issues_detected": "✅ NO ISSUES DETECTED ACROSS LEVELS",
            "isolated_issues": "⚠ ISOLATED ISSUES, NO PROPAGATION"
        }
        
        final_assessment = assessment_map.get(
            summary.get("overall_assessment", ""),
            "ANALYSIS COMPLETE"
        )
        
        report.append("\n" + "=" * 70)
        report.append(f"FINAL ASSESSMENT: {final_assessment}")
        report.append("=" * 70)
        
        return "\n".join(report)