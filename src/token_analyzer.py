"""
Token-level analyzer for lexical stability and token pattern analysis.
"""
from typing import List, Dict, Any, Tuple , Optional
import numpy as np
from collections import Counter
from transformers import GPT2Tokenizer
from .base_analyzer import BaseAnalyzer

class TokenAnalyzer(BaseAnalyzer):
    """Analyzes LLM responses at the token level."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize token analyzer.
        
        Args:
            config: Configuration dictionary
                - tokenizer_name: Name of tokenizer to use (default: 'gpt2')
                - window_size: Size of sliding window for stability (default: 5)
                - repetition_threshold: Threshold for repetition alert (default: 0.3)
        """
        super().__init__("token", config)
        self.tokenizer_name = self.config.get("tokenizer_name", "gpt2")
        self.window_size = self.config.get("window_size", 5)
        self.repetition_threshold = self.config.get("repetition_threshold", 0.3)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
        
    def extract_components(self, text: str) -> List[str]:
        """Extract tokens from text using tokenizer."""
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text at token level.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing token analysis results
        """
        # Extract tokens
        tokens = self.extract_components(text)
        
        # Calculate metrics
        metrics = self.calculate_metrics(tokens)
        
        # Identify issues
        issues = self._identify_issues(tokens, metrics)
        
        # Compile results
        self.results = {
            "text": text,
            "tokens": tokens,
            "metrics": metrics,
            "issues": issues,
            "summary": self._generate_summary(metrics, issues)
        }
        
        return self.results
    
    def calculate_metrics(self, tokens: List[str]) -> Dict:
        """Calculate token-level metrics."""
        if not tokens:
            return {}
            
        metrics = {
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
            "vocabulary_richness": len(set(tokens)) / len(tokens) if tokens else 0,
            "token_stability": self._calculate_stability(tokens),
            "repetition_score": self._calculate_repetition_score(tokens),
            "token_distribution": dict(Counter(tokens)),
            "most_common_tokens": Counter(tokens).most_common(10),
        }
        
        # Additional advanced metrics
        metrics.update(self._calculate_advanced_metrics(tokens))
        
        return metrics
    
    def _calculate_stability(self, tokens: List[str]) -> float:
        """Calculate token sequence stability using sliding window entropy."""
        if len(tokens) < self.window_size:
            return 1.0  # Not enough tokens for meaningful analysis
            
        entropies = []
        for i in range(len(tokens) - self.window_size + 1):
            window = tokens[i:i + self.window_size]
            freq = Counter(window)
            total = len(window)
            entropy = -sum((count/total) * np.log2(count/total) 
                          for count in freq.values())
            entropies.append(entropy)
        
        avg_entropy = np.mean(entropies) if entropies else 0
        # Convert entropy to stability score (0-1, higher is more stable)
        max_possible_entropy = np.log2(min(self.window_size, len(set(tokens))))
        stability = 1 - (avg_entropy / max_possible_entropy) if max_possible_entropy > 0 else 1
        return max(0, min(1, stability))
    
    def _calculate_repetition_score(self, tokens: List[str]) -> float:
        """Calculate repetition pattern score."""
        if len(tokens) < 2:
            return 0.0
            
        repetition_count = 0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                repetition_count += 1
        
        return repetition_count / (len(tokens) - 1)
    
    def _calculate_advanced_metrics(self, tokens: List[str]) -> Dict:
        """Calculate advanced token metrics."""
        # Token type distribution
        token_types = self._classify_token_types(tokens)
        
        # Position-based analysis
        if len(tokens) >= 3:
            start_tokens = set(tokens[:len(tokens)//3])
            end_tokens = set(tokens[2*len(tokens)//3:])
            vocabulary_shift = len(end_tokens - start_tokens) / len(end_tokens) if end_tokens else 0
        else:
            vocabulary_shift = 0.0
            
        return {
            "token_types": token_types,
            "vocabulary_shift": vocabulary_shift,
            "avg_token_length": np.mean([len(t) for t in tokens]) if tokens else 0,
        }
    
    def _classify_token_types(self, tokens: List[str]) -> Dict:
        """Classify tokens into rough categories."""
        classifications = {"special": 0, "word_piece": 0, "punctuation": 0, "other": 0}
        
        for token in tokens:
            if token.startswith("Ġ"):  # GPT-2 word piece marker
                classifications["word_piece"] += 1
            elif token in [".", ",", "!", "?", ";", ":", "'", '"', "(", ")"]:
                classifications["punctuation"] += 1
            elif token.startswith("<") and token.endswith(">"):  # Special tokens
                classifications["special"] += 1
            else:
                classifications["other"] += 1
                
        # Convert to percentages
        total = len(tokens)
        if total > 0:
            for key in classifications:
                classifications[key] = classifications[key] / total
                
        return classifications
    
    def _identify_issues(self, tokens: List[str], metrics: Dict) -> List[Dict]:
        """Identify token-level issues."""
        issues = []
        
        # Check for excessive repetition
        if metrics.get("repetition_score", 0) > self.repetition_threshold:
            issues.append({
                "type": "excessive_repetition",
                "severity": "high",
                "description": f"High token repetition detected: {metrics['repetition_score']:.2f}",
                "metric_value": metrics["repetition_score"],
                "threshold": self.repetition_threshold
            })
        
        # Check for low stability
        if metrics.get("token_stability", 1) < 0.5:
            issues.append({
                "type": "low_token_stability",
                "severity": "medium",
                "description": f"Low token sequence stability: {metrics['token_stability']:.2f}",
                "metric_value": metrics["token_stability"],
                "threshold": 0.5
            })
        
        # Check for vocabulary shift
        if metrics.get("vocabulary_shift", 0) > 0.7:
            issues.append({
                "type": "high_vocabulary_shift",
                "severity": "medium",
                "description": f"High vocabulary shift between start and end: {metrics['vocabulary_shift']:.2f}",
                "metric_value": metrics["vocabulary_shift"],
                "threshold": 0.7
            })
        
        # Check for low vocabulary richness
        if metrics.get("vocabulary_richness", 1) < 0.3:
            issues.append({
                "type": "low_vocabulary_richness",
                "severity": "low",
                "description": f"Low vocabulary richness: {metrics['vocabulary_richness']:.2f}",
                "metric_value": metrics["vocabulary_richness"],
                "threshold": 0.3
            })
        
        return issues
    
    def _generate_summary(self, metrics: Dict, issues: List[Dict]) -> Dict:
        """Generate summary of token analysis."""
        return {
            "has_issues": len(issues) > 0,
            "issue_count": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "high"]),
            "overall_stability": "stable" if metrics.get("token_stability", 1) > 0.7 else "unstable",
            "repetition_level": "high" if metrics.get("repetition_score", 0) > 0.3 else "normal",
        }