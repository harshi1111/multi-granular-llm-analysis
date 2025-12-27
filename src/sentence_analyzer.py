"""
Sentence-level analyzer for semantic coherence and consistency.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from .base_analyzer import BaseAnalyzer
# Add import at top
from .semantic_enhancer import SemanticEnhancer

class SentenceAnalyzer(BaseAnalyzer):
    """Analyzes LLM responses at the sentence level."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sentence analyzer.
        
        Args:
            config: Configuration dictionary
                - model_name: Sentence transformer model (default: 'all-MiniLM-L6-v2')
                - coherence_threshold: Threshold for semantic drift (default: 0.3)
                - spacy_model: SpaCy model for sentence segmentation (default: 'en_core_web_sm')
        """
        super().__init__("sentence", config)
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.coherence_threshold = self.config.get("coherence_threshold", 0.3)
        
        # Initialize models
        self.embedding_model = SentenceTransformer(self.model_name)
        self.spacy_model = spacy.load(self.config.get("spacy_model", "en_core_web_sm"))
        
        # Initialize semantic enhancer
        self.semantic_enhancer = SemanticEnhancer(
            self.config.get("semantic_enhancer", {})
        )
        
    def extract_components(self, text: str) -> List[str]:
        """Extract sentences from text using SpaCy."""
        doc = self.spacy_model(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text at sentence level.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentence analysis results
        """
        # Extract sentences
        sentences = self.extract_components(text)
        
        # Calculate basic metrics
        metrics = self.calculate_metrics(sentences)
        
        # Apply semantic enhancement
        enhanced_metrics = self.semantic_enhancer.enhance_sentence_analysis(sentences, metrics)
        
        # Generate semantic report
        semantic_report = self.semantic_enhancer.generate_semantic_report(enhanced_metrics)
        
        # Identify issues (using enhanced metrics)
        issues = self._identify_issues(sentences, enhanced_metrics)
        
        # Compile results
        self.results = {
            "text": text,
            "sentences": sentences,
            "metrics": enhanced_metrics,  # Use enhanced metrics
            "issues": issues,
            "semantic_report": semantic_report,
            "summary": self._generate_summary(enhanced_metrics, issues)
        }
        
        return self.results
    
    def calculate_metrics(self, sentences: List[str]) -> Dict:
        """Calculate sentence-level metrics."""
        if len(sentences) < 2:
            return {
                "sentence_count": len(sentences),
                "sentences": sentences,
            }
        
        # Get sentence embeddings
        embeddings = self.embedding_model.encode(sentences)
        
        # Calculate coherence scores
        coherence_scores = self._calculate_coherence_scores(embeddings)
        
        metrics = {
            "sentence_count": len(sentences),
            "sentences": sentences,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            "coherence_scores": coherence_scores,
            "avg_coherence": np.mean(coherence_scores) if coherence_scores else 1.0,
            "min_coherence": np.min(coherence_scores) if coherence_scores else 1.0,
            "coherence_drift": self._calculate_coherence_drift(coherence_scores),
            "embeddings": embeddings.tolist(),  # Store for cross-granular analysis
        }
        
        # Additional semantic analysis
        metrics.update(self._calculate_semantic_metrics(sentences, embeddings))
        
        return metrics
    
    def _calculate_coherence_scores(self, embeddings: np.ndarray) -> List[float]:
        """Calculate coherence between consecutive sentences."""
        if len(embeddings) < 2:
            return []
        
        scores = []
        for i in range(len(embeddings) - 1):
            # Cosine similarity between consecutive sentences
            cos_sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            scores.append(float(cos_sim))
        
        return scores
    
    def _calculate_coherence_drift(self, coherence_scores: List[float]) -> float:
        """Calculate how much coherence changes throughout the text."""
        if len(coherence_scores) < 2:
            return 0.0
        
        # Fit a line to coherence scores and measure slope
        x = np.arange(len(coherence_scores))
        slope, _ = np.polyfit(x, coherence_scores, 1)
        return abs(slope)
    
    def _calculate_semantic_metrics(self, sentences: List[str], embeddings: np.ndarray) -> Dict:
        """Calculate additional semantic metrics."""
        # Topic consistency: variance of embeddings
        embedding_variance = np.var(embeddings, axis=0).mean() if len(embeddings) > 1 else 0
        
        # Beginning-end similarity
        if len(embeddings) >= 3:
            start_embedding = np.mean(embeddings[:len(embeddings)//3], axis=0)
            end_embedding = np.mean(embeddings[2*len(embeddings)//3:], axis=0)
            start_end_similarity = np.dot(start_embedding, end_embedding) / (
                np.linalg.norm(start_embedding) * np.linalg.norm(end_embedding)
            )
        else:
            start_end_similarity = 1.0
        
        # Self-similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
        
        return {
            "embedding_variance": float(embedding_variance),
            "start_end_similarity": float(start_end_similarity),
            "similarity_matrix": similarity_matrix.tolist(),
        }
    
    def _identify_issues(self, sentences: List[str], metrics: Dict) -> List[Dict]:
        """Identify sentence-level issues."""
        issues = []
        
        # Check for low enhanced coherence
        enhanced_coherence = metrics.get("enhanced_coherence", 1)
        if enhanced_coherence < 0.5:
            issues.append({
                "type": "low_semantic_coherence",
                "severity": "high",
                "description": f"Low semantic coherence: {enhanced_coherence:.2f}",
                "metric_value": enhanced_coherence,
                "threshold": 0.5
            })
        
        # Check for entity inconsistencies
        entity_consistency = metrics.get("entity_consistency", {})
        entity_inconsistencies = entity_consistency.get("inconsistencies", [])
        if entity_inconsistencies:
            issues.append({
                "type": "entity_inconsistency",
                "severity": "medium",
                "description": f"Found {len(entity_inconsistencies)} entity inconsistency(ies)",
                "inconsistency_count": len(entity_inconsistencies),
                "examples": entity_inconsistencies[:2]
            })
        
        # Check for contradictions
        contradictions = metrics.get("contradictions", [])
        if contradictions:
            issues.append({
                "type": "semantic_contradictions",
                "severity": "high",
                "description": f"Found {len(contradictions)} contradiction(s) between sentences",
                "contradiction_count": len(contradictions),
                "examples": contradictions[:2]
            })
        
        # Check for unverified factual claims
        fact_verification = metrics.get("fact_verification", {})
        verification_rate = fact_verification.get("verification_rate", 1)
        if verification_rate < 0.5:
            issues.append({
                "type": "low_fact_verification",
                "severity": "medium",
                "description": f"Low fact verification rate: {verification_rate:.1%}",
                "metric_value": verification_rate,
                "threshold": 0.5
            })
        
        # Check for contradicted claims
        contradicted_claims = fact_verification.get("contradicted_claims", [])
        if contradicted_claims:
            issues.append({
                "type": "contradicted_facts",
                "severity": "high",
                "description": f"Found {len(contradicted_claims)} contradicted factual claim(s)",
                "contradicted_count": len(contradicted_claims),
                "examples": [c.get("simplified", "")[:50] for c in contradicted_claims[:2]]
            })
        
        # Original checks (keep for backward compatibility)
        coherence_scores = metrics.get("coherence_scores", [])
        if coherence_scores:
            low_coherence_indices = [
                i for i, score in enumerate(coherence_scores) 
                if score < self.coherence_threshold
            ]
            
            for idx in low_coherence_indices:
                if idx + 1 < len(sentences):
                    issues.append({
                        "type": "low_surface_coherence",
                        "severity": "medium",
                        "description": f"Low surface coherence between sentences {idx+1} and {idx+2}",
                        "sentences": [sentences[idx], sentences[idx+1]],
                        "coherence_score": coherence_scores[idx],
                        "threshold": self.coherence_threshold,
                        "position": idx
                    })
        
        # Check for high coherence drift
        if metrics.get("coherence_drift", 0) > 0.2:
            issues.append({
                "type": "high_coherence_drift",
                "severity": "low",
                "description": f"High coherence drift detected: {metrics['coherence_drift']:.2f}",
                "metric_value": metrics["coherence_drift"],
                "threshold": 0.2
            })
        
        # Check for low start-end similarity
        if metrics.get("start_end_similarity", 1) < 0.5:
            issues.append({
                "type": "low_start_end_similarity",
                "severity": "medium",
                "description": f"Low similarity between start and end of text: {metrics['start_end_similarity']:.2f}",
                "metric_value": metrics["start_end_similarity"],
                "threshold": 0.5
            })
        
        # Check for single sentence
        if len(sentences) == 1:
            issues.append({
                "type": "single_sentence",
                "severity": "info",
                "description": "Only one sentence detected - limited coherence analysis possible"
            })
        
        return issues
    
    def _generate_summary(self, metrics: Dict, issues: List[Dict]) -> Dict:
        """Generate summary of sentence analysis."""
        enhanced_coherence = metrics.get("enhanced_coherence", 0.5)
        
        coherence_level = "excellent"
        if enhanced_coherence < 0.4:
            coherence_level = "poor"
        elif enhanced_coherence < 0.6:
            coherence_level = "fair"
        elif enhanced_coherence < 0.8:
            coherence_level = "good"
        
        # Count semantic issues separately
        semantic_issues = [i for i in issues if any(
            keyword in i["type"] for keyword in ["semantic", "entity", "contradiction", "fact"]
        )]
        
        return {
            "has_issues": len(issues) > 0,
            "issue_count": len(issues),
            "semantic_issue_count": len(semantic_issues),
            "critical_issues": len([i for i in issues if i["severity"] == "high"]),
            "coherence_level": coherence_level,
            "enhanced_coherence_score": enhanced_coherence,
            "sentence_flow": "smooth" if metrics.get("coherence_drift", 0) < 0.1 else "unstable",
            "topic_consistency": "consistent" if metrics.get("start_end_similarity", 1) > 0.7 else "divergent",
            "fact_verification_rate": metrics.get("fact_verification", {}).get("verification_rate", 1),
            "entity_consistency": metrics.get("entity_consistency", {}).get("consistency_score", 1),
        }