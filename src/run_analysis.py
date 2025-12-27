#!/usr/bin/env python3
"""
Main script to run multi-granular analysis on LLM responses.
"""
import sys
import os
import json
import yaml
from typing import Dict, List, Any
from datetime import datetime


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


from .token_analyzer import TokenAnalyzer
from .sentence_analyzer import SentenceAnalyzer
# Add reasoning analyzer import
from .reasoning_analyzer import ReasoningAnalyzer
#from .semantic_enhancer import SemanticEnhancer
#rom .logic_validator import LogicValidator
from .cross_granular_comparator import CrossGranularComparator
from .performance_optimizer import AnalysisOptimizer, timeout


class MultiGranularAnalyzer:
    """Main orchestrator for multi-granular analysis."""
   
    def __init__(self, config_path: str = "config/analysis_config.yaml"):
        """
        Initialize the multi-granular analyzer.
       
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
       
        # Initialize analyzers
        self.token_analyzer = TokenAnalyzer(self.config.get("token_analyzer", {}))
        self.sentence_analyzer = SentenceAnalyzer(self.config.get("sentence_analyzer", {}))
        # Initialize reasoning analyzer
        self.reasoning_analyzer = ReasoningAnalyzer(self.config.get("reasoning_analyzer", {}))
        self.cross_comparator = CrossGranularComparator(self.config.get("cross_granular", {}))


        # Results storage
        self.results = {}
   
    def analyze(self, text: str, save_results: bool = True) -> Dict:
        """
        Analyze text at all granularity levels.
       
        Args:
            text: The LLM response text to analyze
            save_results: Whether to save results to file
           
        Returns:
            Complete analysis results
        """
        print(f"Analyzing text of length {len(text)} characters...")
       
        # Initialize optimizer
        optimizer = AnalysisOptimizer({
            "chunk_size": 500,
            "short_text_threshold": 800
        })
       
        # Check if text is long
        if len(text) > 1000:
            print(f"⚠️ Long response detected ({len(text)} chars). Using optimized analysis...")
           
            # Run token and sentence analysis normally
            print("Running token-level analysis...")
            token_results = self.token_analyzer.analyze(text)
           
            print("Running sentence-level analysis...")
            sentence_results = self.sentence_analyzer.analyze(text)
           
            # Use optimized reasoning analysis
            print("Running optimized reasoning-level analysis...")
            reasoning_results = optimizer.optimize_reasoning_analysis(
                text,
                self.reasoning_analyzer.analyze_optimized
            )
        else:
            # Normal analysis for short responses
            print("Running token-level analysis...")
            token_results = self.token_analyzer.analyze(text)
           
            print("Running sentence-level analysis...")
            sentence_results = self.sentence_analyzer.analyze(text)
           
            print("Running reasoning-level analysis...")
            reasoning_results = self.reasoning_analyzer.analyze(text)
       
        # Cross-granular comparison
        print("Performing cross-granular comparison...")
        cross_results = self._cross_granular_comparison(
            token_results, sentence_results, reasoning_results
        )
       
        # Compile all results
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "text_length": len(text),
                "analysis_version": "2.0",  # Updated version
                "optimized_analysis": len(text) > 1000  # Flag for optimized analysis
            },
            "token_level": token_results,
            "sentence_level": sentence_results,
            "reasoning_level": reasoning_results,
            "cross_granular": cross_results,
            "summary": self._generate_overall_summary(
                token_results, sentence_results, reasoning_results, cross_results
            )
        }
       
        # Save results if requested
        if save_results:
            self._save_results()
           
        return self.results
   
    def _cross_granular_comparison(self, token_results: Dict,
                                   sentence_results: Dict,
                                   reasoning_results: Dict) -> Dict:
        """
        Perform enhanced cross-granular comparison using CrossGranularComparator.
        """
        return self.cross_comparator.compare(
             token_results, sentence_results, reasoning_results
        )
 
   
    def _generate_overall_summary(self, token_results: Dict, sentence_results: Dict,
                                reasoning_results: Dict, cross_results: Dict) -> Dict:
        """Generate overall summary of analysis."""
        token_summary = token_results.get("summary", {})
        sentence_summary = sentence_results.get("summary", {})
        reasoning_summary = reasoning_results.get("summary", {})
       
        # Determine overall quality based on all levels
        overall_quality = "good"
       
        # Check for critical issues at any level
        critical_issues = (
            token_summary.get("critical_issues", 0) +
            len([i for i in token_results.get("issues", []) if i.get("severity") == "high"]) +
            len([i for i in sentence_results.get("issues", []) if i.get("severity") == "high"]) +
            len([i for i in reasoning_results.get("issues", []) if i.get("severity") == "high"])
        )
       
        if critical_issues > 0:
            overall_quality = "poor"
        elif (token_summary.get("has_issues", False) or
              sentence_summary.get("has_issues", False) or
              reasoning_summary.get("has_issues", False)):
            overall_quality = "needs_review"
       
        # Check reasoning quality separately (most important)
        reasoning_quality = reasoning_summary.get("reasoning_quality", "no_reasoning")
        if reasoning_quality in ["weak", "no_reasoning"]:
            overall_quality = max(overall_quality, "needs_review")  # Downgrade if reasoning is weak
       
        return {
            "overall_quality": overall_quality,
            "granularity_breakdown": {
                "token_quality": token_summary.get("overall_stability", "unknown"),
                "sentence_quality": sentence_summary.get("coherence_level", "unknown"),
                "reasoning_quality": reasoning_quality,
                "inference_validity": reasoning_summary.get("inference_validity", "unknown")
            },
            "total_issues": (len(token_results.get("issues", [])) + len(sentence_results.get("issues", [])) + len(reasoning_results.get("issues", []))),
            "critical_issues": critical_issues,
            "error_propagation": cross_results.get("propagation_patterns", {}).get("has_propagation", False),
            "granularity_specific_issues": cross_results.get("propagation_patterns", {}).get("has_granularity_specific", False),
            "recommendation": self._generate_recommendation(token_results, sentence_results, reasoning_results)
        }
   
    def _generate_recommendation(self, token_results: Dict,
                                sentence_results: Dict,
                                reasoning_results: Dict) -> str:
        """Generate recommendation based on analysis."""
        recommendations = []
       
        # Token-level recommendations
        for issue in token_results.get("issues", []):
            if issue.get("type") == "excessive_repetition":
                recommendations.append("Reduce token repetition for better readability.")
            elif issue.get("type") == "low_token_stability":
                recommendations.append("Improve token sequence stability.")
       
        # Sentence-level recommendations
        for issue in sentence_results.get("issues", []):
            if issue.get("type") == "low_coherence":
                recommendations.append("Improve coherence between sentences.")
            elif issue.get("type") == "low_start_end_similarity":
                recommendations.append("Ensure topic consistency from start to end.")
       
        # Reasoning-level recommendations
        for issue in reasoning_results.get("issues", []):
            if issue.get("type") == "low_logical_consistency":
                recommendations.append("Improve logical consistency in reasoning.")
            elif issue.get("type") == "logical_contradictions":
                recommendations.append("Resolve logical contradictions in the argument.")
            elif issue.get("type") == "invalid_inferences":
                recommendations.append("Ensure conclusions logically follow from premises.")
            elif issue.get("type") == "no_reasoning_structure":
                recommendations.append("Add clearer reasoning structure with premises and conclusions.")
       
        if not recommendations:
            return "Response appears coherent at all granularity levels."
       
        # Prioritize reasoning issues
        reasoning_recs = [r for r in recommendations if any(keyword in r.lower()
                          for keyword in ["logical", "reasoning", "inference", "premise", "conclusion"])]
        if reasoning_recs:
            return " | ".join(reasoning_recs[:2])  # Top 2 reasoning recommendations
       
        return " | ".join(recommendations[:3])  # Top 3 recommendations
   
    def _save_results(self):
        """Save analysis results to file."""
        output_dir = self.config.get("output", {}).get("save_path", "./results/")
        os.makedirs(output_dir, exist_ok=True)
       
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
       
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
       
        print(f"Results saved to: {filepath}")
   
    def analyze_file(self, filepath: str) -> Dict:
        """
        Analyze text from a file.
       
        Args:
            filepath: Path to text file
           
        Returns:
            Analysis results
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
       
        return self.analyze(text)


def main():
    """Main entry point for command line usage."""
    import argparse
   
    parser = argparse.ArgumentParser(description="Multi-granular analysis of LLM responses")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File containing text to analyze")
    parser.add_argument("--config", type=str, default="config/analysis_config.yaml",
                       help="Path to configuration file")
   
    args = parser.parse_args()
   
    # Initialize analyzer
    analyzer = MultiGranularAnalyzer(args.config)
   
    # Get text to analyze
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Use sample text if none provided
        text = """The capital of France is Paris. Paris is known for its beautiful architecture and rich history.
        However, some people believe the capital is London, which is incorrect. London is actually the capital of the United Kingdom.
        France is a country in Western Europe with a population of approximately 67 million people."""
   
    # Run analysis
    results = analyzer.analyze(text)
   
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
   
    summary = results.get("summary", {})
    print(f"Overall Quality: {summary.get('overall_quality', 'unknown')}")
    print(f"Total Issues: {summary.get('total_issues', 0)}")
    print(f"Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"\nRecommendation: {summary.get('recommendation', '')}")
   
    # Print granularity breakdown
    print(f"\nGranularity Breakdown:")
    breakdown = summary.get("granularity_breakdown", {})
    print(f"  Token Level: {breakdown.get('token_quality', 'unknown')}")
    print(f"  Sentence Level: {breakdown.get('sentence_quality', 'unknown')}")
    print(f"  Reasoning Level: {breakdown.get('reasoning_quality', 'unknown')}")
    print(f"  Inference Validity: {breakdown.get('inference_validity', 'unknown')}")
   
    # Print error propagation info
    print(f"\nError Analysis:")
    print(f"  Error Propagation: {'Yes' if summary.get('error_propagation', False) else 'No'}")
    print(f"  Granularity-Specific Issues: {'Yes' if summary.get('granularity_specific_issues', False) else 'No'}")
   
    print("\n" + "="*60)
    print("Analysis complete. Check results directory for detailed report.")
   
if __name__ == "__main__":
    main() 