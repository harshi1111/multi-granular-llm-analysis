"""
Base analyzer class defining the interface for all granularity analyzers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json

class BaseAnalyzer(ABC):
    """Abstract base class for all granularity analyzers."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize the analyzer.
        
        Args:
            name: Name of the analyzer (e.g., 'token', 'sentence', 'reasoning')
            config: Configuration dictionary for the analyzer
        """
        self.name = name
        self.config = config or {}
        self.results = {}
        
    @abstractmethod
    def analyze(self, input_data: Any) -> Dict:
        """
        Analyze the input data at this granularity level.
        
        Args:
            input_data: The data to analyze (text, tokens, sentences, etc.)
            
        Returns:
            Dictionary containing analysis results and metrics
        """
        pass
    
    @abstractmethod
    def extract_components(self, text: str) -> List:
        """
        Extract components at this granularity level from text.
        
        Args:
            text: Raw text input
            
        Returns:
            List of components (tokens, sentences, reasoning steps)
        """
        pass
    
    def calculate_metrics(self, components: List) -> Dict:
        """
        Calculate metrics for the extracted components.
        
        Args:
            components: List of components at this granularity level
            
        Returns:
            Dictionary of metrics
        """
        return {
            "component_count": len(components),
            "components": components,
        }
    
    def save_results(self, filepath: str):
        """Save analysis results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filepath: str):
        """Load analysis results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    def get_issues(self) -> List[Dict]:
        """
        Extract identified issues from analysis results.
        
        Returns:
            List of issue dictionaries with description and severity
        """
        return self.results.get("issues", [])