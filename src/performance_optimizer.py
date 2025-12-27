"""
Performance optimizer for handling long responses.
"""
import time
import threading
from functools import wraps
from typing import List, Dict, Any, Callable

class TimeoutException(Exception):
    pass

def timeout(seconds=30):
    """Timeout decorator for analysis functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException(f"Function {func.__name__} timeout after {seconds} seconds")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutException(f"Function {func.__name__} timeout after {seconds} seconds")
            
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]
        return wrapper
    return decorator

class ResponseChunker:
    """Chunk long responses for analysis."""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
    
    def chunk_response(self, text: str) -> List[Dict]:
        """Split long response into manageable chunks."""
        import spacy
        
        if len(text) <= self.max_chunk_size:
            return [{"text": text, "chunk_id": 0, "is_whole": True}]
        
        # Load spaCy if available
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
        except:
            # Fallback to simple sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            
            if current_size + sentence_len > self.max_chunk_size and current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "chunk_id": len(chunks),
                    "sentence_range": (i - len(current_chunk), i - 1)
                })
                current_chunk = [sentence]
                current_size = sentence_len
            else:
                current_chunk.append(sentence)
                current_size += sentence_len
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "chunk_id": len(chunks),
                "sentence_range": (len(sentences) - len(current_chunk), len(sentences) - 1)
            })
        
        return chunks

class AnalysisOptimizer:
    """Optimize analysis for long responses."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.chunker = ResponseChunker(
            max_chunk_size=self.config.get("chunk_size", 500)
        )
    
    def optimize_reasoning_analysis(self, text: str, analyzer_func: Callable) -> Dict:
        """Run optimized reasoning analysis on long text."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # If text is short, use normal analysis
        if len(text) <= self.config.get("short_text_threshold", 800):
            return analyzer_func(text)
        
        # Chunk the response
        chunks = self.chunker.chunk_response(text)
        
        # Analyze chunks in parallel
        results = []
        issues = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_chunk = {
                executor.submit(self._analyze_chunk, chunk, analyzer_func): chunk 
                for chunk in chunks[:5]  # Limit to 5 chunks for performance
            }
            
            for future in as_completed(future_to_chunk, timeout=30):
                chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=10)
                    results.append({
                        "chunk_id": chunk["chunk_id"],
                        "result": result
                    })
                except Exception as e:
                    # Log error but continue with other chunks
                    issues.append({
                        "chunk_id": chunk["chunk_id"],
                        "error": str(e)
                    })
        
        # Combine results
        combined = self._combine_chunk_results(results, text)
        combined["chunk_issues"] = issues
        combined["was_chunked"] = True
        combined["chunk_count"] = len(chunks)
        
        return combined
    
    def _analyze_chunk(self, chunk: Dict, analyzer_func: Callable) -> Dict:
        """Analyze a single chunk with timeout protection."""
        try:
            return analyzer_func(chunk["text"])
        except Exception as e:
            return {
                "error": str(e),
                "chunk_id": chunk["chunk_id"]
            }
    
    def _combine_chunk_results(self, results: List[Dict], full_text: str) -> Dict:
        """Combine chunk analysis results."""
        if not results:
            return {"error": "No analysis results", "was_chunked": True}
        
        # Extract successful results
        successful_results = [r["result"] for r in results if "error" not in r["result"]]
        
        if not successful_results:
            return {
                "error": "All chunks failed analysis",
                "chunk_errors": [r["result"]["error"] for r in results if "error" in r["result"]],
                "was_chunked": True
            }
        
        # Combine metrics (average them)
        all_metrics = [r.get("metrics", {}) for r in successful_results]
        combined_metrics = self._average_metrics(all_metrics)
        
        # Combine issues
        all_issues = []
        for r in successful_results:
            issues = r.get("issues", [])
            for issue in issues:
                # Add chunk info to issue
                issue["from_chunk"] = True
                all_issues.append(issue)
        
        # Create summary
        summary = {
            "analyzed_in_chunks": True,
            "successful_chunks": len(successful_results),
            "total_chunks": len(results),
            "combined_metrics": combined_metrics,
            "issue_count": len(all_issues)
        }
        
        return {
            "text": full_text,
            "metrics": combined_metrics,
            "issues": all_issues,
            "summary": summary,
            "chunk_results": [r for r in results]
        }
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average metrics from multiple chunks."""
        import numpy as np
        
        if not metrics_list:
            return {}
        
        # Initialize result dict
        result = {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Average numeric values
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    values.append(val)
            
            if values:
                result[key] = float(np.mean(values))
        
        return result