"""
Enhanced semantic analysis with fact-checking, entity consistency, and contradiction detection.
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
from transformers import pipeline
import spacy
from spacy import displacy
import networkx as nx
from sentence_transformers import SentenceTransformer
import wikipediaapi
import requests
from datetime import datetime
import re

class SemanticEnhancer:
    """Enhanced semantic analysis with external knowledge verification."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize semantic enhancer.
        
        Args:
            config: Configuration dictionary
                - fact_check_model: Model for factual verification
                - entity_consistency_threshold: Threshold for entity consistency
                - enable_wikipedia: Whether to use Wikipedia for fact-checking
                - contradiction_detection: Enable advanced contradiction detection
        """
        self.config = config or {}
        
        # Initialize models
        self.fact_check_model = pipeline(
            "text-classification",
            model=self.config.get("fact_check_model", "facebook/bart-large-mnli"),
            device=-1
        )
        
        self.nli_model = pipeline(
            "text-classification",
            model=self.config.get("nli_model", "facebook/bart-large-mnli"),  # FIXED LINE
            device=-1
        )
        
        self.spacy_model = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Entity tracking
        self.entity_graph = nx.Graph()
        self.entity_relations = defaultdict(list)
        self.factual_claims = []
        self.contradictions = []
        
        # Wikipedia API for fact-checking (optional)
        self.enable_wikipedia = self.config.get("enable_wikipedia", False)
        if self.enable_wikipedia:
            self.wiki = wikipediaapi.Wikipedia(
                user_agent='MultiGranularAnalyzer/1.0',
                language='en'
            )
    
    def enhance_sentence_analysis(self, sentences: List[str], 
                                 original_metrics: Dict) -> Dict:
        """
        Enhance sentence analysis with semantic verification.
        
        Args:
            sentences: List of sentences
            original_metrics: Original sentence analysis metrics
            
        Returns:
            Enhanced semantic analysis results
        """
        enhanced_metrics = original_metrics.copy()
        
        # Extract entities and their relations
        entities, entity_mentions = self._extract_entities(sentences)
        enhanced_metrics["entities"] = entities
        enhanced_metrics["entity_mentions"] = entity_mentions
        
        # Build entity consistency graph
        entity_consistency = self._analyze_entity_consistency(sentences, entities)
        enhanced_metrics["entity_consistency"] = entity_consistency
        
        # Extract factual claims
        factual_claims = self._extract_factual_claims(sentences)
        enhanced_metrics["factual_claims"] = factual_claims
        
        # Verify factual claims (basic version)
        if factual_claims:
            verification_results = self._verify_factual_claims(factual_claims)
            enhanced_metrics["fact_verification"] = verification_results
        
        # Advanced contradiction detection
        contradictions = self._detect_contradictions(sentences)
        enhanced_metrics["contradictions"] = contradictions
        
        # Semantic role consistency
        semantic_roles = self._analyze_semantic_roles(sentences)
        enhanced_metrics["semantic_roles"] = semantic_roles
        
        # Update overall coherence with semantic factors
        enhanced_metrics = self._calculate_semantic_coherence(
            enhanced_metrics, sentences
        )
        
        return enhanced_metrics
    
    def _extract_entities(self, sentences: List[str]) -> Tuple[Dict, Dict]:
        """Extract named entities and track their mentions."""
        entities = {
            "PERSON": set(),
            "ORG": set(),
            "GPE": set(),  # Geo-Political Entity
            "LOC": set(),  # Location
            "DATE": set(),
            "TIME": set(),
            "MONEY": set(),
            "PERCENT": set(),
            "QUANTITY": set(),
            "EVENT": set(),
            "WORK_OF_ART": set(),
            "LAW": set(),
            "LANGUAGE": set(),
            "NORP": set(),  # Nationalities/Religious/Political Groups
        }
        
        entity_mentions = defaultdict(list)
        
        for sent_idx, sentence in enumerate(sentences):
            doc = self.spacy_model(sentence)
            
            for ent in doc.ents:
                # Store entity by type
                if ent.label_ in entities:
                    entities[ent.label_].add(ent.text)
                
                # Track mentions with context
                entity_mentions[ent.text].append({
                    "sentence_idx": sent_idx,
                    "sentence": sentence,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "label": ent.label_,
                    "context": self._get_entity_context(doc, ent)
                })
        
        # Convert sets to lists for JSON serialization
        serializable_entities = {}
        for key, value in entities.items():
            serializable_entities[key] = list(value)
        
        return serializable_entities, dict(entity_mentions)
    
    def _get_entity_context(self, doc, entity, window_size=3) -> str:
        """Get context around an entity."""
        start = max(0, entity.start - window_size)
        end = min(len(doc), entity.end + window_size)
        
        context_tokens = []
        for i in range(start, end):
            if i == entity.start:
                context_tokens.append(f"[{doc[i].text}]")
            else:
                context_tokens.append(doc[i].text)
        
        return " ".join(context_tokens)
    
    def _analyze_entity_consistency(self, sentences: List[str], 
                                   entities: Dict) -> Dict:
        """Analyze consistency of entity mentions across sentences."""
        consistency_metrics = {
            "total_entities": sum(len(v) for v in entities.values()),
            "entity_types": len([v for v in entities.values() if v]),
            "consistency_score": 1.0,
            "inconsistencies": []
        }
        
        # Check for conflicting entity types (e.g., Paris as PERSON vs GPE)
        entity_type_map = {}
        for ent_type, ent_list in entities.items():
            for entity in ent_list:
                if entity not in entity_type_map:
                    entity_type_map[entity] = []
                entity_type_map[entity].append(ent_type)
        
        # Find entities with multiple types (potential inconsistency)
        for entity, types in entity_type_map.items():
            if len(set(types)) > 1:
                consistency_metrics["inconsistencies"].append({
                    "entity": entity,
                    "conflicting_types": list(set(types)),
                    "description": f"Entity '{entity}' appears as multiple types: {', '.join(set(types))}"
                })
        
        # Check entity reference consistency (pronouns, aliases)
        pronoun_resolution = self._check_pronoun_resolution(sentences)
        consistency_metrics["pronoun_resolution"] = pronoun_resolution
        
        # Calculate consistency score
        if consistency_metrics["total_entities"] > 0:
            inconsistency_penalty = len(consistency_metrics["inconsistencies"]) * 0.2
            pronoun_penalty = (1 - pronoun_resolution.get("resolution_rate", 1)) * 0.3
            consistency_metrics["consistency_score"] = max(0, 1 - inconsistency_penalty - pronoun_penalty)
        
        return consistency_metrics
    
    def _check_pronoun_resolution(self, sentences: List[str]) -> Dict:
        """Check pronoun resolution across sentences."""
        pronouns = ["he", "she", "it", "they", "them", "his", "her", "their", "this", "that", "these", "those"]
        
        pronoun_mentions = []
        resolved_pronouns = 0
        total_pronouns = 0
        
        for sent_idx, sentence in enumerate(sentences):
            doc = self.spacy_model(sentence)
            
            for token in doc:
                if token.text.lower() in pronouns and token.pos_ == "PRON":
                    total_pronouns += 1
                    
                    # Try to find antecedent in previous sentences
                    antecedent = self._find_antecedent(token, sentences, sent_idx)
                    
                    pronoun_info = {
                        "pronoun": token.text,
                        "sentence_idx": sent_idx,
                        "position": token.idx,
                        "has_antecedent": antecedent is not None
                    }
                    
                    if antecedent:
                        resolved_pronouns += 1
                        pronoun_info["antecedent"] = antecedent
                    
                    pronoun_mentions.append(pronoun_info)
        
        resolution_rate = resolved_pronouns / total_pronouns if total_pronouns > 0 else 1.0
        
        return {
            "total_pronouns": total_pronouns,
            "resolved_pronouns": resolved_pronouns,
            "resolution_rate": resolution_rate,
            "pronoun_details": pronoun_mentions
        }
    
    def _find_antecedent(self, pronoun, sentences: List[str], 
                        current_idx: int, lookback: int = 3) -> Optional[str]:
        """Find antecedent for a pronoun in previous sentences."""
        # Simple rule-based approach - can be enhanced with coreference resolution
        pronoun_text = pronoun.text.lower()
        
        # Look for potential antecedents in previous sentences
        for i in range(max(0, current_idx - lookback), current_idx):
            prev_doc = self.spacy_model(sentences[i])
            
            # Check for proper nouns or nouns that match pronoun type
            for token in prev_doc:
                if token.pos_ in ["PROPN", "NOUN"]:
                    # Simple gender/number matching (very basic)
                    if pronoun_text in ["he", "his", "him"] and token.text[0].isupper():
                        return token.text
                    elif pronoun_text in ["she", "her"] and token.text[0].isupper():
                        return token.text
                    elif pronoun_text in ["it", "its"]:
                        return token.text
                    elif pronoun_text in ["they", "them", "their"]:
                        return token.text
        
        return None
    
    def _extract_factual_claims(self, sentences: List[str]) -> List[Dict]:
        """Extract factual claims from sentences - IMPROVED."""
        factual_claims = []
        
        for sent_idx, sentence in enumerate(sentences):
            doc = self.spacy_model(sentence)
            
            # Only extract claims that look like factual statements
            # Skip questions, opinions, uncertain statements
            sentence_lower = sentence.lower()
            
            # Skip if it's a question
            if sentence.strip().endswith('?'):
                continue
                
            # Skip if contains hedging words
            hedging_words = ["might", "could", "possibly", "perhaps", "maybe", 
                           "I think", "I believe", "in my opinion"]
            if any(word in sentence_lower for word in hedging_words):
                continue
            
            # Skip very short sentences
            if len(sentence.split()) < 4:
                continue
            
            # Look for factual patterns
            factual_patterns = [
                "is the", "are the", "was the", "were the",
                "has a", "have a", "had a",
                "contains", "includes", "consists of",
                "located in", "situated in", "found in",
                "known for", "famous for",
                "discovered", "invented", "created"
            ]
            
            if any(pattern in sentence_lower for pattern in factual_patterns):
                factual_claims.append({
                    "sentence_idx": sent_idx,
                    "original": sentence,
                    "simplified": sentence,
                    "claim_text": sentence,
                    "confidence": 0.7,
                    "detected_by": "factual_pattern"
                })
        
        return factual_claims
    
    def _simplify_claim(self, sentence: str) -> str:
        """Simplify a sentence to its core factual claim."""
        doc = self.spacy_model(sentence)
        
        # Extract subject, verb, object
        subject = None
        verb = None
        obj = None
        
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subject = self._get_subtree_text(token)
            elif token.dep_ == "ROOT":
                verb = token.text
            elif token.dep_ in ["dobj", "attr", "prep"]:
                obj = self._get_subtree_text(token)
        
        if subject and verb:
            simplified = f"{subject} {verb}"
            if obj:
                simplified += f" {obj}"
            return simplified
        
        return sentence[:100]  # Fallback: first 100 chars
    
    def _get_subtree_text(self, token) -> str:
        """Get text of token's subtree."""
        return " ".join([t.text for t in token.subtree])
    
    def _estimate_claim_confidence(self, sentence: str) -> float:
        """Estimate confidence level of a factual claim."""
        confidence = 0.5  # Base confidence
        
        # Check for hedging words
        hedging_words = ["might", "could", "possibly", "perhaps", "maybe", "likely", "probably"]
        if any(word in sentence.lower() for word in hedging_words):
            confidence -= 0.2
        
        # Check for definitive words
        definitive_words = ["always", "never", "certainly", "definitely", "absolutely"]
        if any(word in sentence.lower() for word in definitive_words):
            confidence += 0.1
        
        # Check for citations or sources
        citation_indicators = ["according to", "studies show", "research indicates", "as per"]
        if any(indicator in sentence.lower() for indicator in citation_indicators):
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _verify_factual_claims(self, claims: List[Dict]) -> Dict:
        """Verify factual claims using NLI and external knowledge."""
        verification_results = {
            "verified_claims": [],
            "unverified_claims": [],
            "contradicted_claims": [],
            "verification_rate": 0.0
        }
        
        # Use config thresholds
        entailment_threshold = self.config.get("nli_entailment_threshold", 0.8)
        contradiction_threshold = self.config.get("nli_contradiction_threshold", 0.85)
        
        # Common knowledge facts - EXPAND WHITELIST
        common_knowledge = {
            # Geography
            "paris is the capital of france": True,
            "london is the capital of england": True,
            "tokyo is the capital of japan": True,
            "france is a country in europe": True,
            "japan is a country in asia": True,
            
            # Science - CORRECT FACTS
            "water boils at 100 degrees celsius": True,
            "water freezes at 0 degrees celsius": True,
            "the earth orbits the sun": True,
            "the earth is round": True,
            "gravity pulls objects toward earth": True,
            "plants produce oxygen": True,
            "humans need oxygen to breathe": True,
            
            # Science - COMMON ERRORS (False)
            "the earth is flat": False,
            "the sun orbits the earth": False,
            "water freezes at 100 degrees celsius": False,
            "plants absorb oxygen during day": False,
            "australia is in europe": False,
            
            # Add more as needed
        }
        
        for claim in claims:
            claim_text = claim.get("simplified", "").lower()
            
            # Check against common knowledge
            if claim_text in common_knowledge:
                if common_knowledge[claim_text]:  # True = verified fact
                    claim["verification"] = {
                        "status": "verified",
                        "method": "common_knowledge",
                        "confidence": 0.9
                    }
                    verification_results["verified_claims"].append(claim)
                else:  # False = known error
                    claim["verification"] = {
                        "status": "contradicted",
                        "method": "common_knowledge_error",
                        "confidence": 0.9
                    }
                    verification_results["contradicted_claims"].append(claim)
                continue
            
            # Use NLI with higher thresholds
            knowledge_premises = self._generate_knowledge_premises(claim_text)
            
            verified = False
            for premise in knowledge_premises:
                try:
                    result = self.nli_model(f"{premise} [SEP] {claim_text}", 
                                          truncation=True, max_length=512)
                    
                    # HIGHER THRESHOLDS
                    if result[0]["label"].upper() == "ENTAILMENT" and result[0]["score"] > entailment_threshold:
                        claim["verification"] = {
                            "status": "verified",
                            "method": "nli_inference",
                            "confidence": result[0]["score"],
                            "premise": premise
                        }
                        verification_results["verified_claims"].append(claim)
                        verified = True
                        break
                    
                    elif result[0]["label"].upper() == "CONTRADICTION" and result[0]["score"] > contradiction_threshold:
                        claim["verification"] = {
                            "status": "contradicted",
                            "method": "nli_contradiction", 
                            "confidence": result[0]["score"],
                            "premise": premise
                        }
                        verification_results["contradicted_claims"].append(claim)
                        verified = True
                        break
                        
                except:
                    continue
            
            if not verified:
                claim["verification"] = {
                    "status": "unverified",
                    "method": "insufficient_evidence",
                    "confidence": 0.3
                }
                verification_results["unverified_claims"].append(claim)
        
        # Calculate verification rate
        total_claims = len(claims)
        if total_claims > 0:
            verified_count = len(verification_results["verified_claims"])
            verification_results["verification_rate"] = verified_count / total_claims
        
        return verification_results
    
    def _generate_knowledge_premises(self, claim: str) -> List[str]:
        """Generate knowledge premises for fact verification - FIXED."""
        premises = []
        claim_lower = claim.lower()
        
        # Add specific premises for common topics
        if any(word in claim_lower for word in ["plant", "photosynthesis", "oxygen"]):
            premises.extend([
                "Plants produce oxygen during photosynthesis.",
                "Plants release oxygen as a byproduct of photosynthesis.",
                "During daylight, plants convert carbon dioxide and water into glucose and oxygen."
            ])
        
        elif any(word in claim_lower for word in ["water", "freeze", "boil", "celsius"]):
            premises.extend([
                "Water freezes at 0°C (32°F) at standard atmospheric pressure.",
                "Water boils at 100°C (212°F) at sea level.",
                "The freezing point of water is 0 degrees Celsius."
            ])
        
        elif any(word in claim_lower for word in ["gravity", "altitude", "height"]):
            premises.extend([
                "Gravity decreases with increasing altitude.",
                "The force of gravity is weaker at higher elevations.",
                "Gravity is strongest at sea level and decreases as you go higher."
            ])
        
        elif any(word in claim_lower for word in ["computer", "power", "electricity", "water"]):
            premises.extend([
                "Computers are powered by electricity.",
                "Water cooling is used to dissipate heat from computer components.",
                "The central processing unit (CPU) is the brain of a computer."
            ])
        
        elif any(word in claim_lower for word in ["dinosaur", "climate", "global warming", "flatulence"]):
            premises.extend([
                "Climate change is primarily caused by human greenhouse gas emissions.",
                "The main greenhouse gases are carbon dioxide, methane, and nitrous oxide.",
                "Dinosaur extinction was caused by an asteroid impact and volcanic activity."
            ])
        
        elif any(word in claim_lower for word in ["capital", "france", "paris", "country"]):
            premises.extend([
                "Paris is the capital city of France.",
                "France is a country in Western Europe.",
                "The capital of France has been Paris since 508 AD."
            ])
        
        # If no specific premises matched, add general ones
        if not premises:
            premises = [
                "Scientific facts and established knowledge.",
                "Commonly accepted facts in relevant field.",
                "Verified information from reliable sources."
            ]
        
        return premises
    
    def _wikipedia_lookup(self, entity: str) -> Optional[str]:
        """Look up entity on Wikipedia."""
        try:
            page = self.wiki.page(entity)
            if page.exists():
                return page.summary[:500]  # First 500 chars of summary
        except:
            pass
        return None
    
    def _detect_contradictions(self, sentences: List[str]) -> List[Dict]:
        """Detect contradictions between sentences using NLI."""
        contradictions = []
        
        if len(sentences) < 2:
            return contradictions
        
        # Use config threshold
        contradiction_threshold = self.config.get("nli_contradiction_threshold", 0.85)
        
        # Compare each sentence with every other sentence
        MAX_CONTRADICTION_PAIRS = 30
        checked_pairs = 0
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if checked_pairs >= MAX_CONTRADICTION_PAIRS:
                    break
                checked_pairs += 1
                try:
                    result = self.nli_model(
                        f"{sentences[i]} [SEP] {sentences[j]}",
                        truncation=True,
                        max_length=512
                    )
                    
                    if result[0]["label"].upper() == "CONTRADICTION" and result[0]["score"] > contradiction_threshold:
                        contradictions.append({
                            "sentence1_idx": i,
                            "sentence1": sentences[i],
                            "sentence2_idx": j,
                            "sentence2": sentences[j],
                            "confidence": result[0]["score"],
                            "type": "direct_contradiction"
                        })
                    
                    # Also check reverse direction
                    result_reverse = self.nli_model(
                        f"{sentences[j]} [SEP] {sentences[i]}",
                        truncation=True,
                        max_length=512
                    )
                    
                    if result_reverse[0]["label"].upper() == "CONTRADICTION" and result_reverse[0]["score"] > contradiction_threshold:
                        # Check if we already have this contradiction
                        existing = any(
                            c["sentence1_idx"] == j and c["sentence2_idx"] == i
                            for c in contradictions
                        )
                        if not existing:
                            contradictions.append({
                                "sentence1_idx": j,
                                "sentence1": sentences[j],
                                "sentence2_idx": i,
                                "sentence2": sentences[i],
                                "confidence": result_reverse[0]["score"],
                                "type": "direct_contradiction"
                            })
                            
                except Exception as e:
                    continue
        
        # Also check for semantic contradictions (weaker signals)
        semantic_contradictions = self._detect_semantic_contradictions(sentences)
        contradictions.extend(semantic_contradictions)
        
        return contradictions
    
    def _detect_semantic_contradictions(self, sentences: List[str]) -> List[Dict]:
        """Detect semantic contradictions using embeddings."""
        contradictions = []
        
        if len(sentences) < 2:
            return contradictions
        
        # Get sentence embeddings
        embeddings = self.embedding_model.encode(sentences)
        
        # Check for extreme dissimilarity with high confidence
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                # Very low similarity might indicate contradiction
                if similarity < 0.1:
                    # Check if sentences are about similar entities
                    doc1 = self.spacy_model(sentences[i])
                    doc2 = self.spacy_model(sentences[j])
                    
                    entities1 = set([ent.text for ent in doc1.ents])
                    entities2 = set([ent.text for ent in doc2.ents])
                    
                    common_entities = entities1.intersection(entities2)
                    
                    if common_entities:
                        contradictions.append({
                            "sentence1_idx": i,
                            "sentence1": sentences[i],
                            "sentence2_idx": j,
                            "sentence2": sentences[j],
                            "confidence": 1 - similarity,
                            "type": "semantic_contradiction",
                            "common_entities": list(common_entities),
                            "similarity_score": float(similarity)
                        })
        
        return contradictions
    
    def _analyze_semantic_roles(self, sentences: List[str]) -> Dict:
        """Analyze consistency of semantic roles across sentences."""
        semantic_roles = {
            "agents": defaultdict(list),
            "patients": defaultdict(list),
            "actions": defaultdict(list),
            "consistency_score": 1.0,
            "role_conflicts": []
        }
        
        for sent_idx, sentence in enumerate(sentences):
            doc = self.spacy_model(sentence)
            
            for token in doc:
                if token.dep_ in ["nsubj", "agent"]:  # Agent
                    agent = self._get_subtree_text(token)
                    action = token.head.text if token.head else "unknown"
                    semantic_roles["agents"][agent].append({
                        "sentence_idx": sent_idx,
                        "action": action,
                        "sentence": sentence
                    })
                
                elif token.dep_ in ["dobj", "pobj", "attr"]:  # Patient/Theme
                    patient = self._get_subtree_text(token)
                    action = token.head.text if token.head else "unknown"
                    semantic_roles["patients"][patient].append({
                        "sentence_idx": sent_idx,
                        "action": action,
                        "sentence": sentence
                    })
                
                elif token.dep_ == "ROOT" and token.pos_ == "VERB":  # Action
                    action = token.text
                    semantic_roles["actions"][action].append({
                        "sentence_idx": sent_idx,
                        "sentence": sentence
                    })
        
        # Check for role conflicts (same entity in conflicting roles)
        for entity in semantic_roles["agents"]:
            if entity in semantic_roles["patients"]:
                semantic_roles["role_conflicts"].append({
                    "entity": entity,
                    "role": "agent_and_patient",
                    "description": f"Entity '{entity}' appears as both agent and patient"
                })
        
        # Calculate consistency score
        total_roles = (
            sum(len(v) for v in semantic_roles["agents"].values()) +
            sum(len(v) for v in semantic_roles["patients"].values()) +
            sum(len(v) for v in semantic_roles["actions"].values())
        )
        
        if total_roles > 0:
            conflict_penalty = len(semantic_roles["role_conflicts"]) * 0.3
            semantic_roles["consistency_score"] = max(0, 1 - conflict_penalty)
        
        return semantic_roles
    
    def _calculate_semantic_coherence(self, metrics: Dict, 
                                     sentences: List[str]) -> Dict:
        """Calculate enhanced semantic coherence score."""
        base_coherence = metrics.get("avg_coherence", 0.5)
        
        # Get semantic factors
        entity_consistency = metrics.get("entity_consistency", {}).get("consistency_score", 1.0)
        fact_verification = metrics.get("fact_verification", {}).get("verification_rate", 1.0)
        contradiction_count = len(metrics.get("contradictions", []))
        semantic_role_consistency = metrics.get("semantic_roles", {}).get("consistency_score", 1.0)
        
        # Calculate penalty for contradictions
        contradiction_penalty = min(0.5, contradiction_count * 0.1)
        
        # Combine factors
        semantic_factors = [
            entity_consistency,
            fact_verification,
            semantic_role_consistency
        ]
        
        avg_semantic_factor = np.mean(semantic_factors) if semantic_factors else 1.0
        
        # Enhanced coherence = base coherence weighted by semantic factors
        enhanced_coherence = base_coherence * 0.6 + avg_semantic_factor * 0.4
        enhanced_coherence = enhanced_coherence * (1 - contradiction_penalty)
        
        metrics["enhanced_coherence"] = max(0, min(1, enhanced_coherence))
        metrics["semantic_factors"] = {
            "entity_consistency": entity_consistency,
            "fact_verification_rate": fact_verification,
            "contradiction_count": contradiction_count,
            "semantic_role_consistency": semantic_role_consistency,
            "contradiction_penalty": contradiction_penalty
        }
        
        return metrics
    
    def generate_semantic_report(self, enhanced_metrics: Dict) -> str:
        """Generate human-readable semantic analysis report."""
        report = []
        report.append("=" * 60)
        report.append("SEMANTIC ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Entity analysis
        entities = enhanced_metrics.get("entities", {})
        if entities:
            report.append("\nENTITIES DETECTED:")
            for ent_type, ent_list in entities.items():
                if ent_list:
                    report.append(f"  {ent_type}: {', '.join(ent_list[:5])}")
                    if len(ent_list) > 5:
                        report.append(f"    ... and {len(ent_list) - 5} more")
        
        # Entity consistency
        entity_consistency = enhanced_metrics.get("entity_consistency", {})
        if entity_consistency.get("inconsistencies"):
            report.append("\nENTITY INCONSISTENCIES:")
            for inconsistency in entity_consistency.get("inconsistencies", [])[:3]:
                report.append(f"  ⚠ {inconsistency['description']}")
        
        # Factual claims
        factual_claims = enhanced_metrics.get("factual_claims", [])
        if factual_claims:
            report.append(f"\nFACTUAL CLAIMS ({len(factual_claims)} found):")
            for i, claim in enumerate(factual_claims[:3], 1):
                report.append(f"  {i}. {claim['simplified'][:80]}...")
        
        # Fact verification
        fact_verification = enhanced_metrics.get("fact_verification", {})
        if fact_verification:
            report.append(f"\nFACT VERIFICATION:")
            report.append(f"  Verified: {len(fact_verification.get('verified_claims', []))}")
            report.append(f"  Contradicted: {len(fact_verification.get('contradicted_claims', []))}")
            report.append(f"  Unverified: {len(fact_verification.get('unverified_claims', []))}")
            report.append(f"  Verification Rate: {fact_verification.get('verification_rate', 0):.1%}")
        
        # Contradictions
        contradictions = enhanced_metrics.get("contradictions", [])
        if contradictions:
            report.append(f"\nCONTRADICTIONS DETECTED ({len(contradictions)}):")
            for i, contr in enumerate(contradictions[:2], 1):
                report.append(f"  {i}. Sentences {contr['sentence1_idx']+1} and {contr['sentence2_idx']+1}")
                report.append(f"     Confidence: {contr.get('confidence', 0):.2f}")
        
        # Semantic coherence
        report.append("\n" + "=" * 60)
        report.append("SEMANTIC COHERENCE SUMMARY")
        report.append("=" * 60)
        
        factors = enhanced_metrics.get("semantic_factors", {})
        report.append(f"Enhanced Coherence Score: {enhanced_metrics.get('enhanced_coherence', 0):.2f}/1.0")
        report.append(f"Entity Consistency: {factors.get('entity_consistency', 1):.2f}")
        report.append(f"Fact Verification Rate: {factors.get('fact_verification_rate', 1):.2f}")
        report.append(f"Semantic Role Consistency: {factors.get('semantic_role_consistency', 1):.2f}")
        report.append(f"Contradiction Penalty: -{factors.get('contradiction_penalty', 0):.2f}")
        
        # Overall assessment
        coherence = enhanced_metrics.get("enhanced_coherence", 0)
        if coherence > 0.8:
            assessment = "EXCELLENT semantic coherence"
        elif coherence > 0.6:
            assessment = "GOOD semantic coherence"
        elif coherence > 0.4:
            assessment = "FAIR semantic coherence, some issues detected"
        else:
            assessment = "POOR semantic coherence, significant issues"
        
        report.append(f"\nOVERALL ASSESSMENT: {assessment}")
        
        return "\n".join(report)