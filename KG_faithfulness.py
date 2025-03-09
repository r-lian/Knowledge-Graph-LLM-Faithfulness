import requests
import json
import math
import networkx as nx
import spacy
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # Fixed import
import matplotlib.pyplot as plt

class KnowledgeGraphBuilder:
    """Build knowledge graphs from text using NLP techniques."""
    
    def __init__(self):
        # Load SpaCy model for NLP processing
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_entities_and_relations(self, text):
        """Extract entities and relations from text."""
        doc = self.nlp(text)
        entities = []
        relations = []
        
        # Extract entities (nouns and named entities)
        for entity in doc.ents:
            entities.append((entity.text, entity.label_))
        
        # Add nouns as entities if not already captured
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not any(token.text in e[0] for e in entities):
                entities.append((token.text, token.pos_))
        
        # Extract relations based on dependency parsing
        for token in doc:
            if token.dep_ in ["ROOT", "xcomp", "ccomp", "conj"]:
                continue
                
            if token.dep_ in ["nsubj", "nsubjpass", "dobj", "pobj", "attr"]:
                for child in token.head.children:
                    if child.dep_ in ["nsubj", "nsubjpass"] and token.dep_ in ["dobj", "pobj", "attr"]:
                        # Subject-verb-object relation
                        subj = child.text
                        verb = token.head.text
                        obj = token.text
                        relations.append((subj, verb, obj))
            
            # Handle prepositional relations
            if token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ == "pobj":
                        # Handle "X prep Y" relations
                        relations.append((token.head.text, token.text, child.text))
        
        return entities, relations
    
    def build_graph(self, text):
        """Build a knowledge graph from text."""
        entities, relations = self.extract_entities_and_relations(text)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add entities as nodes
        for entity, entity_type in entities:
            G.add_node(entity, type=entity_type)
        
        # Add relations as edges
        for subj, rel, obj in relations:
            if subj in G and obj in G:  # Ensure both nodes exist
                G.add_edge(subj, obj, relation=rel)
        
        return G
    
    def visualize_graph(self, G, title="Knowledge Graph"):
        """Visualize the knowledge graph."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Draw edge labels
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        return plt

class DifficultyQuantifier:
    """Quantify the difficulty of questions based on their knowledge graphs."""
    
    def __init__(self, reference_model=None, weights=None):
        self.reference_model = reference_model
        self.weights = weights or {"semantic": 0.4, "reasoning": 0.4, "ambiguity": 0.2}
        self.historical_max = {"semantic": 1.0, "reasoning": 1.0, "ambiguity": 1.0}
    
    def quantify_semantic_complexity(self, graph):
        """Quantify semantic complexity based on graph structure."""
        if not graph:
            return 0
            
        vertices = len(graph.nodes())
        edges = len(graph.edges())
        
        # Calculate diameter (if graph is not connected, take the max of connected components)
        if nx.is_connected(graph.to_undirected()):
            diameter = nx.diameter(graph.to_undirected())
        else:
            diameter = max([nx.diameter(g) for g in 
                          (graph.to_undirected().subgraph(c) for c in 
                           nx.connected_components(graph.to_undirected()))
                          if len(g) > 1], default=1)
        
        return math.log(vertices + edges + 1) * diameter
    
    def extract_operations(self, graph):
        """Extract reasoning operations from the graph."""
        operations = []
        
        # Identify reasoning paths in the graph
        try:
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target and nx.has_path(graph, source, target):
                        for path in nx.all_simple_paths(graph, source, target, cutoff=10):
                            if len(path) >= 3:  # Only consider paths of meaningful length
                                depth = len(path) - 2  # -2 to adjust for source and target
                                operations.append((path, depth))
        except nx.NetworkXError:
            # Fallback if all_simple_paths fails
            operations = []
            for u, v in graph.edges():
                operations.append(([u, v], 1))
                
        return operations
    
    def operation_complexity(self, path, graph):
        """Estimate computational complexity of a reasoning operation."""
        # Simple estimation based on path length and node degrees
        nodes = path
        complexity = 0
        
        for node in nodes:
            # Add complexity based on node degree (more connections = more complex)
            complexity += max(1, graph.degree(node) * 0.5)
            
        return complexity
    
    def quantify_reasoning_complexity(self, graph):
        """Quantify reasoning complexity based on operations."""
        if not graph or len(graph.nodes()) < 3:
            return 0
            
        operations = self.extract_operations(graph)
        complexity = 0
        
        for op, depth in operations:
            complexity += self.operation_complexity(op, graph) * depth
            
        return complexity
    
    def quantify_ambiguity(self, question, graph=None):
        """Quantify ambiguity based on graph structure or model output."""
        if self.reference_model:
            # Use language model to get answer distribution
            answer_probs = self.get_answer_distribution(question)
            entropy = -sum(p * math.log(p) for p in answer_probs.values())
            return entropy
        elif graph:
            # Estimate ambiguity from graph structure
            # More branching = more ambiguous
            branching_factor = sum(graph.out_degree(n) for n in graph.nodes()) / max(1, len(graph.nodes()))
            return min(1.0, branching_factor / 5.0)  # Normalize to [0,1]
        else:
            return 0.2  # Default ambiguity value
    
    def get_answer_distribution(self, question):
        """Get answer probability distribution from reference model."""
        # If no reference model is available, simulate distribution
        return {"answer1": 0.6, "answer2": 0.3, "answer3": 0.1}
    
    def compute_overall_difficulty(self, question, graph):
        """Compute overall difficulty score from multiple dimensions."""
        semantic = self.quantify_semantic_complexity(graph)
        reasoning = self.quantify_reasoning_complexity(graph)
        ambiguity = self.quantify_ambiguity(question, graph)
        
        # Update historical maximums
        self.historical_max["semantic"] = max(self.historical_max["semantic"], semantic)
        self.historical_max["reasoning"] = max(self.historical_max["reasoning"], reasoning)
        self.historical_max["ambiguity"] = max(self.historical_max["ambiguity"], ambiguity)
        
        # Normalize
        normalized_semantic = semantic / self.historical_max["semantic"]
        normalized_reasoning = reasoning / self.historical_max["reasoning"]
        normalized_ambiguity = ambiguity / self.historical_max["ambiguity"]
        
        # Weighted sum
        difficulty = (
            self.weights["semantic"] * normalized_semantic +
            self.weights["reasoning"] * normalized_reasoning +
            self.weights["ambiguity"] * normalized_ambiguity
        )
        
        return {
            "overall": difficulty,
            "semantic": normalized_semantic,
            "reasoning": normalized_reasoning,
            "ambiguity": normalized_ambiguity
        }

class FaithfulnessEvaluator:
    """Evaluate faithfulness by comparing question and answer knowledge graphs."""
    
    def calculate_graph_similarity(self, graph_q, graph_a):
        """Calculate similarity between question and answer knowledge graphs."""
        if not graph_q or not graph_a:
            return 0.0
            
        # Node similarity (Jaccard similarity of node sets)
        q_nodes = set(graph_q.nodes())
        a_nodes = set(graph_a.nodes())
        
        node_intersection = len(q_nodes.intersection(a_nodes))
        node_union = len(q_nodes.union(a_nodes))
        
        node_similarity = node_intersection / max(1, node_union)
        
        # Edge similarity (considering relation types)
        q_edges = set((u, v) for u, v, _ in graph_q.edges(data=True))
        a_edges = set((u, v) for u, v, _ in graph_a.edges(data=True))
        
        edge_intersection = len(q_edges.intersection(a_edges))
        edge_union = len(q_edges.union(a_edges))
        
        edge_similarity = edge_intersection / max(1, edge_union)
        
        # Combine with weights (nodes more important than edges)
        return 0.6 * node_similarity + 0.4 * edge_similarity
    
    def evaluate_faithfulness(self, question, answer, graph_q, graph_a):
        """Evaluate faithfulness of answer based on knowledge graph similarity."""
        graph_similarity = self.calculate_graph_similarity(graph_q, graph_a)
        
        # Calculate additional metrics
        relevance = self.calculate_relevance(question, answer, graph_q, graph_a)
        completeness = self.calculate_completeness(graph_q, graph_a)
        
        # Combine metrics into overall faithfulness score
        faithfulness = 0.5 * graph_similarity + 0.3 * relevance + 0.2 * completeness
        
        return {
            "faithfulness": faithfulness,
            "graph_similarity": graph_similarity,
            "relevance": relevance,
            "completeness": completeness
        }
    
    def calculate_relevance(self, question, answer, graph_q, graph_a):
        """Calculate how relevant the answer is to the question."""
        if not graph_q or not graph_a:
            return 0.0
            
        # Extract key nodes from question graph
        q_important_nodes = set()
        for node in graph_q.nodes():
            if graph_q.degree(node) > 1:  # Important nodes have multiple connections
                q_important_nodes.add(node)
        
        # Check how many important question nodes are in answer graph
        a_nodes = set(graph_a.nodes())
        overlap = len(q_important_nodes.intersection(a_nodes))
        
        return overlap / max(1, len(q_important_nodes))
    
    def calculate_completeness(self, graph_q, graph_a):
        """Calculate how complete the answer is relative to question requirements."""
        if not graph_q or not graph_a:
            return 0.0
            
        # Check what percentage of question graph is covered by answer graph
        q_nodes = set(graph_q.nodes())
        a_nodes = set(graph_a.nodes())
        
        # Calculate percentage of question nodes covered in answer
        coverage = len(q_nodes.intersection(a_nodes)) / max(1, len(q_nodes))
        
        return coverage

def main():
    """Main function to load dataset, build graphs, and evaluate faithfulness."""
    print("Loading MuSiQue dataset...")
    try:
        # Load the MuSiQue dataset
        dataset = load_dataset("dgslibisey/MuSiQue")
        print(f"Dataset loaded successfully. Size: {len(dataset['train'])}")
        
        # Initialize components
        kg_builder = KnowledgeGraphBuilder()
        difficulty_quantifier = DifficultyQuantifier()
        faithfulness_evaluator = FaithfulnessEvaluator()
        
        # Process a sample of questions
        sample_size = min(5, len(dataset['train']))
        
        # Convert NumPy integers to Python integers
        sample_indices = np.random.choice(len(dataset['train']), sample_size, replace=False)
        sample_indices = [int(idx) for idx in sample_indices]  # Convert to Python int
        
        results = []
        
        for idx in sample_indices:
            sample = dataset['train'][idx]
            question = sample['question']
            correct_answer = sample['answer']
            
            print(f"\nProcessing question: {question}")
            print(f"Correct answer: {correct_answer}")
            
            # Build knowledge graphs
            graph_q = kg_builder.build_graph(question)
            graph_a = kg_builder.build_graph(correct_answer)
            
            # Visualize graphs
            kg_builder.visualize_graph(graph_q, f"Question KG - Example {idx}")
            kg_builder.visualize_graph(graph_a, f"Answer KG - Example {idx}")
            
            # Quantify difficulty
            difficulty = difficulty_quantifier.compute_overall_difficulty(question, graph_q)
            
            # Evaluate faithfulness
            faithfulness = faithfulness_evaluator.evaluate_faithfulness(
                question, correct_answer, graph_q, graph_a
            )
            
            # Store results
            results.append({
                "question": question,
                "answer": correct_answer,
                "difficulty": difficulty,
                "faithfulness": faithfulness
            })
            
            print(f"Difficulty: {difficulty['overall']:.2f}")
            print(f"Faithfulness: {faithfulness['faithfulness']:.2f}")
        
        # Output summary
        print("\nSummary of Results:")
        avg_difficulty = sum(r["difficulty"]["overall"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness"]["faithfulness"] for r in results) / len(results)
        
        print(f"Average Question Difficulty: {avg_difficulty:.2f}")
        print(f"Average Answer Faithfulness: {avg_faithfulness:.2f}")
        
        return results
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using sample data instead...")
        
        # Sample data for demonstration
        samples = [
            {
                "question": "Which rivers both flow through Cairo and Alexandria?",
                "answer": "The Nile River flows through Cairo but not Alexandria."
            },
            {
                "question": "Who was both a US president and a Nobel Peace Prize winner?",
                "answer": "Barack Obama served as US president and won the Nobel Peace Prize in 2009."
            }
        ]
        
        # Initialize components
        kg_builder = KnowledgeGraphBuilder()
        difficulty_quantifier = DifficultyQuantifier()
        faithfulness_evaluator = FaithfulnessEvaluator()
        
        results = []
        
        for idx, sample in enumerate(samples):
            question = sample["question"]
            answer = sample["answer"]
            
            print(f"\nProcessing sample question: {question}")
            print(f"Sample answer: {answer}")
            
            # Build knowledge graphs
            graph_q = kg_builder.build_graph(question)
            graph_a = kg_builder.build_graph(answer)
            
            # Quantify difficulty
            difficulty = difficulty_quantifier.compute_overall_difficulty(question, graph_q)
            
            # Evaluate faithfulness
            faithfulness = faithfulness_evaluator.evaluate_faithfulness(
                question, answer, graph_q, graph_a
            )
            
            # Store results
            results.append({
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "faithfulness": faithfulness
            })
            
            print(f"Difficulty: {difficulty['overall']:.2f}")
            print(f"Faithfulness: {faithfulness['faithfulness']:.2f}")
        
        # Output summary
        print("\nSummary of Results:")
        avg_difficulty = sum(r["difficulty"]["overall"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness"]["faithfulness"] for r in results) / len(results)
        
        print(f"Average Question Difficulty: {avg_difficulty:.2f}")
        print(f"Average Answer Faithfulness: {avg_faithfulness:.2f}")
        
        return results

if __name__ == "__main__":
    main()