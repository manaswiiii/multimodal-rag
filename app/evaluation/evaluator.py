from typing import List, Dict, Any, Optional
from rouge_score import rouge_scorer
from sqlalchemy.orm import Session
from app.models import QueryHistory
import statistics

class RAGEvaluator:
    def __init__(self, db: Session):
        self.db = db
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def compute_rouge(self, answer: str, reference_chunks: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores between the generated answer and source chunks.
        This measures how much of the source content made it into the answer.
        """
        if not answer or not reference_chunks:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        reference = " ".join(reference_chunks)

        scores = self.scorer.score(reference, answer)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def evaluate_retrieval(
        self,
        retrieved_chunk_ids: List[int],
        relevant_chunk_ids: List[int],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Compute precision@k and recall@k for retrieval quality.
        relevant_chunk_ids = ground truth (chunks that should have been retrieved)
        """
        if not retrieved_chunk_ids or not relevant_chunk_ids:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0}

        retrieved_set = set(retrieved_chunk_ids[:k])
        relevant_set = set(relevant_chunk_ids)

        hits = len(retrieved_set & relevant_set)
        precision = hits / len(retrieved_set) if retrieved_set else 0.0
        recall = hits / len(relevant_set) if relevant_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "f1_at_k": round(f1, 4),
        }

    def get_latency_stats(self) -> Dict[str, Any]:
        """Compute latency statistics from query history."""
        records = self.db.query(QueryHistory).filter(
            QueryHistory.retrieval_latency_ms != None
        ).all()

        if not records:
            return {"message": "No query history yet"}

        retrieval_latencies = [r.retrieval_latency_ms for r in records]
        generation_latencies = [r.generation_latency_ms for r in records
                                if r.generation_latency_ms]

        def stats(values):
            if not values:
                return {}
            return {
                "mean_ms": round(statistics.mean(values), 2),
                "median_ms": round(statistics.median(values), 2),
                "min_ms": round(min(values), 2),
                "max_ms": round(max(values), 2),
                "stdev_ms": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            }

        return {
            "total_queries": len(records),
            "retrieval": stats(retrieval_latencies),
            "generation": stats(generation_latencies),
        }

    def get_confidence_stats(self) -> Dict[str, Any]:
        """Analyze confidence score distribution."""
        records = self.db.query(QueryHistory).filter(
            QueryHistory.confidence_score != None
        ).all()

        if not records:
            return {"message": "No query history yet"}

        scores = [r.confidence_score for r in records]
        buckets = {"high (>0.8)": 0, "medium (0.5-0.8)": 0, "low (<0.5)": 0}
        for s in scores:
            if s > 0.8:
                buckets["high (>0.8)"] += 1
            elif s >= 0.5:
                buckets["medium (0.5-0.8)"] += 1
            else:
                buckets["low (<0.5)"] += 1

        return {
            "total_queries": len(scores),
            "mean_confidence": round(statistics.mean(scores), 4),
            "distribution": buckets,
        }

    def full_report(self) -> Dict[str, Any]:
        """Generate a complete evaluation report."""
        return {
            "latency": self.get_latency_stats(),
            "confidence": self.get_confidence_stats(),
        }
