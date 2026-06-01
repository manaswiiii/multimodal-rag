import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
FAISS_INDEX_PATH = "data/processed/faiss.index"
FAISS_ID_MAP_PATH = "data/processed/faiss_id_map.npy"

class EmbeddingManager:
    def __init__(self):
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.id_map = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        if Path(FAISS_INDEX_PATH).exists() and Path(FAISS_ID_MAP_PATH).exists():
            print("📂 Loading existing FAISS index...")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.id_map = np.load(FAISS_ID_MAP_PATH).tolist()
            print(f"  ✅ Loaded index with {self.index.ntotal} vectors")
        else:
            print("🆕 Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.id_map = []
            print("  ✅ Fresh index created")

    def embed_chunks(self, chunks: list) -> list:
        if not chunks:
            return []
        texts = [chunk.content for chunk in chunks]
        print(f"  🔢 Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        start_id = self.index.ntotal
        self.index.add(embeddings)
        faiss_ids = []
        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            self.id_map.append(chunk.id)
            faiss_ids.append(str(faiss_id))
        self._save_index()
        print(f"  ✅ FAISS index now has {self.index.ntotal} vectors")
        return faiss_ids

    def _save_index(self):
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        np.save(FAISS_ID_MAP_PATH, np.array(self.id_map))

    def search(self, query: str, top_k: int = 5) -> list:
        if self.index.ntotal == 0:
            return []
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_db_id = self.id_map[idx]
            results.append((chunk_db_id, float(dist)))
        return results
