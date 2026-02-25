import json
import faiss
from pathlib import Path

from .embedding import Embedding

class VectorStore:
    def __init__(self, embedding: Embedding, vector_db_config: dict):
        self.embedding = embedding
        self.index_filename: str = vector_db_config["index_filename"]
        self.metadata_filename: str = vector_db_config["metadata_filename"]
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build_from_json(self, json_path: str | Path) -> None:
        """Load signs.json, embed each word + subjects and build FAISS index"""
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            signs = json.load(f)

        self.metadata = []
        texts: list[str] = []

        for sign in signs:
            word = sign.get("word", "")
            main_subject = sign.get("main_subject", "")
            sub_subject = sign.get("sub_subject", "")

            # Build the text to embed
            text = word
            if main_subject:
                text += f" ({main_subject})"
                if sub_subject:
                    text += f" ({sub_subject})"

            texts.append(text)
            self.metadata.append({
                "id": sign["id"],
                "word": word,
                "variant_rank": sign.get("variant_rank", 999),
                "main_subject": main_subject,
                "sub_subject": sub_subject,
                "link": sign.get("media__main_video", ""),
            })

        print(f"Encoding {len(texts)} signs...")
        embeddings = self.embedding.encode_batch(texts)

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Index built — {self.index.ntotal} vectors, dim={embeddings.shape[1]}")

    def save(self, directory: str | Path) -> None:
        """Write FAISS index + metadata JSON to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / self.index_filename))
        with open(directory / self.metadata_filename, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"Saved index & metadata to {directory}/")

    def load(self, directory: str | Path) -> None:
        """Read FAISS index + metadata JSON from *directory*."""
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / self.index_filename))
        with open(directory / self.metadata_filename, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loaded index ({self.index.ntotal} vectors) from {directory}/")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the *k* closest signs to *query*."""
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_from_json() or load() first.")

        # Keyword fallback: exact matches first (space-insensitive)
        query_norm = query.lower().replace(" ", "")
        exact = sorted(
            [m.copy() for m in self.metadata if m["word"].lower().replace(" ", "") == query_norm],
            key=lambda m: m["variant_rank"],
        )

        # Semantic search fills remaining slots
        query_vec = self.embedding.get_embedding(query).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)

        seen_ids = {e["id"] for e in exact}
        semantic = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                break
            entry = self.metadata[idx].copy()
            if entry["id"] not in seen_ids:
                entry["distance"] = float(dist)
                semantic.append(entry)
                seen_ids.add(entry["id"])

        # Merge: exact hits (dist=1.0) + semantic, capped at k
        results = []
        for e in exact:
            e["distance"] = 1.0
        combined = exact[:k] + semantic[:k - len(exact[:k])]
        for rank, entry in enumerate(combined, start=1):
            entry["rank"] = rank
            results.append(entry)
        return results
