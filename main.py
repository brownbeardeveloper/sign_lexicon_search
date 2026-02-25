from core import _PROJECT_ROOT, _SIGNS_JSON
from core.config import load_config
from core.embedding import Embedding
from core.vector_store import VectorStore


def main():
    config = load_config()
    embedding = Embedding(config["Embedding"])
    embedding.load_model()
    store = VectorStore(embedding, config["Database"])

    index_dir = _PROJECT_ROOT / config.get("VectorStore", {}).get("index_dir", "data")

    if (index_dir / "signs.index").exists():
        store.load(index_dir)
    else:
        store.build_from_json(_SIGNS_JSON)
        store.save(index_dir)

    while True:
        query = input("\nSearch: ").strip()
        if not query:
            break

        results = store.search(query, k=5)
        for r in results:
            subject = r["main_subject"]
            if r["sub_subject"]:
                subject += f" > {r['sub_subject']}"
            label = f" ({subject})" if subject else ""
            print(f"  {r['rank']}. {r['word']}{label}  [id: {r['id']}, vr: {r['variant_rank']}, dist: {r['distance']:.3f}], {r['link']}")


if __name__ == "__main__":
    main()
