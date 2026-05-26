"""Extract unique (title, paragraph) passages from the WoW dataset into corpus.jsonl."""
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SOURCES = ["train.json", "valid_random_split.json", "test_random_split.json",
           "valid_topic_split.json", "test_topic_split.json"]
OUT_PATH = os.path.join(DATA_DIR, "corpus.jsonl")


def iter_passages(dialogues):
    for d in dialogues:
        ctp = d.get("chosen_topic_passage")
        topic = d.get("chosen_topic")
        if ctp and topic:
            yield topic, " ".join(ctp)
        for turn in d.get("dialog", []):
            for rp in turn.get("retrieved_passages", []):
                for title, sentences in rp.items():
                    if sentences:
                        yield title, " ".join(sentences)


def main():
    seen = set()
    n_in = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for fname in SOURCES:
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f"skip missing {fname}")
                continue
            print(f"loading {fname} ...", flush=True)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for title, text in iter_passages(data):
                n_in += 1
                key = (title, text)
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
            print(f"  processed; running unique={len(seen)} from {n_in} raw", flush=True)

    print(f"\nWrote {len(seen)} unique passages to {OUT_PATH}")


if __name__ == "__main__":
    main()
