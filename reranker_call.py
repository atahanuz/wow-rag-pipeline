import requests

RERANKER_URL = "https://3b71-34-87-126-246.ngrok-free.app/v1/rerank"

_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def rerank(query, documents, instruction=DEFAULT_INSTRUCTION, top_k=None, url=RERANKER_URL):
    """Rerank `documents` against `query` using the Qwen3-Reranker server.

    Returns a list of dicts sorted by relevance descending:
        [{"index": int, "document": str, "score": float}, ...]
    """
    formatted_query = f"{_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
    formatted_docs = [f"<Document>: {d}{_SUFFIX}" for d in documents]

    payload = {"query": formatted_query, "documents": formatted_docs}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    results = [
        {"index": r["index"], "document": documents[r["index"]], "score": r["relevance_score"]}
        for r in data["results"]
    ]
    if top_k is not None:
        results = results[:top_k]
    return results


if __name__ == "__main__":
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is a famous landmark.",
        "Apples are a type of fruit grown on trees.",
    ]
    for r in rerank(query, documents):
        print(f"{r['score']:.4f}  {r['document']}")
