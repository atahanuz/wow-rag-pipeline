import requests

url = "https://34ce-34-186-2-62.ngrok-free.app/v1/rerank"

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

instruction = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

TEST_CASES = [
    {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is a famous landmark.",
            "Apples are a type of fruit grown on trees.",
        ],
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "documents": [
            "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
            "The Mona Lisa was painted by Leonardo da Vinci.",
            "Bananas are rich in potassium.",
            "Shakespeare was an English playwright who lived from 1564 to 1616.",
            "A solar eclipse occurs when the Moon passes between the Sun and Earth.",
        ],
    },
    {
        "query": "Best way to treat a sprained ankle at home",
        "documents": [
            "Apply the RICE method: Rest, Ice, Compression, and Elevation to reduce swelling and pain from a sprain.",
            "The capital of Japan is Tokyo.",
            "For ankle sprains, ice the area for 15-20 minutes every few hours during the first 48 hours.",
            "Python is a high-level programming language.",
            "Compression bandages help limit swelling but should not be wrapped too tightly.",
        ],
    },
]


def rerank(query, documents):
    formatted_query = query_template.format(prefix=prefix, instruction=instruction, query=query)
    formatted_docs = [document_template.format(doc=d, suffix=suffix) for d in documents]
    resp = requests.post(url, json={"query": formatted_query, "documents": formatted_docs}).json()
    return resp


def main():
    for case in TEST_CASES:
        raw_docs = case["documents"]
        resp = rerank(case["query"], raw_docs)
        print(f"\nQuery: {case['query']}")
        print("-" * 80)
        for r in resp["results"]:
            text = raw_docs[r["index"]]
            print(f"  {r['relevance_score']:.4f}  [{r['index']}]  {text}")


if __name__ == "__main__":
    main()
