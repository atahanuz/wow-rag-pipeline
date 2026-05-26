export type Turn = { role: "user" | "bot"; text: string };

export type Settings = {
  dense_pool: number;
  top_n_props: number;
  rerank_k: number;
  n_paraphrases: number;
  gen_temperature: number;
  tau_cache: number;
};

export type DenseHit = {
  rank: number;
  title: string;
  text: string;
  best_prop: string;
  dense_score: number;
};

export type RerankedHit = DenseHit & { rerank_score: number };

export type Selection = {
  chosen_idx: number;
  chosen_label: string;
  chosen_title: string;
  reason: string;
  prompt: string;
  fallback: boolean;
};

export type Kedit = {
  source_title: string;
  source_text: string;
  summary: string;
  facts: string[];
  brief: string;
  prompt: string;
  fallback: boolean;
};

export type Flare = {
  triggered: boolean;
  grounded: boolean;
  unsupported_claims: string[];
  retrieval_query: string;
  re_retrieved: RerankedHit[];
  draft_answer: string;
  refined: boolean;
  prompt: string;
  fallback: boolean;
};

export type QAPair = {
  question: string;
  response_answer: string;
  knowledge_answer: string | null;
  match: boolean;
};

export type Faithfulness = {
  q2_score: number;
  q2_threshold: number;
  qa_pairs: QAPair[];
  q2_fallback: boolean;
  begin_label: "Fully Attributable" | "Not Attributable" | "Generic" | string;
  begin_rationale: string;
  begin_fallback: boolean;
  gate_failed: boolean;
  regenerated: boolean;
  runner_up_title: string | null;
  pre_gate_answer: string | null;
};

export type CacheState = {
  dialogue_id: string;
  hit: boolean;
  similarity: number;
  tau_cache: number;
  cached_title: string | null;
  cached_turn: number | null;
  size_after: number;
  max_size: number;
  flare_cache_hit: boolean;
};

export type RunResponse = {
  rewrite: { rewrite: string; paraphrases: string[] };
  dense: DenseHit[];
  reranked: RerankedHit[];
  selection: Selection | null;
  kedit: Kedit | null;
  flare: Flare | null;
  faithfulness: Faithfulness | null;
  cache: CacheState;
  dialogue_id: string;
  answer: string;
  used_titles: string[];
  prompt: string;
  timings_ms: Record<string, number>;
};

export type Info = { parents: number; propositions: number; index_dir: string };

export const DEFAULT_SETTINGS: Settings = {
  dense_pool: 10,
  top_n_props: 20,
  rerank_k: 3,
  n_paraphrases: 2,
  gen_temperature: 0.2,
  tau_cache: 0.7,
};

export async function getInfo(): Promise<Info> {
  const r = await fetch("/api/info");
  if (!r.ok) throw new Error(`info ${r.status}`);
  return r.json();
}

export async function runPipeline(
  user_turn: string,
  history: Turn[],
  settings: Settings,
  dialogue_id: string | null,
): Promise<RunResponse> {
  const r = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_turn, history, settings, dialogue_id }),
  });
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(`run ${r.status}: ${detail}`);
  }
  return r.json();
}

export async function resetDialogue(dialogue_id: string | null): Promise<{ dialogue_id: string }> {
  const url = new URL("/api/dialogue/reset", window.location.origin);
  if (dialogue_id) url.searchParams.set("dialogue_id", dialogue_id);
  const r = await fetch(url.toString(), { method: "POST" });
  if (!r.ok) throw new Error(`reset ${r.status}`);
  return r.json();
}
