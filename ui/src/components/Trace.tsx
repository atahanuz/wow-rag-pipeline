import { useState } from "react";
import type { RunResponse } from "../api";

function StageHeader({
  stage,
  title,
  ms,
}: {
  stage: string;
  title: string;
  ms?: number;
}) {
  return (
    <div className="mb-2 flex items-center justify-between">
      <div className="flex items-baseline gap-2">
        <span className="stage-h">{stage}</span>
        <span className="text-sm font-medium text-ink-700">{title}</span>
      </div>
      {ms !== undefined && (
        <span className="font-mono text-[11px] text-ink-400">{ms.toFixed(0)} ms</span>
      )}
    </div>
  );
}

function ScoreBar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-100">
      <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Trace({ result }: { result: RunResponse }) {
  const [showPrompt, setShowPrompt] = useState(false);
  const t = result.timings_ms;

  return (
    <div className="space-y-6">
      {/* Ext 3: per-dialogue passage cache banner */}
      {result.cache && (
        <section>
          <div
            className={
              "card flex flex-wrap items-baseline gap-2 p-3 " +
              (result.cache.hit ? "border-accent-ring bg-accent-soft/40" : "")
            }
          >
            <span className="stage-h">Ext 3 · Cache</span>
            {result.cache.hit ? (
              <span className="pill-accent">
                HIT · skipped dense / rerank / GenKS
              </span>
            ) : (
              <span className="pill border-ink-200 bg-ink-50 text-ink-600">miss</span>
            )}
            <span className="text-xs text-ink-500">
              sim{" "}
              <span className="font-mono font-semibold text-ink-800">
                {result.cache.similarity.toFixed(3)}
              </span>
              <span className="text-ink-400"> / τ {result.cache.tau_cache.toFixed(2)}</span>
            </span>
            {result.cache.cached_title && (
              <span className="text-xs text-ink-500">
                served{" "}
                <span className="font-medium text-ink-800">{result.cache.cached_title}</span>
                {result.cache.cached_turn !== null && (
                  <span className="text-ink-400"> (added turn {result.cache.cached_turn})</span>
                )}
              </span>
            )}
            <span className="ml-auto text-[11px] text-ink-400">
              {result.cache.size_after} / {result.cache.max_size} cached
            </span>
            {result.cache.flare_cache_hit && (
              <span className="pill-accent">FLARE cache hit</span>
            )}
          </div>
        </section>
      )}

      {/* Stage 1: rewrite */}
      <section>
        <StageHeader stage="Stage 1" title="Query rewrite" ms={t.rewrite} />
        <div className="card p-4">
          <div className="text-xs text-ink-500">Rewrite</div>
          <code className="mt-1 block break-words rounded-md bg-ink-50 px-2 py-1.5 font-mono text-sm text-ink-800">
            {result.rewrite.rewrite}
          </code>
          {result.rewrite.paraphrases.length > 0 && (
            <div className="mt-3 space-y-1.5">
              <div className="text-xs text-ink-500">Paraphrases</div>
              {result.rewrite.paraphrases.map((p, i) => (
                <code
                  key={i}
                  className="block break-words rounded-md bg-ink-50 px-2 py-1.5 font-mono text-sm text-ink-700"
                >
                  {p}
                </code>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Stage 2: dense (skipped on cache hit) */}
      {result.dense.length > 0 && (
      <section>
        <StageHeader stage="Stage 2" title={`Dense retrieval · pool top-${result.dense.length}`} ms={t.dense} />
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-ink-50 text-[11px] uppercase tracking-wider text-ink-500">
              <tr>
                <th className="px-3 py-2 text-left font-medium">#</th>
                <th className="px-3 py-2 text-left font-medium">Dense</th>
                <th className="px-3 py-2 text-left font-medium">Title</th>
                <th className="px-3 py-2 text-left font-medium">Best proposition</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-100">
              {result.dense.map((h) => (
                <tr key={h.rank} className="hover:bg-ink-50/60">
                  <td className="px-3 py-2 font-mono text-xs text-ink-400">{h.rank}</td>
                  <td className="px-3 py-2 font-mono text-xs text-ink-700">
                    {h.dense_score.toFixed(3)}
                  </td>
                  <td className="px-3 py-2 font-medium text-ink-800">{h.title}</td>
                  <td className="px-3 py-2 text-ink-600">{h.best_prop}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      )}

      {/* Stage 3: rerank (skipped on cache hit) */}
      {result.reranked.length > 0 && (
      <section>
        <StageHeader stage="Stage 3" title={`Cross-encoder reranker · top-${result.reranked.length}`} ms={t.rerank} />
        <div className="space-y-2.5">
          {result.reranked.map((h) => {
            const isChosen = result.selection?.chosen_idx === h.rank - 1;
            return (
              <div
                key={h.rank}
                className={
                  "card p-4 transition " +
                  (isChosen ? "border-accent ring-2 ring-accent-soft" : "")
                }
              >
                <div className="flex items-baseline justify-between">
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-xs text-ink-400">[{h.rank}]</span>
                    <h3 className="text-sm font-semibold text-ink-900">{h.title}</h3>
                    {isChosen && (
                      <span className="pill-accent">chosen by GenKS</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-[11px]">
                    <span className="text-ink-500">
                      rerank{" "}
                      <span className="font-mono font-semibold text-ink-800">
                        {h.rerank_score.toFixed(4)}
                      </span>
                    </span>
                    <span className="text-ink-400">
                      dense{" "}
                      <span className="font-mono text-ink-600">{h.dense_score.toFixed(3)}</span>
                    </span>
                  </div>
                </div>
                <div className="mt-2">
                  <ScoreBar value={h.rerank_score} />
                </div>
                <p className="mt-3 text-xs italic text-ink-500">{h.best_prop}</p>
                <p className="mt-2 text-sm leading-relaxed text-ink-700">{h.text}</p>
              </div>
            );
          })}
        </div>
      </section>
      )}

      {/* Stage 4: GenKS selection */}
      {result.selection && (
        <section>
          <StageHeader stage="Stage 4" title="GenKS · knowledge selection" ms={t.genks} />
          <div className="card p-4">
            <div className="flex items-baseline gap-2">
              <span className="pill-accent font-mono">
                {result.selection.chosen_label}
              </span>
              <span className="text-sm font-medium text-ink-800">
                {result.selection.chosen_title}
              </span>
              {result.selection.fallback && (
                <span className="pill border-amber-300 bg-amber-50 text-amber-700">
                  fallback (parse failed)
                </span>
              )}
            </div>
            <p className="mt-2 text-sm leading-relaxed text-ink-600">
              <span className="text-ink-400">reason:</span> {result.selection.reason || "(none)"}
            </p>
          </div>
        </section>
      )}

      {/* Stage 5: KEDiT distillation */}
      {result.kedit && (
        <section>
          <StageHeader stage="Stage 5" title="KEDiT · knowledge distillation" ms={t.kedit} />
          <div className="card p-4">
            <div className="flex items-baseline justify-between">
              <p className="text-sm font-medium text-ink-800">
                {result.kedit.summary || "(no summary)"}
              </p>
              {result.kedit.fallback && (
                <span className="pill border-amber-300 bg-amber-50 text-amber-700">
                  fallback (used raw passage)
                </span>
              )}
            </div>
            {result.kedit.facts.length > 0 && (
              <ul className="mt-3 space-y-1.5 text-sm leading-relaxed text-ink-700">
                {result.kedit.facts.map((f, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="select-none text-accent">•</span>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            )}
            <details className="mt-3 text-xs text-ink-500">
              <summary className="cursor-pointer select-none hover:text-accent">
                Show original passage ({result.kedit.source_text.length} chars)
              </summary>
              <p className="mt-2 rounded-md bg-ink-50 p-2 leading-relaxed text-ink-700">
                {result.kedit.source_text}
              </p>
            </details>
          </div>
        </section>
      )}

      {/* Stage 6: FLARE */}
      {result.flare && (
        <section>
          <StageHeader
            stage="Stage 6"
            title="FLARE · grounding check"
            ms={t.flare_check}
          />
          <div className="card p-4">
            <div className="flex items-baseline justify-between">
              <div className="flex items-baseline gap-2">
                {result.flare.grounded ? (
                  <span className="pill border-emerald-300 bg-emerald-50 text-emerald-700">
                    grounded
                  </span>
                ) : (
                  <span className="pill border-amber-300 bg-amber-50 text-amber-700">
                    unsupported claims found
                  </span>
                )}
                {result.flare.triggered && (
                  <span className="pill-accent">re-retrieval triggered</span>
                )}
                {result.flare.refined && (
                  <span className="pill-accent">answer refined</span>
                )}
                {result.flare.fallback && (
                  <span className="pill border-amber-300 bg-amber-50 text-amber-700">
                    fallback (parse failed)
                  </span>
                )}
              </div>
            </div>

            {result.flare.unsupported_claims.length > 0 && (
              <div className="mt-3">
                <p className="text-[11px] uppercase tracking-wider text-ink-500">
                  Unsupported claims
                </p>
                <ul className="mt-1.5 space-y-1 text-sm text-ink-700">
                  {result.flare.unsupported_claims.map((c, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="select-none text-amber-600">!</span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.flare.triggered && result.flare.retrieval_query && (
              <p className="mt-3 text-xs text-ink-500">
                <span className="text-ink-400">re-retrieval query: </span>
                <code className="rounded bg-ink-50 px-1.5 py-0.5 font-mono text-ink-700">
                  {result.flare.retrieval_query}
                </code>
              </p>
            )}

            {result.flare.re_retrieved.length > 0 && (
              <div className="mt-3">
                <p className="text-[11px] uppercase tracking-wider text-ink-500">
                  Re-retrieved (merged into context)
                </p>
                <ul className="mt-1.5 space-y-1 text-sm text-ink-700">
                  {result.flare.re_retrieved.map((h) => (
                    <li key={h.rank} className="flex items-baseline gap-2">
                      <span className="font-mono text-xs text-ink-400">[{h.rank}]</span>
                      <span className="font-medium">{h.title}</span>
                      <span className="font-mono text-[11px] text-ink-400">
                        rerank {h.rerank_score.toFixed(3)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.flare.refined && (
              <details className="mt-3 text-xs text-ink-500">
                <summary className="cursor-pointer select-none hover:text-accent">
                  Show draft (pre-refinement) answer
                </summary>
                <p className="mt-2 rounded-md bg-ink-50 p-2 leading-relaxed text-ink-700">
                  {result.flare.draft_answer}
                </p>
              </details>
            )}
          </div>
        </section>
      )}

      {/* Stage 7: Faithfulness gate (Q² + BEGIN) */}
      {result.faithfulness && (
        <section>
          <StageHeader
            stage="Stage 7"
            title="Q² + BEGIN · faithfulness verification"
            ms={t.faithfulness}
          />
          <div className="card p-4">
            <div className="flex flex-wrap items-baseline gap-2">
              <span
                className={
                  "pill " +
                  (result.faithfulness.q2_score >= result.faithfulness.q2_threshold
                    ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                    : "border-amber-300 bg-amber-50 text-amber-700")
                }
              >
                Q² {result.faithfulness.q2_score.toFixed(2)} ·
                <span className="ml-1 font-mono text-[10px] text-ink-400">
                  τ={result.faithfulness.q2_threshold.toFixed(2)}
                </span>
              </span>
              <span
                className={
                  "pill " +
                  (result.faithfulness.begin_label === "Fully Attributable"
                    ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                    : result.faithfulness.begin_label === "Not Attributable"
                      ? "border-red-300 bg-red-50 text-red-700"
                      : "border-ink-200 bg-ink-50 text-ink-600")
                }
              >
                BEGIN: {result.faithfulness.begin_label}
              </span>
              {result.faithfulness.gate_failed ? (
                <span className="pill border-red-300 bg-red-50 text-red-700">
                  gate failed
                </span>
              ) : (
                <span className="pill border-emerald-300 bg-emerald-50 text-emerald-700">
                  gate passed
                </span>
              )}
              {result.faithfulness.regenerated && (
                <span className="pill-accent">
                  regenerated with runner-up: {result.faithfulness.runner_up_title}
                </span>
              )}
              {(result.faithfulness.q2_fallback || result.faithfulness.begin_fallback) && (
                <span className="pill border-amber-300 bg-amber-50 text-amber-700">
                  fallback (parse failed)
                </span>
              )}
            </div>

            <p className="mt-2 text-sm leading-relaxed text-ink-600">
              <span className="text-ink-400">BEGIN rationale: </span>
              {result.faithfulness.begin_rationale || "(none)"}
            </p>

            {result.faithfulness.qa_pairs.length > 0 && (
              <details className="mt-3">
                <summary className="cursor-pointer select-none text-xs text-ink-500 hover:text-accent">
                  Show Q² question/answer breakdown ({result.faithfulness.qa_pairs.length} pair
                  {result.faithfulness.qa_pairs.length === 1 ? "" : "s"})
                </summary>
                <ul className="mt-2 space-y-2 text-sm">
                  {result.faithfulness.qa_pairs.map((p, i) => (
                    <li key={i} className="rounded-md bg-ink-50 p-2">
                      <div className="flex items-baseline gap-2">
                        <span
                          className={
                            "select-none font-mono text-xs " +
                            (p.match ? "text-emerald-600" : "text-red-600")
                          }
                        >
                          {p.match ? "✓" : "✗"}
                        </span>
                        <span className="font-medium text-ink-800">{p.question}</span>
                      </div>
                      <div className="mt-1.5 ml-5 space-y-0.5 text-xs text-ink-600">
                        <p>
                          <span className="text-ink-400">response: </span>
                          {p.response_answer}
                        </p>
                        <p>
                          <span className="text-ink-400">knowledge: </span>
                          {p.knowledge_answer || (
                            <span className="italic text-ink-400">(not addressed)</span>
                          )}
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>
              </details>
            )}

            {result.faithfulness.regenerated && result.faithfulness.pre_gate_answer && (
              <details className="mt-3 text-xs text-ink-500">
                <summary className="cursor-pointer select-none hover:text-accent">
                  Show pre-gate answer (before regeneration)
                </summary>
                <p className="mt-2 rounded-md bg-ink-50 p-2 leading-relaxed text-ink-700">
                  {result.faithfulness.pre_gate_answer}
                </p>
              </details>
            )}
          </div>
        </section>
      )}

      {/* Stage 8: answer */}
      <section>
        <StageHeader stage="Stage 8" title="Grounded answer · gpt-oss-120b" ms={t.generate} />
        <div className="card border-accent-ring bg-accent-soft/40 p-4">
          <p className="whitespace-pre-wrap text-base leading-relaxed text-ink-900">
            {result.answer}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-500">
            <span>grounded in</span>
            {result.used_titles.map((t) => (
              <span key={t} className="pill bg-white">
                {t}
              </span>
            ))}
          </div>
        </div>
        <button
          onClick={() => setShowPrompt((v) => !v)}
          className="mt-2 text-xs text-ink-500 underline-offset-2 hover:text-accent hover:underline"
        >
          {showPrompt ? "Hide" : "Show"} full prompt sent to the LLM
        </button>
        {showPrompt && (
          <pre className="mt-2 max-h-80 overflow-auto rounded-lg border border-ink-200 bg-ink-900 p-3 font-mono text-[11px] leading-relaxed text-ink-100">
            {result.prompt}
          </pre>
        )}
      </section>
    </div>
  );
}
