import { useEffect, useRef, useState } from "react";
import {
  DEFAULT_SETTINGS,
  getInfo,
  resetDialogue,
  runPipeline,
  type Info,
  type RunResponse,
  type Settings,
  type Turn,
} from "./api";
import { Sidebar } from "./components/Sidebar";
import { Examples } from "./components/Examples";
import { HistoryEditor } from "./components/HistoryEditor";
import { Trace } from "./components/Trace";

export default function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [history, setHistory] = useState<Turn[]>([]);
  const [userTurn, setUserTurn] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dialogueId, setDialogueId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const traceRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getInfo()
      .then(setInfo)
      .catch((e) => setInfoError(String(e.message ?? e)));
  }, []);

  const run = async (turnOverride?: string, historyOverride?: Turn[]) => {
    const turn = (turnOverride ?? userTurn).trim();
    const hist = historyOverride ?? history;
    if (!turn) {
      inputRef.current?.focus();
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const r = await runPipeline(turn, hist, settings, dialogueId);
      setResult(r);
      setDialogueId(r.dialogue_id);
      setTimeout(() => traceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  const newConversation = async () => {
    try {
      const r = await resetDialogue(dialogueId);
      setDialogueId(r.dialogue_id);
    } catch {
      // best-effort; clear local state regardless
      setDialogueId(null);
    }
    setHistory([]);
    setUserTurn("");
    setResult(null);
    setError(null);
    inputRef.current?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) {
      e.preventDefault();
      run();
    }
  };

  return (
    <div className="min-h-screen lg:flex">
      <div className="lg:sticky lg:top-0 lg:h-screen lg:w-80 lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-ink-200 lg:bg-white">
        <Sidebar info={info} infoError={infoError} settings={settings} onChange={setSettings} />
      </div>

      <main className="mx-auto w-full max-w-3xl flex-1 px-6 py-10 lg:py-12">
        <header className="mb-8 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-ink-900">
              RAG Pipeline Tester
            </h2>
            <p className="mt-1 text-sm text-ink-500">
              Type a question or pick an example, then watch each pipeline stage produce its output.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {dialogueId && (
              <span
                className="font-mono text-[11px] text-ink-400"
                title="Per-dialogue passage cache key"
              >
                dialogue · {dialogueId.slice(0, 8)}
              </span>
            )}
            <button
              onClick={newConversation}
              className="rounded-md border border-ink-200 bg-white px-3 py-1.5 text-xs font-medium text-ink-700 transition hover:border-accent-ring hover:text-accent"
              title="Clear history, reset the per-dialogue cache, start fresh"
            >
              New conversation
            </button>
          </div>
        </header>

        <div className="space-y-6">
          <Examples
            onPick={(ex) => {
              setHistory(ex.history);
              setUserTurn(ex.turn);
              run(ex.turn, ex.history);
            }}
          />

          <HistoryEditor history={history} onChange={setHistory} />

          <div className="card flex items-center gap-2 p-2 pl-3 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent-soft">
            <span className="select-none font-mono text-sm text-ink-400">›</span>
            <input
              ref={inputRef}
              value={userTurn}
              onChange={(e) => setUserTurn(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask anything grounded in the loaded passages..."
              className="flex-1 bg-transparent py-2 text-base text-ink-800 placeholder:text-ink-400 focus:outline-none"
            />
            <button
              onClick={() => run()}
              disabled={running || !userTurn.trim()}
              className="flex items-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-ink-300"
            >
              {running ? (
                <>
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                  Running
                </>
              ) : (
                "Run"
              )}
            </button>
          </div>

          {error && (
            <div className="card border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        <div ref={traceRef} className="mt-10">
          {result && <Trace result={result} />}
          {!result && !running && (
            <div className="mt-8 rounded-xl border border-dashed border-ink-200 px-6 py-10 text-center text-sm text-ink-400">
              Pipeline output will appear here.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
