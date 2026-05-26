import { useState } from "react";
import type { Turn } from "../api";

type Props = {
  history: Turn[];
  onChange: (h: Turn[]) => void;
};

export function HistoryEditor({ history, onChange }: Props) {
  const [open, setOpen] = useState(history.length > 0);
  const empty = history.length === 0;

  const update = (i: number, patch: Partial<Turn>) =>
    onChange(history.map((t, j) => (i === j ? { ...t, ...patch } : t)));
  const remove = (i: number) => onChange(history.filter((_, j) => j !== i));
  const add = (role: Turn["role"]) =>
    onChange([...history, { role, text: "" }]);

  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left transition hover:bg-ink-50"
      >
        <div className="flex items-center gap-2">
          <span className="stage-h">Dialogue history</span>
          <span className="text-xs text-ink-400">
            {empty ? "(none)" : `${history.length} turn${history.length === 1 ? "" : "s"}`}
          </span>
        </div>
        <span className="text-ink-400">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="border-t border-ink-100 p-3">
          {history.length > 0 && (
            <div className="space-y-2">
              {history.map((t, i) => (
                <div key={i} className="flex items-center gap-2">
                  <select
                    value={t.role}
                    onChange={(e) => update(i, { role: e.target.value as Turn["role"] })}
                    className="rounded-md border border-ink-200 bg-white px-2 py-1.5 text-xs text-ink-700 focus:border-accent focus:outline-none"
                  >
                    <option value="user">user</option>
                    <option value="bot">bot</option>
                  </select>
                  <input
                    value={t.text}
                    onChange={(e) => update(i, { text: e.target.value })}
                    placeholder={t.role === "user" ? "What the user said..." : "What the assistant said..."}
                    className="flex-1 rounded-md border border-ink-200 bg-white px-2.5 py-1.5 text-sm text-ink-800 placeholder:text-ink-400 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-soft"
                  />
                  <button
                    onClick={() => remove(i)}
                    className="rounded-md px-2 py-1 text-sm text-ink-400 transition hover:bg-ink-50 hover:text-red-600"
                    title="Remove turn"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="mt-2 flex gap-2">
            <button
              onClick={() => add("user")}
              className="rounded-md border border-ink-200 bg-white px-2.5 py-1 text-xs font-medium text-ink-700 transition hover:bg-ink-50"
            >
              + user turn
            </button>
            <button
              onClick={() => add("bot")}
              className="rounded-md border border-ink-200 bg-white px-2.5 py-1 text-xs font-medium text-ink-700 transition hover:bg-ink-50"
            >
              + bot turn
            </button>
            {history.length > 0 && (
              <button
                onClick={() => onChange([])}
                className="ml-auto rounded-md px-2.5 py-1 text-xs text-ink-500 transition hover:text-red-600"
              >
                clear
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
