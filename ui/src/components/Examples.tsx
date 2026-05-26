import { EXAMPLES, type Example } from "../examples";

export function Examples({ onPick }: { onPick: (e: Example) => void }) {
  return (
    <div>
      <p className="stage-h mb-2">Try an example</p>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            onClick={() => onPick(ex)}
            className="group card flex h-full flex-col items-start gap-2 p-3 text-left transition hover:border-accent-ring hover:shadow-cardHover"
          >
            <div className="flex w-full items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wide text-accent">
                {ex.label}
              </span>
              <span className="text-ink-300 transition group-hover:text-accent">→</span>
            </div>
            <p className="text-xs text-ink-500">{ex.blurb}</p>
            <p className="text-sm leading-snug text-ink-800">{ex.turn}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
