import { SequenceTextarea } from "./SequenceTextarea";

export function SidebarPanel({
  sequence,
  amrIdentifier,
  isBusy,
  onSequenceChange,
  onAmrIdentifierChange,
  onRunPrediction,
  onLoadSample
}) {
  return (
    <aside className="glass-panel flex h-full min-h-[760px] flex-col p-5 sm:p-6">
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-teal-300/80">Input Panel</p>
        <h2 className="mt-2 text-2xl font-semibold text-white">Prediction Workspace</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          Submit a DNA sequence and AMR identifier to generate a probability-ranked antibiogram.
        </p>
      </div>

      <div className="space-y-5">
        <div>
          <label className="mb-3 block text-sm font-medium text-slate-100">DNA sequence</label>
          <SequenceTextarea value={sequence} onChange={onSequenceChange} />
        </div>

        <div>
          <label className="mb-3 block text-sm font-medium text-slate-100">AMR identifier</label>
          <input
            value={amrIdentifier}
            onChange={(event) => onAmrIdentifierChange(event.target.value)}
            placeholder="bla_7"
            className="w-full rounded-2xl border border-slate-700/70 bg-slate-950/80 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-teal-400/60 focus:ring-2 focus:ring-teal-400/20"
          />
        </div>
      </div>

      <div className="mt-6 flex flex-col gap-3 sm:flex-row xl:flex-col">
        <button
          onClick={onRunPrediction}
          disabled={isBusy}
          className="rounded-2xl bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition duration-300 hover:scale-[1.02] hover:shadow-[0_20px_45px_rgba(20,184,166,0.25)] disabled:cursor-wait disabled:opacity-70"
        >
          {isBusy ? "Running Prediction" : "Run Prediction"}
        </button>

        <button
          onClick={onLoadSample}
          disabled={isBusy}
          className="rounded-2xl border border-slate-600/80 bg-white/[0.02] px-5 py-3 text-sm font-semibold text-slate-200 transition duration-300 hover:scale-[1.02] hover:border-teal-400/40 hover:bg-teal-400/5 disabled:cursor-wait disabled:opacity-70"
        >
          Load Sample
        </button>
      </div>

      <div className="mt-auto rounded-2xl border border-indigo-400/20 bg-indigo-500/5 p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-indigo-200/90">Sequence Notes</p>
        <ul className="mt-3 space-y-2 text-sm text-slate-300">
          <li>Use a clean biological sequence without separators.</li>
          <li>Monospace highlighting distinguishes A, T, G, and C visually.</li>
          <li>Example AMR identifiers: `bla_7`, `tet_3`, `erm_2`.</li>
        </ul>
      </div>
    </aside>
  );
}
