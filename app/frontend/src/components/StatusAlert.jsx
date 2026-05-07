export function StatusAlert({ mdrWarning }) {
  const isPending = typeof mdrWarning === "undefined";

  return (
    <div
      className={`glass-panel p-5 transition duration-300 ${
        isPending
          ? "border-slate-700/60"
          : mdrWarning
            ? "border-red-400/30 bg-red-500/8 shadow-[0_0_0_1px_rgba(248,113,113,0.08),0_22px_44px_rgba(127,29,29,0.18)]"
            : "border-emerald-400/30 bg-emerald-500/8 shadow-[0_0_0_1px_rgba(52,211,153,0.08),0_22px_44px_rgba(6,95,70,0.18)]"
      }`}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-300">System Alert</p>
      <h3 className="mt-3 text-lg font-semibold text-white">
        {isPending
          ? "Prediction status pending"
          : mdrWarning
            ? "MDR warning detected"
            : "No MDR warning detected"}
      </h3>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        {isPending
          ? "A completed prediction will populate the current multi-drug resistance status."
          : mdrWarning
            ? "Three or more antibiotics entered the high-risk band. Review recommendation ranking carefully."
            : "The predicted profile did not trigger the multi-drug resistance threshold for this sequence."}
      </p>
    </div>
  );
}
