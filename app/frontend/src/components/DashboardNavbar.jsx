function StatusPill({ label, state, tone }) {
  const toneClasses = {
    success: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
    warning: "border-amber-400/30 bg-amber-400/10 text-amber-200",
    idle: "border-slate-400/25 bg-slate-400/10 text-slate-200"
  };

  return (
    <div className={`rounded-full border px-3 py-2 ${toneClasses[tone]}`}>
      <div className="text-[10px] font-semibold uppercase tracking-[0.24em] opacity-80">{label}</div>
      <div className="mt-1 text-sm font-medium">{state}</div>
    </div>
  );
}

export function DashboardNavbar({ datasetReady, modelsReady, isBusy }) {
  return (
    <header className="glass-panel flex flex-col gap-5 px-5 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.28em] text-teal-300/80">
          DNA Antibiogram Prediction Workspace
        </p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Scientific prediction dashboard
        </h1>
      </div>

      <div className="flex flex-wrap gap-3">
        <StatusPill label="Ready" state={datasetReady ? "Ready" : "Missing"} tone={datasetReady ? "success" : "warning"} />
        <StatusPill label="Trained" state={modelsReady ? "Trained" : "Pending"} tone={modelsReady ? "success" : "warning"} />
        <StatusPill label="Idle" state={isBusy ? "Running" : "Idle"} tone={isBusy ? "warning" : "idle"} />
      </div>
    </header>
  );
}
