function MetricCard({ title, value, tone, subtitle }) {
  const toneMap = {
    green: "from-emerald-400/18 to-teal-500/8 border-emerald-400/20",
    yellow: "from-amber-400/18 to-orange-500/8 border-amber-400/20",
    red: "from-rose-400/18 to-red-500/8 border-rose-400/20"
  };

  return (
    <div
      className={`glass-panel bg-gradient-to-br ${toneMap[tone]} p-5 transition duration-300 hover:-translate-y-1 hover:shadow-glass`}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-300">{title}</p>
      <div className="mt-4 text-2xl font-semibold text-white">{value}</div>
      <p className="mt-2 text-sm text-slate-300">{subtitle}</p>
    </div>
  );
}

export function MetricCards({ metrics }) {
  const riskTone =
    metrics.highestRisk === "High" ? "red" : metrics.highestRisk === "Moderate" ? "yellow" : "green";
  const mdrTone = metrics.mdrStatus === "Warning" ? "red" : "green";

  return (
    <section className="grid gap-4 md:grid-cols-3">
      <MetricCard
        title="Top Recommendation"
        value={metrics.topRecommendation}
        tone="green"
        subtitle="Lowest estimated resistance profile"
      />
      <MetricCard
        title="Highest Risk Band"
        value={metrics.highestRisk}
        tone={riskTone}
        subtitle="Most elevated band across predicted antibiotics"
      />
      <MetricCard
        title="MDR Status"
        value={metrics.mdrStatus}
        tone={mdrTone}
        subtitle="Multi-drug resistance summary for current sample"
      />
    </section>
  );
}
