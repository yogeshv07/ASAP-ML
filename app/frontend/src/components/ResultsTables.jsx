import { DataTable } from "./DataTable";

export function ResultsTables({ topRecommended = [], recommendations = [], antibiogram = [], antibiogramOnly = false }) {
  if (antibiogramOnly) {
    return <DataTable rows={antibiogram} />;
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Top Recommendations</h3>
          <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Ranked shortlist</span>
        </div>
        <DataTable rows={topRecommended} compact />
      </div>

      <div>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Recommendation Matrix</h3>
          <span className="text-xs uppercase tracking-[0.2em] text-slate-400">All antibiotics</span>
        </div>
        <DataTable rows={recommendations} />
      </div>
    </div>
  );
}
