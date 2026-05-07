export function DataTable({ rows, compact = false }) {
  if (!rows.length) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-700/70 bg-slate-950/35 px-4 py-6 text-sm text-slate-400">
        No data available yet.
      </div>
    );
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-800/80 bg-slate-950/55">
      <div className="max-h-[320px] overflow-auto">
        <table className="min-w-full divide-y divide-slate-800/80 text-left">
          <thead className="sticky top-0 z-10 bg-slate-950/95 backdrop-blur">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className={`px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400 ${
                    compact ? "" : "whitespace-nowrap"
                  }`}
                >
                  {column.replaceAll("_", " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {rows.map((row, index) => (
              <tr key={index} className="transition hover:bg-white/[0.03]">
                {columns.map((column) => (
                  <td key={column} className="px-4 py-3 text-sm text-slate-200">
                    {typeof row[column] === "number" ? row[column].toFixed(4) : String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
