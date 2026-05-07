import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

function getBarColor(probability) {
  if (probability > 0.8) return "#ef4444";
  if (probability >= 0.6) return "#f59e0b";
  return "#14b8a6";
}

export function AntibiogramChart({ data }) {
  if (!data.length) {
    return (
      <div className="flex h-[420px] items-center justify-center rounded-3xl border border-dashed border-slate-700/70 bg-slate-950/40 text-sm text-slate-400">
        Run a prediction to populate the antibiogram probability profile.
      </div>
    );
  }

  return (
    <div className="h-[420px] w-full rounded-3xl border border-slate-800/70 bg-slate-950/45 p-3 shadow-violet">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 16, right: 20, left: 0, bottom: 20 }}>
          <defs>
            <linearGradient id="safeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#34d399" stopOpacity={0.95} />
              <stop offset="100%" stopColor="#0f766e" stopOpacity={0.6} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis
            dataKey="antibiotic"
            stroke="#94a3b8"
            tickLine={false}
            axisLine={false}
            angle={-20}
            height={60}
            textAnchor="end"
          />
          <YAxis
            domain={[0, 1]}
            stroke="#94a3b8"
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value.toFixed(1)}
          />
          <Tooltip
            cursor={{ fill: "rgba(20,184,166,0.06)" }}
            contentStyle={{
              background: "rgba(15, 23, 42, 0.95)",
              border: "1px solid rgba(148, 163, 184, 0.2)",
              borderRadius: "16px",
              color: "#e2e8f0"
            }}
          />
          <ReferenceLine
            y={0.6}
            stroke="#f59e0b"
            strokeDasharray="6 6"
            strokeWidth={2}
            label={{ value: "0.6 threshold", fill: "#fbbf24", position: "insideTopRight" }}
          />
          <ReferenceLine
            y={0.8}
            stroke="#ef4444"
            strokeDasharray="6 6"
            strokeWidth={2}
            label={{ value: "0.8 threshold", fill: "#fca5a5", position: "insideTopLeft" }}
          />
          <Bar dataKey="probability" radius={[10, 10, 4, 4]} fill="url(#safeGradient)" maxBarSize={48}>
            {data.map((entry) => (
              <Cell key={entry.antibiotic} fill={getBarColor(entry.probability)} fillOpacity={0.92} />
            ))}
            <LabelList
              dataKey="probability"
              position="top"
              formatter={(value) => Number(value).toFixed(2)}
              fill="#cbd5e1"
              fontSize={12}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
