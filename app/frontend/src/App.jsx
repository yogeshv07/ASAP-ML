import { useEffect, useMemo, useState } from "react";
import { DashboardNavbar } from "./components/DashboardNavbar";
import { SidebarPanel } from "./components/SidebarPanel";
import { MetricCards } from "./components/MetricCards";
import { AntibiogramChart } from "./components/AntibiogramChart";
import { StatusAlert } from "./components/StatusAlert";
import { ResultsTables } from "./components/ResultsTables";

const defaultSequence =
  "ATGCGTACGGGTTTAACCGTATGGATCGGTATATGCCGATACCGGTTATGCGTACGTTAGC";

const defaultAmr = "bla_7";

function App() {
  const [sequence, setSequence] = useState(defaultSequence);
  const [amrIdentifier, setAmrIdentifier] = useState(defaultAmr);
  const [prediction, setPrediction] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [isBusy, setIsBusy] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => {});
  }, []);

  const metrics = useMemo(() => {
    const topRecommendation = prediction?.top_recommended?.[0]?.antibiotic || "Awaiting analysis";
    const highestRisk = prediction?.antibiogram?.[0]?.level || "Pending";
    const mdrStatus = prediction ? (prediction.mdr_warning ? "Warning" : "Clear") : "Pending";
    return { topRecommendation, highestRisk, mdrStatus };
  }, [prediction]);

  const runPrediction = async () => {
    setError("");
    setIsBusy(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gene_sequence: sequence,
          amr_identifier: amrIdentifier
        })
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }
      setPrediction(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  };

  const loadSample = () => {
    setSequence(defaultSequence);
    setAmrIdentifier(defaultAmr);
    setError("");
  };

  return (
    <div className="min-h-screen px-4 py-4 sm:px-6 lg:px-8">
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] max-w-[1700px] flex-col gap-6">
        <DashboardNavbar
          datasetReady={Boolean(health?.dataset_exists)}
          modelsReady={Boolean(health?.models_available)}
          isBusy={isBusy}
        />

        {error ? (
          <div className="glass-panel border-red-400/25 px-5 py-4 text-sm text-red-200">{error}</div>
        ) : null}

        <div className="grid flex-1 gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <SidebarPanel
            sequence={sequence}
            amrIdentifier={amrIdentifier}
            isBusy={isBusy}
            onSequenceChange={setSequence}
            onAmrIdentifierChange={setAmrIdentifier}
            onRunPrediction={runPrediction}
            onLoadSample={loadSample}
          />

          <main className="flex min-h-0 flex-col gap-6">
            <MetricCards metrics={metrics} />

            <section className="grid min-h-0 gap-6 2xl:grid-cols-[minmax(0,1.2fr)_420px]">
              <div className="glass-panel soft-grid p-5 sm:p-6">
                <div className="mb-5 flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-teal-300/80">
                      Results
                    </p>
                    <h2 className="mt-2 text-2xl font-semibold text-white">
                      Antibiogram Probability Profile
                    </h2>
                    <p className="mt-2 max-w-2xl text-sm text-slate-300">
                      Probability estimates are mapped to risk bands with threshold overlays for moderate
                      and high resistance.
                    </p>
                  </div>
                </div>

                <AntibiogramChart data={prediction?.antibiogram || []} />
              </div>

              <div className="flex flex-col gap-6">
                <StatusAlert mdrWarning={prediction?.mdr_warning} />
                <ResultsTables
                  topRecommended={prediction?.top_recommended || []}
                  recommendations={prediction?.recommendations || []}
                />
              </div>
            </section>

            <section className="glass-panel p-5 sm:p-6">
              <div className="mb-4">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-indigo-300/80">
                  Detailed Output
                </p>
                <h3 className="mt-2 text-xl font-semibold text-white">Full Antibiogram Table</h3>
              </div>
              <ResultsTables antibiogramOnly antibiogram={prediction?.antibiogram || []} />
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
