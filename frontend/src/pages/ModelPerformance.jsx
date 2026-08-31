import { useEffect, useState } from "react";
import TopBar from "../components/TopBar.jsx";
import { getMetrics, getModelInfo } from "../services/api";
import { formatInr, formatNumber } from "../utils/format";

export default function ModelPerformance() {
  const [metrics, setMetrics] = useState(null);
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {});
    getModelInfo().then(setMeta).catch(() => {});
  }, []);

  return (
    <>
      <TopBar title="Model Performance" description="Hold-out test metrics from the saved training run." />
      <div className="grid grid-4">
        <article className="card stat-card">
          <div className="label">Best model</div>
          <div className="value" style={{ fontSize: "1.15rem" }}>
            {metrics?.best_model}
          </div>
        </article>
        <article className="card stat-card">
          <div className="label">MAE</div>
          <div className="value">{formatInr(metrics?.mae)}</div>
        </article>
        <article className="card stat-card">
          <div className="label">RMSE</div>
          <div className="value">{formatInr(metrics?.rmse)}</div>
        </article>
        <article className="card stat-card">
          <div className="label">R²</div>
          <div className="value">{metrics?.r2?.toFixed(4)}</div>
        </article>
      </div>
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>What the metrics mean</h2>
        <p>
          <strong>MAE:</strong> on average, predictions differ from historical fares by about {formatInr(metrics?.mae)}.
        </p>
        <p>
          <strong>RMSE:</strong> large misses are penalised more heavily ({formatInr(metrics?.rmse)}).
        </p>
        <p>
          <strong>R²:</strong> share of airfare variation explained on the test set ({formatNumber(metrics?.r2, 3)}).
        </p>
        <p>
          <strong>MSE:</strong> {metrics?.mse ? Math.round(metrics.mse).toLocaleString("en-IN") : "—"}.
        </p>
        <p className="muted" style={{ marginTop: "0.8rem" }}>
          Training samples: {meta?.training_rows?.toLocaleString("en-IN")} · Test samples:{" "}
          {meta?.test_rows?.toLocaleString("en-IN")} · Features: {meta?.features?.length}
        </p>
      </section>
    </>
  );
}
