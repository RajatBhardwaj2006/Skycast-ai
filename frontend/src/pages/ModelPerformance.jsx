import { useEffect, useState } from "react";
import TopBar from "../components/TopBar.jsx";
import { getMetrics, getModelInfo, getValidation } from "../services/api";
import { formatInr, formatNumber } from "../utils/format";

export default function ModelPerformance() {
  const [metrics, setMetrics] = useState(null);
  const [meta, setMeta] = useState(null);
  const [val, setVal] = useState(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {});
    getModelInfo().then(setMeta).catch(() => {});
    getValidation().then(setVal).catch(() => {});
  }, []);

  const candidates = metrics?.comparison || [];
  const routeMetrics = val?.route_group_audit?.metrics;
  const airportMetrics = val?.airport_group_audit?.metrics;
  const strategies = val?.strategy_comparison || [];

  return (
    <>
      <TopBar title="Model Performance" description="Comprehensive holdout validation and unseen-route generalization audits." />
      
      {/* Top Metric Cards */}
      <div className="grid grid-4">
        <article className="card stat-card">
          <div className="label">Final Model</div>
          <div className="value" style={{ fontSize: "1.1rem" }}>
            {metrics?.best_model || "Random Forest (tuned)"}
          </div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>Hyperparameter-tuned</span>
        </article>
        <article className="card stat-card">
          <div className="label">In-Distribution MAE</div>
          <div className="value">{formatInr(metrics?.mae)}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>R²: {metrics?.r2?.toFixed(4)} · RMSE: {formatInr(metrics?.rmse)}</span>
        </article>
        <article className="card stat-card">
          <div className="label">Route-Group Audit MAE</div>
          <div className="value">{routeMetrics ? formatInr(routeMetrics.mae) : "—"}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>R²: {routeMetrics?.r2?.toFixed(4)} (Unseen routes)</span>
        </article>
        <article className="card stat-card">
          <div className="label">Airport-Group Audit MAE</div>
          <div className="value">{airportMetrics ? formatInr(airportMetrics.mae) : "—"}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>R²: {airportMetrics?.r2?.toFixed(4)} (Unseen airports)</span>
        </article>
      </div>

      {/* Validation Strategy Breakdown */}
      <div className="grid grid-2" style={{ marginTop: "1rem" }}>
        <section className="card">
          <h2>Validation Strategy Comparison</h2>
          <p className="muted" style={{ marginBottom: "0.8rem" }}>
            Why route-group and airport-group splits matter more for unseen Indian routes than random-row splits.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>Evaluation Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Random Holdout</strong></td>
                <td>{formatInr(metrics?.mae)}</td>
                <td>{formatInr(metrics?.rmse)}</td>
                <td>{metrics?.r2?.toFixed(4)}</td>
                <td>In-distribution performance on historical routes</td>
              </tr>
              <tr>
                <td><strong>Route-Group Holdout</strong></td>
                <td>{routeMetrics ? formatInr(routeMetrics.mae) : "—"}</td>
                <td>{routeMetrics ? formatInr(routeMetrics.rmse) : "—"}</td>
                <td>{routeMetrics?.r2?.toFixed(4)}</td>
                <td>Generalization to routes never seen during training</td>
              </tr>
              <tr>
                <td><strong>Airport-Group Holdout</strong></td>
                <td>{airportMetrics ? formatInr(airportMetrics.mae) : "—"}</td>
                <td>{airportMetrics ? formatInr(airportMetrics.rmse) : "—"}</td>
                <td>{airportMetrics?.r2?.toFixed(4)}</td>
                <td>Generalization to origin airports completely held out</td>
              </tr>
            </tbody>
          </table>
          {val?.route_group_audit?.held_out_routes && (
            <p className="muted" style={{ marginTop: "0.8rem", fontSize: "0.82rem" }}>
              <strong>Held-out test routes:</strong> {val.route_group_audit.held_out_routes.join(", ")}
            </p>
          )}
        </section>

        <section className="card">
          <h2>Dataset Expansion Comparison (Prompt 1.4)</h2>
          <p className="muted" style={{ marginBottom: "0.8rem" }}>
            Comparison of models trained on isolated sources vs. unified merged dataset.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Dataset Strategy</th>
                <th>Rows</th>
                <th>Holdout MAE</th>
                <th>Holdout R²</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((st) => (
                <tr key={st.strategy}>
                  <td><strong>{st.strategy}</strong></td>
                  <td>{st.rows?.toLocaleString("en-IN")}</td>
                  <td>{formatInr(st.mae)}</td>
                  <td>{st.r2?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ marginTop: "0.8rem", fontSize: "0.85rem" }}>
            Model C (Unified Merged Dataset) integrates all compatible records without fabricated features, achieving superior MAE and broader route coverage.
          </p>
        </section>
      </div>

      {/* Candidate Models Comparison Table */}
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Candidate Model Comparison</h2>
        <p className="muted" style={{ marginBottom: "0.8rem" }}>
          Evaluated using MAE, RMSE, R², training time, and inference latency per 1,000 predictions.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE (₹)</th>
                <th>RMSE (₹)</th>
                <th>R² Score</th>
                <th>Training Time</th>
                <th>Inference (ms / 1k)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((cand) => {
                const isWinner = cand.model === metrics?.best_model;
                return (
                  <tr key={cand.model} style={isWinner ? { background: "rgba(16, 185, 129, 0.08)" } : {}}>
                    <td>
                      <strong>{cand.model}</strong> {isWinner && <span style={{ color: "#10b981", marginLeft: "0.4rem" }}>★ Selected</span>}
                    </td>
                    <td>{formatInr(cand.mae)}</td>
                    <td>{formatInr(cand.rmse)}</td>
                    <td>{cand.r2?.toFixed(4)}</td>
                    <td>{cand.training_seconds != null ? `${cand.training_seconds}s` : "—"}</td>
                    <td>{cand.inference_ms_per_1k != null ? `${cand.inference_ms_per_1k}ms` : "—"}</td>
                    <td>{isWinner ? "Deployed Pipeline" : "Benchmark"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: "0.8rem" }}>
          Total dataset size: {meta?.training_rows?.toLocaleString("en-IN")} train rows · {meta?.test_rows?.toLocaleString("en-IN")} test rows · {meta?.features?.length} engineered features.
        </p>
      </section>
    </>
  );
}
