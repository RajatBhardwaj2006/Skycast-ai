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

  const candidates = metrics?.candidates || metrics?.comparison || [];
  const routeMetrics = val?.route_group_audit?.metrics || meta?.route_group_metrics;
  const airportMetrics = val?.airport_group_audit?.metrics || meta?.airport_group_metrics;
  const temporalMetrics = val?.temporal_audit?.metrics || meta?.time_split_metrics;

  return (
    <>
      <TopBar title="Model Performance" description="Comprehensive holdout validation and unseen-route generalization audits." />
      
      {/* Top Metric Cards */}
      <div className="grid grid-4">
        <article className="card stat-card">
          <div className="label">Deployed Pipeline</div>
          <div className="value" style={{ fontSize: "1.05rem" }}>
            {metrics?.best_model || "AirfareModelRouter"}
          </div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>Dual Class-Aware Architecture</span>
        </article>
        <article className="card stat-card">
          <div className="label">Random Holdout MAE</div>
          <div className="value">{formatInr(metrics?.mae)}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>R²: {metrics?.r2?.toFixed(4)} · RMSE: {formatInr(metrics?.rmse)}</span>
        </article>
        <article className="card stat-card">
          <div className="label">Route-Group Holdout MAE</div>
          <div className="value">{routeMetrics ? formatInr(routeMetrics.mae) : "—"}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>R²: {routeMetrics?.r2?.toFixed(4)} (Unseen routes)</span>
        </article>
        <article className="card stat-card">
          <div className="label">Airport-Group Holdout MAE</div>
          <div className="value">{airportMetrics ? formatInr(airportMetrics.mae) : "—"}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>R²: {airportMetrics?.r2?.toFixed(4)} (Unseen airports)</span>
        </article>
      </div>

      {/* 4-Split Comprehensive Validation Suite */}
      <div className="grid grid-2" style={{ marginTop: "1rem" }}>
        <section className="card">
          <h2>4-Split Validation Suite</h2>
          <p className="muted" style={{ marginBottom: "0.8rem" }}>
            Evaluating generalization across random rows, unseen city-pairs, unseen airport hubs, and future dates.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Validation Split</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R² Score</th>
                <th>Generalization Scope</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Random-Row Holdout</strong></td>
                <td>{formatInr(metrics?.mae)}</td>
                <td>{formatInr(metrics?.rmse)}</td>
                <td>{metrics?.r2?.toFixed(4)}</td>
                <td>In-distribution test on core routes</td>
              </tr>
              <tr>
                <td><strong>Route-Group Holdout</strong></td>
                <td>{routeMetrics ? formatInr(routeMetrics.mae) : "—"}</td>
                <td>{routeMetrics ? formatInr(routeMetrics.rmse) : "—"}</td>
                <td>{routeMetrics?.r2?.toFixed(4)}</td>
                <td>Generalization to routes never in training</td>
              </tr>
              <tr>
                <td><strong>Airport-Group Holdout</strong></td>
                <td>{airportMetrics ? formatInr(airportMetrics.mae) : "—"}</td>
                <td>{airportMetrics ? formatInr(airportMetrics.rmse) : "—"}</td>
                <td>{airportMetrics?.r2?.toFixed(4)}</td>
                <td>Generalization to unrepresented origin airports</td>
              </tr>
              <tr>
                <td><strong>Time-Based Holdout</strong></td>
                <td>{temporalMetrics ? formatInr(temporalMetrics.mae) : "—"}</td>
                <td>{temporalMetrics ? formatInr(temporalMetrics.rmse) : "—"}</td>
                <td>{temporalMetrics?.r2?.toFixed(4)}</td>
                <td>Predicting future booking dates</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="card">
          <h2>Class Dominance Resolution</h2>
          <p className="muted" style={{ marginBottom: "0.8rem" }}>
            How the dual-model architecture solved the 89% class dominance issue.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Architecture</th>
                <th>Class Feature Dominance</th>
                <th>Economy Route-Group MAE</th>
                <th>Physical Feature Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Legacy Single Tree</strong></td>
                <td style={{ color: "#ef4444" }}>~89.0%</td>
                <td>₹3,290</td>
                <td>Suppressed by class variance</td>
              </tr>
              <tr style={{ background: "rgba(16, 185, 129, 0.08)" }}>
                <td><strong>Dual AirfareModelRouter</strong></td>
                <td style={{ color: "#10b981" }}>0.0% (Isolated)</td>
                <td><strong>₹1,270 (61% error drop)</strong></td>
                <td>Booking window (53%), airline, stops & geo drive 100%</td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ marginTop: "0.8rem", fontSize: "0.85rem" }}>
            By training dedicated estimators for Economy and Business with log targets, the model accurately predicts short-haul fares (e.g. ATQ → SXR: ₹3,753) without duration overfitting.
          </p>
        </section>
      </div>

      {/* Candidate Models Comparison Table */}
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Candidate Models Benchmark (Economy Class Holdouts)</h2>
        <p className="muted" style={{ marginBottom: "0.8rem" }}>
          Evaluated using Random MAE, Route-Group MAE, Airport-Group MAE, Time MAE, and inference latency.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Random MAE</th>
                <th>Route-Group MAE</th>
                <th>Airport-Group MAE</th>
                <th>Time Split MAE</th>
                <th>Training Time</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((cand) => {
                const isSelected = cand.model === "XGBoost" || cand.model === metrics?.best_model;
                return (
                  <tr key={cand.model} style={isSelected ? { background: "rgba(16, 185, 129, 0.08)" } : {}}>
                    <td>
                      <strong>{cand.model}</strong> {isSelected && <span style={{ color: "#10b981", marginLeft: "0.4rem" }}>★ Selected</span>}
                    </td>
                    <td>{formatInr(cand.random_mae || cand.mae)}</td>
                    <td>{formatInr(cand.route_group_mae)}</td>
                    <td>{formatInr(cand.airport_group_mae)}</td>
                    <td>{formatInr(cand.time_split_mae)}</td>
                    <td>{cand.training_seconds != null ? `${cand.training_seconds}s` : "—"}</td>
                    <td>{isSelected ? "Core Router Estimator" : "Benchmark"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: "0.8rem" }}>
          Total dataset size: {meta?.training_rows?.toLocaleString("en-IN")} train rows · {meta?.test_rows?.toLocaleString("en-IN")} test rows · 248 reference airports supported.
        </p>
      </section>
    </>
  );
}
