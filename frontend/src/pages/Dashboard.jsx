import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import TopBar from "../components/TopBar.jsx";
import { getMetrics, getModelInfo } from "../services/api";
import { formatInr } from "../utils/format";

export default function Dashboard({ history }) {
  const [metrics, setMetrics] = useState(null);
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {});
    getModelInfo().then(setMeta).catch(() => {});
  }, []);

  return (
    <>
      <TopBar title="Welcome to SkyCast" description="AI-powered airfare estimation from historical flight records." />
      <div className="grid grid-4">
        <article className="card stat-card">
          <div className="label">Training records</div>
          <div className="value">{meta?.training_rows?.toLocaleString("en-IN") || "300K+"}</div>
        </article>
        <article className="card stat-card">
          <div className="label">Airlines in data</div>
          <div className="value">{meta?.airlines?.length || "—"}</div>
        </article>
        <article className="card stat-card">
          <div className="label">Training cities</div>
          <div className="value">{meta?.training_cities?.length || "—"}</div>
        </article>
        <article className="card stat-card">
          <div className="label">Best model</div>
          <div className="value" style={{ fontSize: "1.2rem" }}>
            {metrics?.best_model || "—"}
          </div>
        </article>
      </div>
      <div className="grid grid-2" style={{ marginTop: "1rem" }}>
        <section className="card">
          <h2>Recent estimates</h2>
          <p className="muted">This session only — nothing is invented or stored on a server.</p>
          {!history.length && <p style={{ marginTop: "0.8rem" }}>No predictions yet. <Link to="/predict">Predict a fare</Link>.</p>}
          {history.slice(0, 6).map((item, idx) => (
            <div className="history-item" key={idx}>
              <div>
                <strong>
                  {item.source?.city} → {item.destination?.city}
                </strong>
                <div className="muted">{new Date(item.savedAt).toLocaleString()}</div>
              </div>
              <strong>{formatInr(item.predicted_price)}</strong>
            </div>
          ))}
        </section>
        <section className="card">
          <h2>How SkyCast works</h2>
          <p className="muted">
            Search any Indian city or airport in the location database, compute great-circle distance, then score the
            fare with the saved regression pipeline. The UI can search many airports; the model was trained on six
            metro city pairs plus geographic features.
          </p>
          <Link className="btn btn-primary" style={{ display: "inline-block", marginTop: "1rem" }} to="/predict">
            Predict Fare
          </Link>
        </section>
      </div>
    </>
  );
}
