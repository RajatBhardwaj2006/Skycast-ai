import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import TopBar from "../components/TopBar.jsx";
import { getFeatureImportance, getGeoExperiment, getMetrics } from "../services/api";
import { formatInr } from "../utils/format";

export default function Insights() {
  const [metrics, setMetrics] = useState(null);
  const [importance, setImportance] = useState([]);
  const [geo, setGeo] = useState(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {});
    getFeatureImportance()
      .then((d) => setImportance(d.features || []))
      .catch(() => {});
    getGeoExperiment().then(setGeo).catch(() => {});
  }, []);

  const comparison = metrics?.comparison || [];

  return (
    <>
      <TopBar title="Insights" description="Model comparison, geographic-feature experiment, and learned importances." />
      <section className="card">
        <h2>Model comparison</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>R²</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map((row) => (
              <tr key={row.model}>
                <td>{row.model}</td>
                <td>{formatInr(row.mae)}</td>
                <td>{formatInr(row.rmse)}</td>
                <td>{row.r2?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <div className="grid grid-2" style={{ marginTop: "1rem" }}>
        <section className="card">
          <h2>MAE</h2>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={comparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" hide />
                <YAxis />
                <Tooltip />
                <Bar dataKey="mae" fill="#0f6f73" name="MAE" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="card">
          <h2>R²</h2>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={comparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" hide />
                <YAxis domain={[0.8, 1]} />
                <Tooltip />
                <Bar dataKey="r2" fill="#c9893a" name="R²" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Geographic feature experiment</h2>
        <p className="muted">Same estimator (HistGradientBoosting) to isolate distance and coordinates. Winner: {geo?.winner || "—"}</p>
        <table className="table">
          <thead>
            <tr>
              <th>Variant</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>R²</th>
            </tr>
          </thead>
          <tbody>
            {(geo?.results || []).map((row) => (
              <tr key={row.model}>
                <td>{row.model}</td>
                <td>{formatInr(row.mae)}</td>
                <td>{formatInr(row.rmse)}</td>
                <td>{row.r2?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Feature importance</h2>
        <div style={{ height: 320 }}>
          <ResponsiveContainer>
            <BarChart data={importance.slice(0, 10)} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" />
              <Tooltip />
              <Legend />
              <Bar dataKey="importance" fill="#0f6f73" name="Importance" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </>
  );
}
