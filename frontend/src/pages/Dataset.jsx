import { useEffect, useState } from "react";
import TopBar from "../components/TopBar.jsx";
import { getDatasetInfo } from "../services/api";
import { formatInr } from "../utils/format";

const DICTIONARY = [
  ["airline", "Categorical", "Operating airline name (e.g. Air India, Indigo, Vistara)", "Yes"],
  ["source_city / destination_city", "Categorical", "City names from the flight booking record", "Yes"],
  ["departure_time / arrival_time", "Categorical", "Time-of-day buckets (Early Morning, Morning, Afternoon, Evening, Night, Late Night)", "Yes"],
  ["stops", "Categorical", "Number of stops: zero, one, two_or_more", "Yes"],
  ["class", "Categorical", "Cabin class: Economy or Business (Unknown for new data without explicit class)", "Yes"],
  ["duration", "Numerical", "Flight travel time in decimal hours", "Yes"],
  ["days_left", "Numerical", "Booking window before departure (median imputed when unavailable)", "Yes"],
  ["distance_km", "Numerical", "Dynamic Haversine great-circle distance computed from airport coordinates", "Yes"],
  ["source/dest lat/lon", "Numerical", "WGS84 airport coordinates for geospatial generalization", "Yes"],
  ["price", "Numerical", "Target airfare in INR (₹1,105 to ₹123,071)", "Target"],
  ["seat_type, baggage, lounge, meal", "—", "Not present in historical datasets (unsupported)", "No"],
];

export default function Dataset() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    getDatasetInfo().then(setInfo).catch(() => {});
  }, []);

  const quality = info?.quality || {};
  const decision = info?.decision || {};
  const meta = info?.metadata || {};
  const numStats = quality?.numerical_statistics?.price || {};

  return (
    <>
      <TopBar title="Dataset & Data Expansion" description="Historical Indian aviation datasets, schema integration, and geographic catalog." />
      
      {/* Overview Stat Cards */}
      <div className="grid grid-4">
        <article className="card stat-card">
          <div className="label">Total Training Records</div>
          <div className="value">{quality.rows?.toLocaleString("en-IN") || "310,522"}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>Deduplicated clean rows</span>
        </article>
        <article className="card stat-card">
          <div className="label">Features Used</div>
          <div className="value">{meta.features?.length || 14}</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>Categorical + Geospatial</span>
        </article>
        <article className="card stat-card">
          <div className="label">Supported Airports</div>
          <div className="value">248</div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>All Indian states & UTs</span>
        </article>
        <article className="card stat-card">
          <div className="label">Price Range (INR)</div>
          <div className="value" style={{ fontSize: "1.1rem" }}>
            {formatInr(numStats.min || 1105)} – {formatInr(numStats.max || 123071)}
          </div>
          <span className="muted" style={{ fontSize: "0.8rem" }}>Median: {formatInr(numStats.median || 7466)}</span>
        </article>
      </div>

      {/* Data Sources Used */}
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Data Sources & Integration Strategy (Prompt 1.4)</h2>
        <table className="table" style={{ marginTop: "0.8rem" }}>
          <thead>
            <tr>
              <th>Source Dataset</th>
              <th>Records</th>
              <th>Date Range</th>
              <th>Integration Status</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Clean_Dataset.csv</strong></td>
              <td>300,153</td>
              <td>Feb – Mar 2022</td>
              <td><span style={{ color: "#10b981" }}>Merged (Primary)</span></td>
              <td>Contains tier-1 routes with verified cabin class & booking window (days_left).</td>
            </tr>
            <tr>
              <td><strong>Indian Flight Dataset</strong> (CSV)</td>
              <td>10,463 (deduped)</td>
              <td>Mar – Jun 2019</td>
              <td><span style={{ color: "#10b981" }}>Merged</span></td>
              <td>Adds additional routes (e.g. Kochi destination). Class retained as Unknown; days_left not fabricated.</td>
            </tr>
            <tr>
              <td><strong>Flight Data.xlsx</strong></td>
              <td>10,683</td>
              <td>Mar – Jun 2019</td>
              <td><span style={{ color: "#f59e0b" }}>Excluded</span></td>
              <td>Identified as an exact duplicate of the CSV source; excluded to prevent duplicate leakage.</td>
            </tr>
            <tr>
              <td><strong>Airports Reference Catalog</strong></td>
              <td>248 airports</td>
              <td>Current (WGS84)</td>
              <td><span style={{ color: "#10b981" }}>Active</span></td>
              <td>Provides IATA, city, state, coordinates, and aliases for dynamic Haversine calculation across India.</td>
            </tr>
          </tbody>
        </table>
        <p className="muted" style={{ marginTop: "0.8rem", fontSize: "0.85rem" }}>
          Modeling Decision: {decision.reason || "Integrated via standardized schema with zero fabricated features."}
        </p>
      </section>

      {/* Feature Dictionary */}
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Engineered Feature Dictionary</h2>
        <p className="muted" style={{ marginBottom: "0.8rem" }}>
          Full contract of inputs used by the deployed pipeline during training and inference.
        </p>
        <table className="table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Type</th>
              <th>Description</th>
              <th>Included in Model?</th>
            </tr>
          </thead>
          <tbody>
            {DICTIONARY.map((row) => (
              <tr key={row[0]}>
                {row.map((cell, idx) => (
                  <td key={cell}>
                    {idx === 3 && cell === "Yes" ? (
                      <span style={{ color: "#10b981", fontWeight: 600 }}>Yes</span>
                    ) : idx === 3 && cell === "Target" ? (
                      <span style={{ color: "#06b6d4", fontWeight: 600 }}>Target</span>
                    ) : idx === 3 && cell === "No" ? (
                      <span style={{ color: "#ef4444" }}>No</span>
                    ) : (
                      cell
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
