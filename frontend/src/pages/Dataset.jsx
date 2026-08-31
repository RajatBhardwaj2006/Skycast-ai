import { useEffect, useState } from "react";
import TopBar from "../components/TopBar.jsx";
import { getDatasetInfo } from "../services/api";

const DICTIONARY = [
  ["airline", "Categorical", "Operating airline", "Yes"],
  ["source_city / destination_city", "Categorical", "City names from the booking record (unknown labels ignored at inference)", "Yes"],
  ["departure_time / arrival_time", "Categorical", "Time-of-day buckets", "Yes"],
  ["stops", "Categorical", "zero / one / two_or_more", "Yes"],
  ["class", "Categorical", "Economy or Business", "Yes"],
  ["duration", "Numerical", "Block time in hours", "Yes"],
  ["days_left", "Numerical", "Booking window before departure", "Yes"],
  ["distance_km", "Numerical", "Haversine distance from airport coordinates", "Yes"],
  ["source/destination lat/lon", "Numerical", "Geographic coordinates for generalization", "Yes"],
  ["price", "Numerical", "Target airfare in INR", "Target"],
  ["seat_type, baggage, lounge, meal, aircraft", "—", "Not in the training files", "No"],
];

export default function Dataset() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    getDatasetInfo().then(setInfo).catch(() => {});
  }, []);

  const quality = info?.quality || {};
  const decision = info?.decision || {};
  const meta = info?.metadata || {};

  return (
    <>
      <TopBar title="Dataset" description="This application was trained using historical airfare data, not live airline feeds." />
      <div className="grid grid-4">
        <article className="card stat-card">
          <div className="label">Records</div>
          <div className="value">{quality.rows?.toLocaleString("en-IN") || "—"}</div>
        </article>
        <article className="card stat-card">
          <div className="label">Columns</div>
          <div className="value">{quality.columns ?? "—"}</div>
        </article>
        <article className="card stat-card">
          <div className="label">Airlines</div>
          <div className="value">{meta.airlines?.length ?? "—"}</div>
        </article>
        <article className="card stat-card">
          <div className="label">Source cities in training</div>
          <div className="value">{meta.training_cities?.length ?? "—"}</div>
        </article>
      </div>
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Modeling decision</h2>
        <p>{decision.reason}</p>
        <p className="muted" style={{ marginTop: "0.5rem" }}>
          Recommended file: {decision.recommended_modeling_dataset}. days_left was not invented from business/economy dates.
        </p>
      </section>
      <section className="card" style={{ marginTop: "1rem" }}>
        <h2>Feature dictionary</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Type</th>
              <th>Description</th>
              <th>Used?</th>
            </tr>
          </thead>
          <tbody>
            {DICTIONARY.map((row) => (
              <tr key={row[0]}>
                {row.map((cell) => (
                  <td key={cell}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
