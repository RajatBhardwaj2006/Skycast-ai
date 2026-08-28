import { useEffect, useMemo, useState } from "react";
import LocationSearch from "./components/LocationSearch";
import { useSessionHistory } from "./hooks/useSessionHistory";
import { getCatalog, getDatasetInfo, getFeatureImportance, getGeoExperiment, getMetrics, getRouteDistance, predictFare } from "./services/api";
import { formatDuration, formatInr, formatKm, formatNumber, formatStops } from "./utils/format";

const PAGES = ["Dashboard", "Predict Fare", "Insights", "Model Performance", "Dataset", "History", "About"];
const timeBand = (value) => {
  const hour = Number(value?.split(":")[0] || 0);
  if (hour < 4) return "Late Night";
  if (hour < 7) return "Early Morning";
  if (hour < 12) return "Morning";
  if (hour < 16) return "Afternoon";
  if (hour < 20) return "Evening";
  return "Night";
};
const icons = { Dashboard: "⌂", "Predict Fare": "✈", Insights: "◈", "Model Performance": "▥", Dataset: "▤", History: "◷", About: "i" };

function Stat({ label, value }) { return <div className="card stat"><div className="label">{label}</div><div className="value">{value}</div></div>; }
function BarChart({ rows, label = "importance" }) {
  const max = Math.max(...rows.map((r) => Number(r[label]) || 0), 1);
  return <div className="chart-list">{rows.map((row) => <div className="chart-row" key={row.feature || row.model}><div className="chart-label">{row.feature || row.model}</div><div className="bar"><span style={{ width: `${(Number(row[label]) / max) * 100}%` }} /></div><div className="chart-value">{Number(row[label]).toFixed(label === "importance" ? 3 : 0)}</div></div>)}</div>;
}

export default function App() {
  const [page, setPage] = useState("Dashboard");
  const [menuOpen, setMenuOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [data, setData] = useState({});
  const [origin, setOrigin] = useState(null);
  const [destination, setDestination] = useState(null);
  const [route, setRoute] = useState(null);
  const [routeError, setRouteError] = useState("");
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [formError, setFormError] = useState("");
  const { items, add, clear } = useSessionHistory();
  const [form, setForm] = useState({ airline: "", class: "Economy", departure: "09:00", arrival: "11:15", stops: "zero", duration: 2.25, daysLeft: 20 });

  useEffect(() => {
    Promise.all([getCatalog(), getMetrics(), getFeatureImportance(), getGeoExperiment(), getDatasetInfo()])
      .then(([catalog, metrics, importance, geo, dataset]) => { setData({ catalog, metrics, importance, geo, dataset }); setForm((v) => ({ ...v, airline: catalog.airlines[0] || "", class: catalog.classes[0] || "Economy" })); })
      .catch(() => setError("SkyCast API is unavailable. Please start the backend."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    setRoute(null); setRouteError("");
    if (!origin || !destination) return;
    getRouteDistance(origin.iata, destination.iata).then(setRoute).catch((err) => setRouteError(err.message));
  }, [origin, destination]);

  const metrics = data.metrics || {};
  const catalog = data.catalog || {};
  const quality = data.dataset?.quality || {};
  const navigate = (next) => { setPage(next); setMenuOpen(false); };
  const update = (field, value) => setForm((current) => ({ ...current, [field]: value }));
  const swap = () => { setOrigin(destination); setDestination(origin); };

  async function submit(event) {
    event.preventDefault(); setFormError(""); setResult(null);
    if (!origin || !destination) { setFormError("Select a valid departure and destination city or airport."); return; }
    if (!route) { setFormError(routeError || "Route distance is not available yet."); return; }
    if (Number(form.duration) <= 0 || Number(form.daysLeft) < 0 || Number(form.daysLeft) > 365) { setFormError("Enter a duration above zero and a booking window from 0 to 365 days."); return; }
    setPredicting(true);
    try {
      const response = await predictFare({ airline: form.airline, source_iata: origin.iata, destination_iata: destination.iata, class: form.class, departure_time: timeBand(form.departure), arrival_time: timeBand(form.arrival), stops: form.stops, duration: Number(form.duration), days_left: Number(form.daysLeft) });
      setResult(response); add(response); setPage("Predict Fare");
    } catch (err) { setFormError(err.message || "Prediction could not be completed."); }
    finally { setPredicting(false); }
  }

  const featureRows = (data.importance?.features || []).slice(0, 8);
  const renderPredict = () => <div className="grid grid-2 predict-layout">
    <form className="card search-card" onSubmit={submit}>
      <h3>Where are you flying?</h3><p className="sub">Search any city or airport in the reference database. Distance is calculated from airport coordinates.</p>
      <div className="route-row"><LocationSearch label="From" value={origin} onChange={setOrigin} placeholder="Search city or airport" /><button className="swap-btn" type="button" onClick={swap} aria-label="Swap origin and destination">⇄</button><LocationSearch label="To" value={destination} onChange={setDestination} placeholder="Search city or airport" /></div>
      {route ? <div className="route-visual"><strong>{route.source.city} ({route.source.iata})</strong><div className="route-line"><span className="dot" /> {formatKm(route.distance_km)} <span>✈</span> <span className="dot" /></div><strong>{route.destination.city} ({route.destination.iata})</strong></div> : routeError ? <div className="error">{routeError}</div> : <p className="helper">Select both locations to calculate route distance.</p>}
      <div className="grid grid-2">
        <Field label="Airline"><select value={form.airline} onChange={(e) => update("airline", e.target.value)}>{(catalog.airlines || []).map((x) => <option key={x}>{x}</option>)}</select></Field>
        <Field label="Travel class"><select value={form.class} onChange={(e) => update("class", e.target.value)}>{(catalog.classes || []).map((x) => <option key={x}>{x}</option>)}</select></Field>
        <Field label="Departure time"><input type="time" value={form.departure} onChange={(e) => update("departure", e.target.value)} /></Field>
        <Field label="Arrival time"><input type="time" value={form.arrival} onChange={(e) => update("arrival", e.target.value)} /></Field>
        <Field label="Stops"><select value={form.stops} onChange={(e) => update("stops", e.target.value)}><option value="zero">Non-stop</option><option value="one">1 stop</option><option value="two_or_more">2+ stops</option></select></Field>
        <Field label="Duration (hours)"><input type="number" min="0.25" max="48" step="0.25" value={form.duration} onChange={(e) => update("duration", e.target.value)} /></Field>
      </div>
      <Field label="Booking window (days left)"><input type="range" min="0" max="365" value={form.daysLeft} onChange={(e) => update("daysLeft", e.target.value)} /><div className="helper">{form.daysLeft} days left</div></Field>
      <details className="prefs"><summary>Additional preferences <span className="coming-soon">Not used by current model</span></summary><div className="grid grid-2" style={{ marginTop: "0.8rem" }}><Field label="Seat type"><select disabled><option>Window</option><option>Middle</option><option>Aisle</option></select></Field><Field label="Luggage"><select disabled><option>Standard</option><option>15 kg</option><option>20 kg</option><option>30 kg</option></select></Field><Field label="Lounge access"><select disabled><option>No</option><option>Yes</option></select></Field><Field label="Aircraft type"><input disabled placeholder="Future data feature" /></Field></div></details>
      {formError && <div className="error">{formError}</div>}<button className="cta" disabled={predicting}>{predicting ? "Analyzing historical airfare patterns…" : "✈ Predict fare"}</button>
    </form>
    <PredictionCard result={result} metrics={metrics} />
  </div>;

  let content;
  if (loading) content = <div className="card">Loading SkyCast model information…</div>;
  else if (error) content = <div className="card error">{error}</div>;
  else if (page === "Predict Fare") content = renderPredict();
  else if (page === "Dashboard") content = <><div className="grid grid-4"><Stat label="Training records" value={formatNumber(data.dataset?.metadata?.training_rows, 0)} /><Stat label="Airlines" value={catalog.airlines?.length || 0} /><Stat label="Routes" value={data.dataset?.metadata?.route_count || "—"} /><Stat label="Best model" value={metrics.best_model || "—"} /></div><div style={{ marginTop: "1rem" }}>{renderPredict()}</div></>;
  else if (page === "Insights") content = <div className="grid grid-2"><section className="card"><h3>What influenced this estimate?</h3><p className="muted">Saved model feature importance, grouped by input feature.</p><BarChart rows={featureRows} /></section><section className="card"><h3>Geographic feature experiment</h3><p className="muted">{data.geo?.note}</p><DataTable rows={data.geo?.results || []} columns={["model", "mae", "rmse", "r2"]} /></section></div>;
  else if (page === "Model Performance") content = <><div className="grid grid-3"><Stat label="Held-out MAE" value={formatInr(metrics.mae)} /><Stat label="Held-out RMSE" value={formatInr(metrics.rmse)} /><Stat label="Held-out R²" value={formatNumber(metrics.r2, 4)} /></div><section className="card" style={{ marginTop: "1rem" }}><h3>Model comparison</h3><DataTable rows={metrics.comparison || []} columns={["model", "mae", "mse", "rmse", "r2", "training_seconds"]} /><p className="helper">These are random-row held-out results, not a guarantee of future or unseen-route performance.</p></section></>;
  else if (page === "Dataset") content = <div className="grid grid-2"><section className="card"><h3>Dataset profile</h3><DataTable rows={[{ records: quality.rows, columns: quality.columns, duplicates: quality.duplicate_rows, target: data.dataset?.metadata?.target, train_rows: data.dataset?.metadata?.training_rows, test_rows: data.dataset?.metadata?.test_rows }]} columns={["records", "columns", "duplicates", "target", "train_rows", "test_rows"]} /></section><section className="card"><h3>Feature contract</h3><DataTable rows={(data.dataset?.metadata?.features || []).map((feature) => ({ feature, type: data.dataset.metadata.categorical_features.includes(feature) ? "categorical" : "numeric", used_by_model: "Yes" }))} columns={["feature", "type", "used_by_model"]} /></section></div>;
  else if (page === "History") content = <section className="card"><div className="actions"><div><h3 style={{ margin: 0 }}>Current session</h3><p className="muted">Only predictions made in this browser session are stored here.</p></div>{items.length > 0 && <button className="ghost" onClick={clear}>Clear history</button>}</div>{items.length ? <DataTable rows={items.map((x) => ({ route: `${x.source.city} → ${x.destination.city}`, fare: formatInr(x.predicted_price), distance: formatKm(x.distance_km), model: x.model }))} columns={["route", "fare", "distance", "model"]} /> : <p className="muted">No predictions yet. Generate an estimate to see it here.</p>}</section>;
  else content = <section className="card"><h3>About SkyCast</h3><p>SkyCast is a historical airfare-estimation system using a saved regression pipeline. It is not a live inventory, booking, or market-price service.</p><h3>Methodology and limitations</h3><p>The deployed model uses historical route, airline, time band, class, duration, booking-window and geographic features. It safely accepts new airport locations but flags routes outside the six training cities as less reliable. The displayed error band is prediction ± held-out MAE, not a confidence interval.</p></section>;

  return <div className="app-shell"><button className="hamburger" onClick={() => setMenuOpen(!menuOpen)} aria-label="Open menu">☰</button><aside className={`sidebar ${menuOpen ? "open" : ""}`}><div className="brand"><div className="brand-mark">✈</div><div><h1>SKYCAST</h1><p>AI Airfare Intelligence</p></div></div><nav className="nav-list">{PAGES.map((item) => <a href={`#${item.toLowerCase().replaceAll(" ", "-")}`} className={`nav-link ${page === item ? "active" : ""}`} key={item} onClick={(e) => { e.preventDefault(); navigate(item); }}><span>{icons[item]}</span><span>{item}</span></a>)}</nav><div className="nav-bottom"><a className="nav-link" href="#settings"><span>⚙</span>Settings</a><a className="nav-link" href="#help"><span>?</span>Help</a></div></aside><div className={`overlay ${menuOpen ? "show" : ""}`} onClick={() => setMenuOpen(false)} /><main className="main"><header className="topbar"><h2>{page}</h2><p>{page === "Dashboard" ? "Predict your fare before you book." : "Historical model estimates with transparent methodology."}</p></header>{content}</main></div>;
}

function Field({ label, children }) { return <div className="field"><label>{label}</label>{children}</div>; }
function DataTable({ rows, columns }) { return <div style={{ overflowX: "auto" }}><table className="table"><thead><tr>{columns.map((column) => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{typeof row[column] === "number" ? formatNumber(row[column], column === "r2" ? 4 : 2) : row[column] ?? "—"}</td>)}</tr>)}</tbody></table></div>; }
function PredictionCard({ result, metrics }) { if (!result) return <section className="card"><h3>Your model estimate</h3><p className="muted">Search and select a route, then enter flight details to run the saved Random Forest model.</p></section>; const range = result.expected_price_range; return <section className="card"><div className="eyebrow">ESTIMATED AIRFARE</div><div className="price">{formatInr(result.predicted_price)}</div><p className="muted">Approximate prediction error band: {formatInr(range.low)} – {formatInr(range.high)}</p><div className="notice">This is an approximate historical/model error band based on held-out MAE, not a statistical confidence interval.</div><div className="route-visual"><strong>{result.source.city} ({result.source.iata}) → {result.destination.city} ({result.destination.iata})</strong><div className="helper">{formatKm(result.distance_km)} · {result.summary.airline} · {result.summary.class}</div></div><DataTable rows={[{ stops: formatStops(result.summary.stops), duration: formatDuration(result.summary.duration), booking_window: `${result.summary.days_left} days`, model: result.model }]} columns={["stops", "duration", "booking_window", "model"]} /><div className="grid grid-3" style={{ marginTop: "1rem" }}><Stat label="MAE" value={formatInr(metrics.mae)} /><Stat label="RMSE" value={formatInr(metrics.rmse)} /><Stat label="R²" value={formatNumber(metrics.r2, 4)} /></div>{result.out_of_training_distribution && <div className="notice" style={{ marginTop: "1rem" }}>{result.reliability_note}</div>}</section>; }
