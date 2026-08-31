import { useEffect, useMemo, useState } from "react";
import TopBar from "../components/TopBar.jsx";
import LocationSearch from "../components/LocationSearch.jsx";
import { getCatalog, getMetrics, predictFare } from "../services/api";
import { formatDuration, formatInr, formatKm, formatStops } from "../utils/format";

const STOP_OPTIONS = [
  { value: "zero", label: "Non-stop" },
  { value: "one", label: "1 Stop" },
  { value: "two_or_more", label: "2+ Stops" },
];

function haversine(a, b) {
  if (!a || !b) return null;
  const R = 6371.0088;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b.latitude - a.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const aa =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(a.latitude)) * Math.cos(toRad(b.latitude)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(aa), Math.sqrt(1 - aa));
}

export default function Predict({ onPredicted }) {
  const [catalog, setCatalog] = useState(null);
  const [source, setSource] = useState(null);
  const [destination, setDestination] = useState(null);
  const [airline, setAirline] = useState("Air India");
  const [travelClass, setTravelClass] = useState("Economy");
  const [stops, setStops] = useState("zero");
  const [departure, setDeparture] = useState("Morning");
  const [arrival, setArrival] = useState("Evening");
  const [duration, setDuration] = useState(2.25);
  const [daysLeft, setDaysLeft] = useState(15);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [prefsOpen, setPrefsOpen] = useState(false);

  useEffect(() => {
    getCatalog()
      .then((data) => {
        setCatalog(data);
        if (data.example_input) {
          setAirline(data.example_input.airline || "Air India");
          setTravelClass(data.example_input.class || "Economy");
        }
      })
      .catch(() => {});
  }, []);

  const liveDistance = useMemo(() => haversine(source, destination), [source, destination]);
  const sameCity = source && destination && source.iata === destination.iata;

  function swap() {
    setSource(destination);
    setDestination(source);
  }

  async function fillExample() {
    const example = catalog?.example_input;
    try {
      const [fromRes, toRes] = await Promise.all([
        fetch(`${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}/locations/search?q=Delhi`).then((r) => r.json()),
        fetch(`${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}/locations/search?q=Mumbai`).then((r) => r.json()),
      ]);
      setSource(fromRes.results?.[0] || null);
      setDestination(toRes.results?.[0] || null);
    } catch {
      /* catalog still fills other fields */
    }
    setAirline(example?.airline || "Air India");
    setTravelClass(example?.class || "Economy");
    setStops(example?.stops || "one");
    setDeparture(example?.departure_time || "Morning");
    setArrival(example?.arrival_time || "Evening");
    setDuration(example?.duration || 5.5);
    setDaysLeft(example?.days_left || 15);
    setError("");
    setResult(null);
  }

  function reset() {
    setSource(null);
    setDestination(null);
    setResult(null);
    setError("");
    setDuration(2.25);
    setDaysLeft(15);
  }

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    if (!source || !destination) {
      setError("Search and select both departure and destination airports.");
      return;
    }
    if (sameCity) {
      setError("Departure and destination cannot be the same.");
      return;
    }
    setLoading(true);
    try {
      const payload = {
        airline,
        source_iata: source.iata,
        destination_iata: destination.iata,
        departure_time: departure,
        arrival_time: arrival,
        stops,
        class: travelClass,
        duration: Number(duration),
        days_left: Number(daysLeft),
      };
      const prediction = await predictFare(payload);
      setResult(prediction);
      onPredicted?.(prediction);
    } catch (err) {
      setError(err.message || "Unable to calculate fare. Please check your inputs.");
    } finally {
      setLoading(false);
    }
  }

  const maxImportance = Math.max(0.0001, ...(result?.feature_importance || []).map((f) => f.importance || 0));

  return (
    <>
      <TopBar
        title="Predict Your Fare"
        description="Estimate airfare using historical flight data and machine learning."
      />
      <form className="card search-card" onSubmit={onSubmit}>
        <div className="search-head">
          <h2>Where are you flying?</h2>
          <p className="muted">Enter your journey details to estimate the expected airfare.</p>
        </div>
        <div className="route-row">
          <LocationSearch id="from" label="FROM" value={source} onSelect={setSource} placeholder="🔍 Search city or airport..." />
          <button type="button" className="swap-btn" onClick={swap} aria-label="Swap origin and destination">
            ⇄
          </button>
          <LocationSearch id="to" label="TO" value={destination} onSelect={setDestination} placeholder="🔍 Search city or airport..." />
        </div>
        {sameCity && <p className="empty-msg">Departure and destination cannot be the same.</p>}
        {source && destination && !sameCity && (
          <div className="route-visual">
            <div className="route-end">
              <strong>
                {source.city.toUpperCase()} ({source.iata})
              </strong>
              <span className="muted">{source.airport}</span>
            </div>
            <div className="route-line">
              <span className="bar" />
              ✈
              <span className="bar" />
            </div>
            <div className="route-end" style={{ textAlign: "right" }}>
              <strong>
                {destination.city.toUpperCase()} ({destination.iata})
              </strong>
              <span className="muted">{destination.airport}</span>
            </div>
          </div>
        )}
        {liveDistance != null && !sameCity && (
          <div className="distance-pill">Route distance: {formatKm(liveDistance)}</div>
        )}

        <div className="details-grid">
          <div className="field">
            <label htmlFor="airline">Airline</label>
            <select id="airline" value={airline} onChange={(e) => setAirline(e.target.value)}>
              {(catalog?.airlines || ["Air India", "Vistara", "Indigo"]).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="class">Class</label>
            <select id="class" value={travelClass} onChange={(e) => setTravelClass(e.target.value)}>
              {(catalog?.classes || ["Economy", "Business"]).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="stops">Stops</label>
            <select id="stops" value={stops} onChange={(e) => setStops(e.target.value)}>
              {STOP_OPTIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="dep">Departure</label>
            <select id="dep" value={departure} onChange={(e) => setDeparture(e.target.value)}>
              {(catalog?.departure_times || []).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="arr">Arrival</label>
            <select id="arr" value={arrival} onChange={(e) => setArrival(e.target.value)}>
              {(catalog?.arrival_times || []).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="duration">Duration (hours)</label>
            <input
              id="duration"
              type="number"
              min="0.5"
              max="48"
              step="0.25"
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
            />
          </div>
        </div>
        <div className="field" style={{ marginTop: "1rem" }}>
          <label htmlFor="days">Days Until Departure — {daysLeft} days</label>
          <input id="days" className="slider" type="range" min="1" max="49" value={daysLeft} onChange={(e) => setDaysLeft(e.target.value)} />
        </div>

        <details className="accordion" open={prefsOpen} onToggle={(e) => setPrefsOpen(e.target.open)}>
          <summary>Additional Preferences</summary>
          <p className="coming-soon">Coming soon — not used by the current prediction model.</p>
          <div className="details-grid">
            <div className="field">
              <label>Seat Preference</label>
              <select disabled defaultValue="window">
                <option value="window">Window</option>
                <option value="middle">Middle</option>
                <option value="aisle">Aisle</option>
              </select>
              <p className="coming-soon">Seat preference is currently not included in the trained model.</p>
            </div>
            <div className="field">
              <label>Baggage</label>
              <select disabled defaultValue="0">
                <option>No extra baggage</option>
                <option>15 kg</option>
                <option>20 kg</option>
                <option>30 kg</option>
              </select>
            </div>
            <div className="field">
              <label>Lounge Access</label>
              <select disabled>
                <option>No</option>
                <option>Yes</option>
              </select>
            </div>
          </div>
        </details>

        <div className="cta-row">
          <button className="btn btn-primary" type="submit" disabled={loading}>
            ✈ {loading ? "Analyzing historical airfare patterns..." : "Predict Fare"}
          </button>
          <button className="btn btn-ghost" type="button" onClick={fillExample}>
            Try Example
          </button>
          <button className="btn btn-ghost" type="button" onClick={reset}>
            Reset
          </button>
        </div>
        {error && <p className="empty-msg">{error}</p>}
        {!result && !error && !loading && <p className="muted" style={{ marginTop: "0.8rem" }}>Enter your flight details to get started.</p>}
      </form>

      {result && (
        <div className="grid grid-2" style={{ marginTop: "1rem" }}>
          <section className="card result-hero">
            <div className="eyebrow" style={{ color: "#b7e4e0" }}>
              Estimated airfare
            </div>
            <div className="price">{formatInr(result.predicted_price)}</div>
            <p>Historical model estimate — not a live ticket price.</p>
            <p style={{ marginTop: "0.6rem" }}>
              {result.source.city} → {result.destination.city}
              <br />
              {result.summary.airline} • {result.summary.class} • {formatStops(result.summary.stops)}
            </p>
            {result.reliability_note && <p className="coming-soon">{result.reliability_note}</p>}
          </section>
          <section className="card">
            <h2>Fare summary</h2>
            <table className="table">
              <tbody>
                <tr>
                  <th>Flight</th>
                  <td>
                    {result.source.city} → {result.destination.city}
                  </td>
                </tr>
                <tr>
                  <th>Distance</th>
                  <td>{formatKm(result.distance_km)}</td>
                </tr>
                <tr>
                  <th>Airline</th>
                  <td>{result.summary.airline}</td>
                </tr>
                <tr>
                  <th>Class</th>
                  <td>{result.summary.class}</td>
                </tr>
                <tr>
                  <th>Stops</th>
                  <td>{formatStops(result.summary.stops)}</td>
                </tr>
                <tr>
                  <th>Duration</th>
                  <td>{formatDuration(result.summary.duration)}</td>
                </tr>
                <tr>
                  <th>Booking window</th>
                  <td>{result.summary.days_left} days</td>
                </tr>
                <tr>
                  <th>Fare band</th>
                  <td>{result.fare_band} (from training price distribution)</td>
                </tr>
              </tbody>
            </table>
          </section>
          <section className="card">
            <h2>Why this fare?</h2>
            <p className="muted">Actual grouped feature importance from the trained model.</p>
            <div className="bars" style={{ marginTop: "0.8rem" }}>
              {(result.feature_importance || []).slice(0, 8).map((item) => (
                <div className="bar-row" key={item.feature}>
                  <span>{item.feature}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${(item.importance / maxImportance) * 100}%` }} />
                  </div>
                  <span>{(item.importance * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </section>
          <ModelCard />
        </div>
      )}
    </>
  );
}

function ModelCard() {
  const [metrics, setMetrics] = useState(null);
  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {});
  }, []);
  if (!metrics) return null;
  return (
    <section className="card">
      <h2>Powered by</h2>
      <p>
        <strong>{metrics.best_model}</strong>
      </p>
      <p className="muted">R² {metrics.r2?.toFixed(3)} · MAE {formatInr(metrics.mae)} · RMSE {formatInr(metrics.rmse)}</p>
    </section>
  );
}
