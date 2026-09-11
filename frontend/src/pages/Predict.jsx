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

function calculateEstimatedDuration(distanceKm, stops) {
  if (!distanceKm || distanceKm <= 0) return 2.25;
  const flightTime = distanceKm / 600 + 0.35;
  const layover = stops === "one" ? 2.5 : stops === "two_or_more" ? 5.0 : 0.0;
  return Math.max(0.75, Math.round((flightTime + layover) * 4) / 4);
}

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

  // Auto-calibrate suggested duration whenever route or stops change
  useEffect(() => {
    if (liveDistance && !sameCity) {
      const suggested = calculateEstimatedDuration(liveDistance, stops);
      setDuration(suggested);
    }
  }, [liveDistance, stops, sameCity]);

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
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <label htmlFor="duration">Duration (hours)</label>
              {liveDistance != null && (
                <span className="muted" style={{ fontSize: "0.75rem" }}>
                  Suggested: {calculateEstimatedDuration(liveDistance, stops)}h
                </span>
              )}
            </div>
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
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="eyebrow" style={{ color: "#b7e4e0" }}>
                {result.market_calibration?.calibration_label || "ESTIMATED MARKET FARE"}
              </div>
              <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.5rem", borderRadius: "4px", background: result.market_calibration?.validation_status === "validated_loro" ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)", color: result.market_calibration?.validation_status === "validated_loro" ? "#10b981" : "#f59e0b", fontWeight: "bold" }}>
                {result.market_calibration?.validation_status === "validated_loro" ? "✓ LORO Validated" : "⚠ Experimental"}
              </span>
            </div>
            <div className="price">{formatInr(result.predicted_price)}</div>
            <p style={{ fontSize: "0.85rem", opacity: 0.9 }}>
              {result.market_calibration?.note || "Market-calibrated fare estimate."}
            </p>
            <p style={{ marginTop: "0.6rem" }}>
              <strong>{result.source.city} ({result.source.iata}) → {result.destination.city} ({result.destination.iata})</strong>
              <br />
              {result.summary.airline} • {result.summary.class} • {formatStops(result.summary.stops)}
            </p>

            {/* Historical Baseline Comparison Pill */}
            {result.historical_baseline && (
              <div style={{ marginTop: "0.8rem", padding: "0.6rem 0.8rem", borderRadius: "6px", background: "rgba(255, 255, 255, 0.05)", border: "1px solid rgba(255, 255, 255, 0.15)", fontSize: "0.82rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                  <span style={{ color: "#94a3b8" }}>2022 Historical ML Baseline:</span>
                  <strong>{formatInr(result.historical_baseline.smearing_corrected_price || result.historical_baseline.raw_model_price)}</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                  <span style={{ color: "#94a3b8" }}>Macro Inflation Adjustment:</span>
                  <strong style={{ color: "#38bdf8" }}>+{formatInr(result.market_calibration?.macro_adjustment_inr || 0)} ({result.market_calibration?.inflation_multiplier || 1}x)</strong>
                </div>
                <p style={{ margin: "0.3rem 0 0", fontSize: "0.75rem", color: "#cbd5e1" }}>
                  {result.historical_baseline.note}
                </p>
              </div>
            )}

            {result.reliability_tier === "Good historical coverage" ? (
              <div style={{ marginTop: "0.8rem", padding: "0.5rem 0.8rem", borderRadius: "6px", background: "rgba(16, 185, 129, 0.15)", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
                <strong style={{ color: "#10b981" }}>✓ {result.reliability_tier}</strong>
                <p style={{ margin: "0.2rem 0 0", fontSize: "0.82rem", opacity: 0.9 }}>{result.reliability_note}</p>
              </div>
            ) : (
              <div style={{ marginTop: "0.8rem", padding: "0.5rem 0.8rem", borderRadius: "6px", background: "rgba(245, 158, 11, 0.15)", border: "1px solid rgba(245, 158, 11, 0.3)" }}>
                <strong style={{ color: "#f59e0b" }}>⚠️ {result.reliability_tier || "Limited historical coverage"}</strong>
                <p style={{ margin: "0.2rem 0 0", fontSize: "0.82rem", opacity: 0.9 }}>{result.reliability_note}</p>
              </div>
            )}
            <p className="muted" style={{ marginTop: "0.6rem", fontSize: "0.82rem" }}>
              Typical calibration error: ±{formatInr(result.uncertainty?.typical_error_inr || result.expected_price_range?.error_band)} (Expected range: {formatInr(result.expected_price_range?.low)} – {formatInr(result.expected_price_range?.high)})
            </p>
          </section>

          <section className="card">
            <h2>Fare Summary & Analytics</h2>
            <table className="table">
              <tbody>
                <tr>
                  <th>Route</th>
                  <td>
                    {result.source.city} ({result.source.iata}) → {result.destination.city} ({result.destination.iata})
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
                  <th>2026 Market Estimate</th>
                  <td><strong>{formatInr(result.predicted_price)}</strong></td>
                </tr>
                <tr>
                  <th>2022 ML Baseline</th>
                  <td>{result.historical_baseline ? formatInr(result.historical_baseline.raw_model_price) : "N/A"}</td>
                </tr>
                <tr>
                  <th>Directional 2022 Median</th>
                  <td>{result.historical_comparables?.median ? formatInr(result.historical_comparables.median) : "N/A"}</td>
                </tr>
                <tr>
                  <th>Reliability</th>
                  <td>{result.reliability_tier || (result.out_of_training_distribution ? "Limited coverage" : "Good coverage")}</td>
                </tr>
              </tbody>
            </table>
          </section>

          {result.historical_comparables?.samples?.length > 0 && (
            <section className="card" style={{ gridColumn: "1 / -1" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "0.5rem" }}>
                <div>
                  <h2>Nearest Historical Flight Tickets (2022 Records)</h2>
                  <span style={{ fontSize: "0.8rem", padding: "0.15rem 0.4rem", borderRadius: "4px", background: result.historical_comparables.route_match === "exact_directional" ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)", color: result.historical_comparables.route_match === "exact_directional" ? "#10b981" : "#f59e0b" }}>
                    {result.historical_comparables.route_match === "exact_directional" ? "✓ Exact Directional Route" : "Corridor Similarity"}
                  </span>
                </div>
                <div style={{ fontSize: "0.88rem" }}>
                  2022 Historical Median: <strong>{formatInr(result.historical_comparables.median)}</strong>
                </div>
              </div>
              <p className="muted" style={{ fontSize: "0.82rem", margin: "0.4rem 0 0.8rem" }}>
                {result.historical_comparables.note || "Actual tickets from historical data matching this airline, class, stops, duration, and booking window."}
              </p>
              <table className="table" style={{ fontSize: "0.85rem" }}>
                <thead>
                  <tr>
                    <th>Comparable Route</th>
                    <th>Match Type</th>
                    <th>Airline</th>
                    <th>Stops</th>
                    <th>Duration</th>
                    <th>Days Left</th>
                    <th>Distance</th>
                    <th style={{ textAlign: "right" }}>2022 Fare</th>
                  </tr>
                </thead>
                <tbody>
                  {result.historical_comparables.samples.map((c, i) => (
                    <tr key={i}>
                      <td>{c.source_city} → {c.destination_city}</td>
                      <td>
                        <span style={{ fontSize: "0.75rem", padding: "0.1rem 0.35rem", borderRadius: "3px", background: c.match_type === "exact_route" ? "rgba(16, 185, 129, 0.15)" : "rgba(255, 255, 255, 0.08)", color: c.match_type === "exact_route" ? "#10b981" : "#94a3b8" }}>
                          {c.match_type === "exact_route" ? "Exact" : c.match_type === "reverse_route" ? "Reverse" : "Corridor"}
                        </span>
                      </td>
                      <td>{c.airline}</td>
                      <td>{formatStops(c.stops)}</td>
                      <td>{c.duration}h</td>
                      <td>{c.days_left}d</td>
                      <td>{c.distance_km} km</td>
                      <td style={{ textAlign: "right", fontWeight: "bold" }}>{formatInr(c.price)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}


          <section className="card" style={{ gridColumn: "1 / -1" }}>
            <h2>Real-Time Live Airline Fares (Current Market)</h2>
            {result.live_fares?.configured ? (
              <div style={{ marginTop: "0.6rem" }}>
                <p>Provider: <strong>{result.live_fares.provider}</strong></p>
                <div style={{ display: "flex", gap: "2rem", marginTop: "0.5rem" }}>
                  <div>Lowest Live Quote: <strong>{formatInr(result.live_fares.lowest_fare)}</strong></div>
                  <div>Median Live Quote: <strong>{formatInr(result.live_fares.median_fare)}</strong></div>
                  <div>Available Offers: <strong>{result.live_fares.live_offers_count}</strong></div>
                </div>
              </div>
            ) : (
              <div style={{ marginTop: "0.6rem", padding: "0.8rem 1rem", borderRadius: "6px", background: "rgba(255, 255, 255, 0.04)", border: "1px solid rgba(255, 255, 255, 0.1)" }}>
                <strong style={{ color: "#94a3b8" }}>ℹ️ Live Carrier Offers API: Unconfigured</strong>
                <p style={{ margin: "0.3rem 0 0", fontSize: "0.82rem", opacity: 0.85 }}>
                  Real-time ticket search connector is ready for Amadeus (`AMADEUS_CLIENT_ID` / `AMADEUS_CLIENT_SECRET`) and Duffel (`DUFFEL_API_TOKEN`). When credentials are configured in your backend environment, live quotes will appear here alongside the historical ML estimate.
                </p>
              </div>
            )}
          </section>

          <section className="card">
            <h2>Model feature importance (Economy Class)</h2>
            <p className="muted" style={{ fontSize: "0.85rem" }}>
              Dynamic factors driving fares in this class. Class is isolated as its own model so physical features and booking window drive 100% of the variance.
            </p>
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

          <ModelCard modelName={result.model} />
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
