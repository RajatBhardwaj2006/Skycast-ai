import TopBar from "../components/TopBar.jsx";
import { formatInr, formatKm } from "../utils/format";

export default function History({ history, onClear }) {
  return (
    <>
      <TopBar title="Prediction History" description="Stored in this browser tab only. Closing the tab clears the list." />
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
          <h2>This session</h2>
          {history.length > 0 && (
            <button className="btn btn-ghost" type="button" onClick={onClear}>
              Clear
            </button>
          )}
        </div>
        {!history.length && <p className="muted">No estimates yet.</p>}
        {history.map((item, idx) => (
          <div className="history-item" key={idx}>
            <div>
              <strong>
                {item.source?.city} ({item.source?.iata}) → {item.destination?.city} ({item.destination?.iata})
              </strong>
              <div className="muted">
                {formatKm(item.distance_km)} · {item.summary?.airline} · {item.summary?.class}
              </div>
            </div>
            <strong>{formatInr(item.predicted_price)}</strong>
          </div>
        ))}
      </section>
    </>
  );
}
