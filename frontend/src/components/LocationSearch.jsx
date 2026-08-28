import { useEffect, useId, useRef, useState } from "react";
import { searchLocations } from "../services/api";

export default function LocationSearch({ label, value, onChange, placeholder }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const boxId = useId();
  const timer = useRef(null);

  useEffect(() => {
    if (value) {
      setQuery("");
      setResults([]);
      setOpen(false);
      setError("");
    }
  }, [value]);

  function lookup(text) {
    clearTimeout(timer.current);
    setQuery(text);
    setError("");
    if (!text.trim()) {
      setResults([]);
      setOpen(false);
      return;
    }
    timer.current = setTimeout(async () => {
      setLoading(true);
      try {
        const data = await searchLocations(text.trim());
        setResults(data.results || []);
        setOpen(true);
        setActive(0);
        setError("");
      } catch (err) {
        setResults([]);
        setOpen(false);
        setError(err.message === "Location not found." ? "Location not found." : err.message);
      } finally {
        setLoading(false);
      }
    }, 180);
  }

  function select(item) {
    onChange(item);
    setQuery("");
    setOpen(false);
    setError("");
  }

  function onKeyDown(event) {
    if (!open || !results.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((i) => (i + 1) % results.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => (i - 1 + results.length) % results.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      select(results[active]);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  }

  if (value) {
    return (
      <div className="field">
        <label>{label}</label>
        <div className="selected-chip">
          <div>
            <strong>
              {value.city} ({value.iata})
            </strong>
            <div className="muted" style={{ fontSize: "0.8rem" }}>
              {value.airport}
            </div>
          </div>
          <button type="button" aria-label={`Clear ${label}`} onClick={() => onChange(null)}>
            ✕
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="field combo">
      <label htmlFor={boxId}>{label}</label>
      <input
        id={boxId}
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        placeholder={placeholder}
        value={query}
        onChange={(e) => lookup(e.target.value)}
        onKeyDown={onKeyDown}
        autoComplete="off"
      />
      {loading ? <div className="helper">Searching airports…</div> : null}
      {error ? (
        <div className="error">
          {error}
          <div className="helper">Try another city or airport.</div>
        </div>
      ) : null}
      {open && results.length > 0 ? (
        <div className="combo-panel" role="listbox">
          {results.map((item, index) => (
            <button
              type="button"
              key={item.iata}
              role="option"
              className={`combo-item ${index === active ? "active" : ""}`}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => select(item)}
            >
              <strong>
                {item.city} · {item.iata}
              </strong>
              <span>{item.airport}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
