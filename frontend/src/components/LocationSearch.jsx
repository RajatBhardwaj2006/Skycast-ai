import { useEffect, useRef, useState } from "react";
import { searchLocations } from "../services/api";

export default function LocationSearch({ id, label, value, onSelect, placeholder }) {
  const [query, setQuery] = useState(value ? `${value.city} (${value.iata})` : "");
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState([]);
  const [active, setActive] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const boxRef = useRef(null);

  useEffect(() => {
    if (value) setQuery(`${value.city} (${value.iata})`);
  }, [value]);

  useEffect(() => {
    function onDoc(event) {
      if (boxRef.current && !boxRef.current.contains(event.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  useEffect(() => {
    const text = query.trim();
    if (value && query === `${value.city} (${value.iata})`) {
      setResults([]);
      setError("");
      return;
    }
    if (text.length < 2) {
      setResults([]);
      setError("");
      return;
    }
    const handle = setTimeout(async () => {
      setLoading(true);
      try {
        const data = await searchLocations(text);
        setResults(data.results || []);
        setError("");
        setOpen(true);
        setActive(0);
      } catch (err) {
        setResults([]);
        setError(err.message || "Location not found.");
        setOpen(false);
      } finally {
        setLoading(false);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [query, value]);

  function choose(item) {
    onSelect(item);
    setQuery(`${item.city} (${item.iata})`);
    setOpen(false);
    setError("");
  }

  function onKeyDown(event) {
    if (!open || !results.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((i) => Math.min(i + 1, results.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      choose(results[active]);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="field combo" ref={boxRef}>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-autocomplete="list"
        placeholder={placeholder}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          if (value) onSelect(null);
        }}
        onFocus={() => results.length && setOpen(true)}
        onKeyDown={onKeyDown}
        autoComplete="off"
      />
      {value && <span className="selected-chip">{value.airport}</span>}
      {loading && <span className="muted">Searching…</span>}
      {error && (
        <p className="empty-msg" role="status">
          {error}
          <br />
          Try another city or airport.
        </p>
      )}
      {open && results.length > 0 && (
        <div className="suggest" id={`${id}-list`} role="listbox">
          {results.map((item, index) => (
            <button
              type="button"
              key={item.iata}
              role="option"
              className={index === active ? "active" : ""}
              onMouseEnter={() => setActive(index)}
              onClick={() => choose(item)}
            >
              <div className="city">
                {item.city} · {item.iata}
              </div>
              <div className="meta">{item.airport}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
