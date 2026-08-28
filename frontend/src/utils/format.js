export function formatInr(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

export function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

export function formatKm(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `~${Math.round(Number(value)).toLocaleString("en-IN")} km`;
}

export function formatDuration(hours) {
  const n = Number(hours);
  if (!n) return "—";
  const h = Math.floor(n);
  const m = Math.round((n - h) * 60);
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

export function formatStops(value) {
  if (value === "zero") return "Non-stop";
  if (value === "one") return "1 Stop";
  if (value === "two_or_more") return "2+ Stops";
  return value;
}
