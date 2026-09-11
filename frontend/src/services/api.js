const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message || detail?.hint || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return data;
}

export function searchLocations(q, limit = 8) {
  const query = encodeURIComponent(q);
  return request(`/locations/search?q=${query}&limit=${limit}`);
}

export function getCatalog() {
  return request("/catalog");
}

export function getModelInfo() {
  return request("/model-info");
}

export function getMetrics() {
  return request("/metrics");
}

export function getFeatureImportance() {
  return request("/feature-importance");
}

export function getGeoExperiment() {
  return request("/geo-experiment");
}

export function getDatasetInfo() {
  return request("/dataset-info");
}

export function getValidation() {
  return request("/validation");
}

export function getRouteDistance(sourceIata, destinationIata) {
  return request(`/route-distance?source_iata=${encodeURIComponent(sourceIata)}&destination_iata=${encodeURIComponent(destinationIata)}`);
}

export function predictFare(payload) {
  return request("/predict", { method: "POST", body: JSON.stringify(payload) });
}

export { API_BASE };
