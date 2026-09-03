// Thin wrapper around the FastAPI backend built in Step 5.
// Base URL points at your local uvicorn server. In a real deployment
// you'd move this to an environment variable, but for a local portfolio
// project a hardcoded localhost URL is honest about how it actually runs.
const BASE_URL = "http://127.0.0.1:8000";

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  historical: () => request("/historical"),
  predict: (payload) =>
    request("/predict", { method: "POST", body: JSON.stringify(payload) }),
  modelsCompare: () => request("/models/compare"),
  diagnostics: () => request("/diagnostics"),
};
