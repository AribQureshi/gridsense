import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function WhatIf() {
  const [temp, setTemp] = useState(30);
  const [humidity, setHumidity] = useState(50);
  const [precipitation, setPrecipitation] = useState(0);
  const [date, setDate] = useState("2020-06-15");
  const [isLockdown, setIsLockdown] = useState(false);

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Debounce: wait 350ms after the last slider move before calling the API,
  // so dragging a slider doesn't fire a request on every pixel of movement.
  useEffect(() => {
    setLoading(true);
    setError(null);
    const timer = setTimeout(() => {
      api
        .predict({
          date,
          temp_mean_c: temp,
          humidity_mean_pct: humidity,
          precipitation_mm: precipitation,
          is_lockdown: isLockdown,
        })
        .then((res) => setResult(res))
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }, 350);
    return () => clearTimeout(timer);
  }, [temp, humidity, precipitation, date, isLockdown]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: 24,
        }}
      >
        <h3 style={{ fontSize: 15, marginBottom: 20 }}>Simulation inputs</h3>

        <Field label="Date">
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            style={inputStyle}
          />
        </Field>

        <Slider label="Mean temperature" value={temp} onChange={setTemp} min={-5} max={45} unit="°C" accent="var(--accent-heat)" />
        <Slider label="Humidity" value={humidity} onChange={setHumidity} min={0} max={100} unit="%" accent="var(--accent-cool)" />
        <Slider label="Precipitation" value={precipitation} onChange={setPrecipitation} min={0} max={50} unit="mm" accent="var(--accent-cool)" />

        <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            id="lockdown"
            checked={isLockdown}
            onChange={(e) => setIsLockdown(e.target.checked)}
          />
          <label htmlFor="lockdown" style={{ fontSize: 14, color: "var(--text-muted)" }}>
            Simulate lockdown-like demand shock
          </label>
        </div>
      </div>

      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: 24,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <h3 style={{ fontSize: 15, marginBottom: 20 }}>Predicted demand</h3>

        {error && (
          <p style={{ color: "var(--accent-alert)", fontSize: 13 }}>
            Couldn't reach the backend: {error}
          </p>
        )}

        {!error && (
          <>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 56,
                lineHeight: 1,
                color: loading ? "var(--text-muted)" : "var(--accent-heat)",
                transition: "color 150ms",
              }}
            >
              {result ? result.predicted_demand_mu : "—"}
              <span style={{ fontSize: 20, color: "var(--text-muted)", marginLeft: 8 }}>MU</span>
            </div>

            {result && (
              <div style={{ marginTop: 24, fontSize: 13, color: "var(--text-muted)" }}>
                <Row label="Heating degree days" value={result.inputs_used.hdd} />
                <Row label="Cooling degree days" value={result.inputs_used.cdd} />
                <Row label="Day of week" value={result.inputs_used.is_weekend ? "Weekend" : "Weekday"} />
                {result.note && (
                  <p style={{ marginTop: 12, fontSize: 12, fontStyle: "italic" }}>{result.note}</p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: "block", fontSize: 13, color: "var(--text-muted)", marginBottom: 6 }}>{label}</label>
      {children}
    </div>
  );
}

function Slider({ label, value, onChange, min, max, unit, accent }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
        <span style={{ color: "var(--text-muted)" }}>{label}</span>
        <span style={{ fontFamily: "var(--font-mono)" }}>
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: accent }}
      />
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}>
      <span>{label}</span>
      <span className="mono" style={{ color: "var(--text)" }}>{value}</span>
    </div>
  );
}

const inputStyle = {
  width: "100%",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 4,
  color: "var(--text)",
  padding: "8px 10px",
  fontFamily: "var(--font-mono)",
  fontSize: 13,
};
