import { useEffect, useState } from "react";
import { api } from "../api.js";

function vifColor(vif) {
  if (vif >= 999999) return "var(--accent-alert)";
  if (vif > 10) return "var(--accent-alert)";
  if (vif > 5) return "var(--accent-heat)";
  return "var(--accent-good)";
}

export default function Diagnostics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.diagnostics().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <p style={{ color: "var(--accent-alert)" }}>Couldn't load diagnostics: {error}</p>;
  if (!data) return <p style={{ color: "var(--text-muted)" }}>Loading diagnostics…</p>;

  return (
    <div style={{ display: "grid", gap: 24 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Panel title="Breusch-Pagan test" subtitle="Tests whether residual variance is constant (homoscedasticity)">
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 28 }}>
            p = {data.breusch_pagan.p_value}
          </div>
          <p style={{ fontSize: 13, color: data.breusch_pagan.heteroscedastic ? "var(--accent-alert)" : "var(--accent-good)" }}>
            {data.breusch_pagan.heteroscedastic
              ? "Heteroscedasticity detected (p < 0.05)"
              : "No strong evidence of heteroscedasticity"}
          </p>
        </Panel>

        <Panel title="Cook's distance" subtitle="Influential points that disproportionately affect the model fit">
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 28 }}>
            {data.cooks_distance.n_influential} / {data.cooks_distance.total_points}
          </div>
          <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
            {((data.cooks_distance.n_influential / data.cooks_distance.total_points) * 100).toFixed(1)}% of training points flagged
            (threshold: {data.cooks_distance.threshold})
          </p>
        </Panel>
      </div>

      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: 20,
        }}
      >
        <h3 style={{ fontSize: 15, marginBottom: 4 }}>Variance Inflation Factors</h3>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 0, marginBottom: 16 }}>{data.note}</p>

        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>
              <th style={{ padding: "6px 8px", fontWeight: 400 }}>Feature</th>
              <th style={{ padding: "6px 8px", fontWeight: 400 }}>VIF</th>
            </tr>
          </thead>
          <tbody>
            {data.vif_scores
              .filter((row) => row.feature !== "const")
              .sort((a, b) => b.VIF - a.VIF)
              .map((row) => (
                <tr key={row.feature} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td className="mono" style={{ padding: "8px" }}>{row.feature}</td>
                  <td style={{ padding: "8px" }}>
                    <span
                      className="mono"
                      style={{
                        color: vifColor(row.VIF),
                        fontWeight: 600,
                      }}
                    >
                      {row.VIF >= 999999 ? "∞" : row.VIF}
                    </span>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Panel({ title, subtitle, children }) {
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: 20,
      }}
    >
      <h3 style={{ fontSize: 15, marginBottom: 4 }}>{title}</h3>
      <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 0, marginBottom: 12 }}>{subtitle}</p>
      {children}
    </div>
  );
}
