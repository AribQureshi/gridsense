import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { api } from "../api.js";

export default function ModelComparison() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.modelsCompare().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <p style={{ color: "var(--accent-alert)" }}>Couldn't load model comparison: {error}</p>;
  if (!data) return <p style={{ color: "var(--text-muted)" }}>Loading model comparison…</p>;

  const best = data.reduce((min, row) => (row.RMSE < min.RMSE ? row : min), data[0]);

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: 20,
      }}
    >
      <h3 style={{ fontSize: 15, marginBottom: 4 }}>Held-out test RMSE by model</h3>
      <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 0, marginBottom: 16 }}>
        Lower is better. <span style={{ color: "var(--accent-good)" }}>{best.model}</span> is the best performer on unseen data.
      </p>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ left: -10 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="model" tick={{ fill: "var(--text-muted)", fontSize: 12, fontFamily: "var(--font-mono)" }} />
          <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }} />
          <Tooltip
            contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4 }}
            labelStyle={{ color: "var(--text)" }}
          />
          <Bar dataKey="RMSE" radius={[4, 4, 0, 0]}>
            {data.map((row) => (
              <Cell key={row.model} fill={row.model === best.model ? "var(--accent-good)" : "var(--accent-heat)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, marginTop: 20 }}>
        <thead>
          <tr style={{ textAlign: "left", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>
            <th style={{ padding: "6px 8px", fontWeight: 400 }}>Model</th>
            <th style={{ padding: "6px 8px", fontWeight: 400 }}>RMSE</th>
            <th style={{ padding: "6px 8px", fontWeight: 400 }}>MAE</th>
            <th style={{ padding: "6px 8px", fontWeight: 400 }}>R²</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.model} style={{ borderBottom: "1px solid var(--border)" }}>
              <td className="mono" style={{ padding: "8px", color: row.model === best.model ? "var(--accent-good)" : "var(--text)" }}>
                {row.model}
              </td>
              <td className="mono" style={{ padding: "8px" }}>{row.RMSE}</td>
              <td className="mono" style={{ padding: "8px" }}>{row.MAE}</td>
              <td className="mono" style={{ padding: "8px" }}>{row.R2}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
