import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api } from "../api.js";

const LOCKDOWN_START = "2020-03-25";
const LOCKDOWN_END = "2020-05-23";

export default function Overview() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .historical()
      .then((res) => {
        const merged = res.dates.map((d, i) => ({
          date: d,
          actual: res.actual[i],
          predicted: res.predicted[i],
        }));
        setData(merged);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <p style={{ color: "var(--accent-alert)" }}>
        Couldn't load historical data: {error}. Is the backend running at http://127.0.0.1:8000?
      </p>
    );
  }
  if (!data) return <p style={{ color: "var(--text-muted)" }}>Loading historical data…</p>;

  const actualRmse = Math.sqrt(
    data.reduce((sum, d) => sum + (d.actual - d.predicted) ** 2, 0) / data.length
  );

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 24 }}>
        <StatCard label="Days modeled" value={data.length} />
        <StatCard label="In-sample RMSE" value={actualRmse.toFixed(1)} unit="MU" />
        <StatCard label="Avg demand" value={(data.reduce((s, d) => s + d.actual, 0) / data.length).toFixed(1)} unit="MU" />
        <StatCard label="Peak demand" value={Math.max(...data.map((d) => d.actual)).toFixed(1)} unit="MU" />
      </div>

      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: 20,
        }}
      >
        <h3 style={{ fontSize: 15, marginBottom: 4 }}>Actual vs. Model-Fitted Demand</h3>
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 0, marginBottom: 16 }}>
          Shaded region marks the COVID lockdown period — a real demand shock the model accounts for via a dedicated flag.
        </p>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={data} margin={{ left: -10 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              tickFormatter={(d) => d.slice(0, 7)}
              minTickGap={40}
            />
            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }} />
            <Tooltip
              contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4 }}
              labelStyle={{ color: "var(--text)" }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <ReferenceArea
              x1={LOCKDOWN_START}
              x2={LOCKDOWN_END}
              fill="var(--accent-alert)"
              fillOpacity={0.08}
            />
            <Line type="monotone" dataKey="actual" stroke="var(--accent-cool)" dot={false} strokeWidth={1.5} name="Actual" />
            <Line type="monotone" dataKey="predicted" stroke="var(--accent-heat)" dot={false} strokeWidth={1.5} name="Model fit" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StatCard({ label, value, unit }) {
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "14px 16px",
      }}
    >
      <div style={{ color: "var(--text-muted)", fontSize: 12 }}>{label}</div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: 22, marginTop: 4 }}>
        {value}
        {unit && <span style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: 4 }}>{unit}</span>}
      </div>
    </div>
  );
}
