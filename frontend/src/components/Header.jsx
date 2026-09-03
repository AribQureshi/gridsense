import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function Header({ activeTab, onTabChange }) {
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    api
      .health()
      .then((res) => setStatus(res.model_loaded ? "online" : "no-model"))
      .catch(() => setStatus("offline"));
  }, []);

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "whatif", label: "What-If" },
    { id: "diagnostics", label: "Diagnostics" },
    { id: "models", label: "Model Comparison" },
  ];

  const statusColor =
    status === "online" ? "var(--accent-good)" : status === "checking" ? "var(--text-muted)" : "var(--accent-alert)";
  const statusLabel =
    status === "online" ? "Backend connected" : status === "checking" ? "Connecting…" : "Backend offline";

  return (
    <header style={{ borderBottom: "1px solid var(--border)" }}>
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          padding: "24px 24px 0",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 22 }}>GridSense</h1>
            <p style={{ margin: "4px 0 0", color: "var(--text-muted)", fontSize: 14 }}>
              Delhi electricity demand — regression forecasting &amp; diagnostics
            </p>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              color: "var(--text-muted)",
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: statusColor,
                display: "inline-block",
              }}
            />
            {statusLabel}
          </div>
        </div>

        <nav style={{ display: "flex", gap: 4, marginTop: 20 }}>
          {tabs.map((tab) => {
            const isActive = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                style={{
                  background: "none",
                  border: "none",
                  borderBottom: isActive ? "2px solid var(--accent-heat)" : "2px solid transparent",
                  color: isActive ? "var(--text)" : "var(--text-muted)",
                  padding: "10px 14px",
                  fontSize: 14,
                  fontWeight: isActive ? 600 : 400,
                  cursor: "pointer",
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
