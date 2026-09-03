import { useState } from "react";
import Header from "./components/Header.jsx";
import Overview from "./components/Overview.jsx";
import WhatIf from "./components/WhatIf.jsx";
import Diagnostics from "./components/Diagnostics.jsx";
import ModelComparison from "./components/ModelComparison.jsx";

export default function App() {
  const [tab, setTab] = useState("overview");

  return (
    <div>
      <Header activeTab={tab} onTabChange={setTab} />
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "24px" }}>
        {tab === "overview" && <Overview />}
        {tab === "whatif" && <WhatIf />}
        {tab === "diagnostics" && <Diagnostics />}
        {tab === "models" && <ModelComparison />}
      </main>
    </div>
  );
}
