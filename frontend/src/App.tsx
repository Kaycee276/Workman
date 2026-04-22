import { useState } from "react";
import { useWorkman } from "./hooks/useWorkman";
import { IssueCard } from "./components/IssueCard";
import { LogConsole } from "./components/LogConsole";
import type { LogRange } from "./types";
import "./App.css";

export default function App() {
  const { issues, logs, steps, connected, range, setRange } = useWorkman();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const issueList = Object.values(issues).sort(
    (a, b) =>
      new Date(b.started_at).getTime() - new Date(a.started_at).getTime(),
  );

  const toggleSelect = (id: string) =>
    setSelectedId((prev) => (prev === id ? null : id));

  const filterLabel = selectedId
    ? `#${selectedId.split("#")[1] ?? selectedId}`
    : "all";

  return (
    <div className="layout">
      <header className="header">
        <img src="/workman.png" alt="Workman" className="logo" />
        <span className="conn">
          <span className={`dot ${connected ? "on" : "off"}`} />
          {connected ? "connected" : "reconnecting..."}
        </span>
      </header>

      <main className="main">
        <aside className="issues-panel">
          <div className="panel-title">Issues</div>
          {issueList.length === 0 ? (
            <div className="empty">
              No issues yet. Apply on Drips — Workman will take it from there.
            </div>
          ) : (
            issueList.map((issue) => (
              <IssueCard
                key={issue.id}
                issue={issue}
                steps={steps}
                selected={selectedId === issue.id}
                onClick={() => toggleSelect(issue.id)}
              />
            ))
          )}
        </aside>

        <section className="logs-panel">
          <div className="logs-header">
            <span>Logs</span>
            <div className="logs-controls">
              <select
                className="range-select"
                value={range}
                onChange={(e) => setRange(e.target.value as LogRange)}
              >
                <option value="1h">Past 1 hour</option>
                <option value="24h">Past 24 hours</option>
                <option value="3d">Past 3 days</option>
              </select>
              <span className="filter-label">{filterLabel}</span>
            </div>
          </div>
          <LogConsole logs={logs} filterId={selectedId} />
        </section>
      </main>
    </div>
  );
}
