import { useEffect, useRef } from "react";
import type { LogEntry } from "../types";
import styles from "./LogConsole.module.css";

interface Props {
  logs: LogEntry[];
  filterId: string | null;
}

export function LogConsole({ logs, filterId }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  const filtered = filterId
    ? logs.filter((l) => l.issue_id === filterId || l.issue_id === null)
    : logs;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      autoScrollRef.current =
        el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (autoScrollRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [filtered.length]);

  return (
    <div ref={containerRef} className={styles.console}>
      {filtered.length === 0 && (
        <div className={styles.empty}>Logs will appear here once a pipeline starts.</div>
      )}
      {filtered.map((entry, i) => {
        const ts = entry.ts ? new Date(entry.ts).toLocaleTimeString() : "";
        const msg = entry.message ?? "";
        const cls = /error|exception|traceback|failed|✗/i.test(msg)
          ? styles.err
          : /warn/i.test(msg)
          ? styles.warn
          : /pr created|fix complete|clone complete|branch pushed|fork ready|installed|pushed|done|success/i.test(msg)
          ? styles.success
          : /pipeline started|fetching|forking|cloning|detecting|solving|pushing|analyzing|queued|polling|committing|creating pull/i.test(msg)
          ? styles.step
          : /sleeping|no new|no matches/i.test(msg)
          ? styles.dim
          : "";
        return (
          <div key={i} className={`${styles.line} ${cls}`}>
            <span className={styles.ts}>[{ts}] </span>
            <span className={styles.msg}>{msg}</span>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
