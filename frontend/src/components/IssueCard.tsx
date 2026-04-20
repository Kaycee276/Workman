import type { Issue, Step } from "../types";
import styles from "./IssueCard.module.css";

interface Props {
  issue: Issue;
  steps: Step[];
  selected: boolean;
  onClick: () => void;
}

const STEP_LABELS: Record<Step, string> = {
  queued: "Queued",
  detected: "Detected",
  fetching: "Fetching",
  forking: "Forking",
  cloning: "Cloning",
  setup: "Setup",
  solving: "Solving",
  pushing: "Pushing",
  done: "Done",
};

export function IssueCard({ issue, steps, selected, onClick }: Props) {
  const stepIdx = steps.indexOf(issue.step);
  const statusClass = issue.failed
    ? styles.failed
    : issue.step === "done"
    ? styles.done
    : issue.step === "queued"
    ? styles.queued
    : "";

  return (
    <div
      className={`${styles.card} ${statusClass} ${selected ? styles.selected : ""}`}
      onClick={onClick}
    >
      <div className={styles.issueId}>{issue.id}</div>
      <div className={styles.title}>{issue.title}</div>

      <div className={styles.steps}>
        {steps.map((s, i) => {
          let dotClass = styles.dot;
          if (issue.failed && i === stepIdx) dotClass += ` ${styles.failedDot}`;
          else if (i < stepIdx || issue.step === "done") dotClass += ` ${styles.doneDot}`;
          else if (i === stepIdx) dotClass += ` ${styles.activeDot}`;
          return <span key={s} className={dotClass} title={s} />;
        })}
      </div>

      <div className={styles.stepLabel}>
        {STEP_LABELS[issue.step] ?? issue.step}
        {issue.failed && " — failed"}
      </div>

      {issue.pr_url && (
        <a
          className={styles.prLink}
          href={issue.pr_url}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
        >
          ↗ View PR
        </a>
      )}

      {issue.error && <div className={styles.error}>✗ {issue.error}</div>}
    </div>
  );
}
