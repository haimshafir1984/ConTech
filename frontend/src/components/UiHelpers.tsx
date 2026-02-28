import React from "react";

// ── PlanningCanvasErrorBoundary ───────────────────────────────────────────────
// Catches render errors inside the SVG/canvas area (e.g. NaN coordinates, bad
// data) so a corrupt item doesn't crash the entire planning page.
interface EBState { hasError: boolean; message: string }
export class PlanningCanvasErrorBoundary extends React.Component<
  { children: React.ReactNode },
  EBState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, message: "" };
  }
  static getDerivedStateFromError(error: unknown): EBState {
    const message = error instanceof Error ? error.message : String(error);
    return { hasError: true, message };
  }
  override componentDidCatch(error: unknown, info: React.ErrorInfo) {
    console.error("[PlanningCanvasErrorBoundary]", error, info.componentStack);
  }
  override render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", gap: 12, padding: 32, color: "#fff",
          background: "#1A2744", borderRadius: 8, minHeight: 200,
        }}>
          <span style={{ fontSize: 28 }}>⚠️</span>
          <p style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>שגיאה בהצגת הקנבס</p>
          <p style={{ fontSize: 12, opacity: 0.7, margin: 0, maxWidth: 280, textAlign: "center" }}>
            {this.state.message}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, message: "" })}
            style={{ marginTop: 8, padding: "8px 20px", borderRadius: 8, background: "var(--blue)", color: "#fff", border: "none", cursor: "pointer", fontSize: 13, fontWeight: 600 }}
          >
            נסה שוב
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── ErrorAlert ────────────────────────────────────────────────────────────────
// רכיב שגיאה אחיד לכל הדפים
interface ErrorAlertProps {
  message: string;
  onDismiss?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ message, onDismiss }) => (
  <div
    role="alert"
    style={{
      display: "flex",
      alignItems: "flex-start",
      gap: 10,
      fontSize: 13,
      color: "#991B1B",
      background: "#FEF2F2",
      border: "1px solid #FECACA",
      borderRadius: 10,
      padding: "10px 14px",
    }}
  >
    <span style={{ fontSize: 16, lineHeight: 1.3 }}>⚠️</span>
    <span style={{ flex: 1 }}>{message}</span>
    {onDismiss && (
      <button
        type="button"
        onClick={onDismiss}
        aria-label="סגור שגיאה"
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "#991B1B",
          fontSize: 16,
          lineHeight: 1,
          padding: 0,
          opacity: 0.6,
        }}
        onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.opacity = "1")}
        onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.opacity = "0.6")}
      >
        ✕
      </button>
    )}
  </div>
);

// ── SkeletonCard ──────────────────────────────────────────────────────────────
// פלייסהולדר בזמן טעינת נתונים
interface SkeletonCardProps {
  rows?: number;
  height?: number;
}

const pulse: React.CSSProperties = {
  background: "linear-gradient(90deg, #F1F5F9 25%, #E2E8F0 50%, #F1F5F9 75%)",
  backgroundSize: "200% 100%",
  animation: "skeleton-pulse 1.4s ease infinite",
  borderRadius: 8,
};

export const SkeletonCard: React.FC<SkeletonCardProps> = ({ rows = 3, height = 16 }) => (
  <>
    <style>{`
      @keyframes skeleton-pulse {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
    `}</style>
    <div
      style={{
        background: "#fff",
        border: "1px solid #E2E8F0",
        borderRadius: 12,
        padding: "16px 18px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          style={{
            ...pulse,
            height,
            width: i === 0 ? "60%" : i % 2 === 0 ? "85%" : "75%",
          }}
        />
      ))}
    </div>
  </>
);

// ── SkeletonGrid ──────────────────────────────────────────────────────────────
// גריד של skeleton cards — לסדנת עבודה ולדשבורד
interface SkeletonGridProps {
  count?: number;
  columns?: string;
}

export const SkeletonGrid: React.FC<SkeletonGridProps> = ({
  count = 4,
  columns = "repeat(auto-fill, minmax(180px, 1fr))",
}) => (
  <div style={{ display: "grid", gridTemplateColumns: columns, gap: 14 }}>
    {Array.from({ length: count }).map((_, i) => (
      <SkeletonCard key={i} rows={2} height={14} />
    ))}
  </div>
);
