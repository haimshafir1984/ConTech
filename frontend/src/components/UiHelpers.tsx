import React from "react";

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
