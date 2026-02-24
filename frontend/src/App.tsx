import React from "react";
import { ToastProvider } from "./components/Toast";
import { ConfirmProvider } from "./components/ConfirmDialog";
import { LayerModePage } from "./pages/LayerModePage";
import { WorkshopPage } from "./pages/WorkshopPage";
import { PlanningPage } from "./pages/PlanningPage";
import { WorkerPage } from "./pages/WorkerPage";
import { DrawingDataPage } from "./pages/DrawingDataPage";
import { AreaAnalysisPage } from "./pages/AreaAnalysisPage";
import { DashboardPage } from "./pages/DashboardPage";
import { InvoicesPage } from "./pages/InvoicesPage";
import { CorrectionsPage } from "./pages/CorrectionsPage";

type TabId =
  | "workshop"
  | "planning"
  | "drawingData"
  | "corrections"
  | "areaAnalysis"
  | "dashboard"
  | "invoices"
  | "worker"
  | "layers";

interface NavItem {
  id: TabId;
  icon: string;
  label: string;
  description: string;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: "מנהל פרויקט",
    items: [
      { id: "workshop",     icon: "📂", label: "סדנת עבודה",      description: "העלאה וניתוח תוכניות" },
      { id: "planning",     icon: "🧱", label: "הגדרת תכולה",     description: "סימון קירות ופתחים" },
      { id: "layers",       icon: "🎨", label: "שכבות מנהל",      description: "כמויות לפי חדרים" },
      { id: "corrections",  icon: "✏️",  label: "תיקונים ידניים",  description: "הוספה/הסרת קירות" },
      { id: "areaAnalysis", icon: "📐", label: "ניתוח שטחים",     description: "חדרים ומדידות" },
    ],
  },
  {
    label: "נתונים ודוחות",
    items: [
      { id: "dashboard",   icon: "📊", label: "דשבורד",       description: "סקירה ודוחות BOQ" },
      { id: "drawingData", icon: "📄", label: "נתוני שרטוט",  description: "יצוא CSV/JSON" },
      { id: "invoices",    icon: "💰", label: "חשבוניות",     description: "חישוב תשלומים" },
    ],
  },
  {
    label: "צד שטח",
    items: [
      { id: "worker", icon: "👷", label: "ממשק עובד", description: "סימון ודיווח בשטח" },
    ],
  },
];

const ALL_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);

// Bottom nav shown on mobile — most important 4 tabs + "more"
const BOTTOM_NAV: Array<{ id: TabId; icon: string; label: string }> = [
  { id: "workshop",  icon: "📂", label: "סדנה" },
  { id: "planning",  icon: "🧱", label: "תכולה" },
  { id: "worker",    icon: "👷", label: "עובד" },
  { id: "dashboard", icon: "📊", label: "דשבורד" },
];

// ─── Sidebar content (shared between desktop + mobile overlay) ───────────────
const SidebarNav: React.FC<{
  activeTab: TabId;
  onNavigate: (id: TabId) => void;
}> = ({ activeTab, onNavigate }) => (
  <nav style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
    {NAV_GROUPS.map((group) => (
      <div key={group.label} style={{ marginBottom: 4 }}>
        <div
          style={{
            color: "rgba(255,255,255,0.38)",
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            padding: "10px 18px 4px",
          }}
        >
          {group.label}
        </div>
        {group.items.map((item) => {
          const active = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                width: "calc(100% - 16px)",
                border: "none",
                cursor: "pointer",
                padding: "9px 18px",
                background: active ? "rgba(255,255,255,0.15)" : "transparent",
                borderRadius: 8,
                margin: "1px 8px",
                position: "relative",
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) => {
                if (!active)
                  (e.currentTarget as HTMLButtonElement).style.background =
                    "rgba(255,255,255,0.08)";
              }}
              onMouseLeave={(e) => {
                if (!active)
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
              }}
            >
              {active && (
                <div
                  style={{
                    position: "absolute",
                    right: -8,
                    top: "50%",
                    transform: "translateY(-50%)",
                    width: 3,
                    height: 22,
                    background: "#10B981",
                    borderRadius: "2px 0 0 2px",
                  }}
                />
              )}
              <span style={{ fontSize: 17, flexShrink: 0 }}>{item.icon}</span>
              <div style={{ textAlign: "right", overflow: "hidden" }}>
                <div
                  style={{
                    color: active ? "#fff" : "rgba(255,255,255,0.78)",
                    fontSize: 13,
                    fontWeight: active ? 600 : 400,
                    whiteSpace: "nowrap",
                  }}
                >
                  {item.label}
                </div>
                <div
                  style={{
                    color: "rgba(255,255,255,0.36)",
                    fontSize: 10,
                    whiteSpace: "nowrap",
                  }}
                >
                  {item.description}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    ))}
  </nav>
);

// ─── Main App ────────────────────────────────────────────────────────────────
export const App: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState<TabId>("workshop");
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const activeItem = ALL_ITEMS.find((i) => i.id === activeTab)!;

  const navigate = (id: TabId) => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <ToastProvider>
    <ConfirmProvider>
    <div
      dir="rtl"
      style={{
        minHeight: "100dvh",
        display: "flex",
        background: "#F4F6F9",
        fontFamily: "'Heebo', 'Segoe UI', sans-serif",
      }}
    >
      {/* ══════════ DESKTOP SIDEBAR (hidden on mobile via CSS) ══════════ */}
      <aside
        className="hidden-mobile"
        style={{
          width: 220,
          minWidth: 220,
          background: "#1B3A6B",
          display: "flex",
          flexDirection: "column",
          position: "sticky",
          top: 0,
          height: "100vh",
          zIndex: 20,
          boxShadow: "2px 0 12px rgba(0,0,0,0.18)",
          flexShrink: 0,
        }}
      >
        {/* Logo */}
        <div
          style={{
            padding: "20px 16px 14px",
            display: "flex",
            alignItems: "center",
            gap: 10,
            borderBottom: "1px solid rgba(255,255,255,0.1)",
            flexShrink: 0,
          }}
        >
          <img
            src="/planex_logo.png"
            alt="Planex"
            style={{ height: 34, width: "auto", filter: "brightness(0) invert(1)" }}
            onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
          />
          <div>
            <div style={{ color: "#fff", fontWeight: 700, fontSize: 17, lineHeight: 1.2 }}>Planex</div>
            <div style={{ color: "rgba(255,255,255,0.45)", fontSize: 10, marginTop: 1 }}>ניהול פרויקטי בנייה</div>
          </div>
        </div>

        <SidebarNav activeTab={activeTab} onNavigate={navigate} />
      </aside>

      {/* ══════════ MAIN COLUMN ══════════ */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>

        {/* ── Mobile top bar ── */}
        <header
          className="show-mobile"
          style={{
            display: "none", /* overridden by CSS class on mobile */
            height: 52,
            background: "#fff",
            borderBottom: "1px solid #E5E7EB",
            padding: "0 16px",
            alignItems: "center",
            gap: 12,
            position: "sticky",
            top: 0,
            zIndex: 20,
            boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
            flexShrink: 0,
          }}
        >
          <button
            type="button"
            onClick={() => setMobileMenuOpen(true)}
            style={{
              background: "none",
              border: "none",
              fontSize: 22,
              color: "#1B3A6B",
              cursor: "pointer",
              lineHeight: 1,
              padding: "4px 2px",
              flexShrink: 0,
            }}
          >
            ☰
          </button>
          <span style={{ fontSize: 20, lineHeight: 1 }}>{activeItem.icon}</span>
          <span style={{ fontWeight: 700, fontSize: 14, color: "#1B3A6B", flex: 1 }}>
            {activeItem.label}
          </span>
        </header>

        {/* ── Desktop top bar ── */}
        <header
          className="hidden-mobile"
          style={{
            background: "#fff",
            borderBottom: "1px solid #E5E7EB",
            padding: "0 28px",
            height: 56,
            display: "flex",
            alignItems: "center",
            gap: 12,
            position: "sticky",
            top: 0,
            zIndex: 10,
            boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
            flexShrink: 0,
          }}
        >
          <span style={{ fontSize: 22 }}>{activeItem.icon}</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: "#1B3A6B", lineHeight: 1.2 }}>
              {activeItem.label}
            </div>
            <div style={{ fontSize: 11, color: "#9CA3AF", lineHeight: 1.2 }}>
              {activeItem.description}
            </div>
          </div>
        </header>

        {/* ── Page Content ── */}
        <main
          style={{
            flex: 1,
            padding: "20px 24px 32px",
            overflowY: "auto",
            /* Extra bottom padding on mobile so content isn't hidden behind bottom nav */
          }}
          className="main-content"
        >
          {activeTab === "workshop"     && <WorkshopPage />}
          {activeTab === "planning"     && <PlanningPage />}
          {activeTab === "drawingData"  && <DrawingDataPage />}
          {activeTab === "corrections"  && <CorrectionsPage />}
          {activeTab === "areaAnalysis" && <AreaAnalysisPage />}
          {activeTab === "dashboard"    && <DashboardPage />}
          {activeTab === "invoices"     && <InvoicesPage />}
          {activeTab === "worker"       && <WorkerPage />}
          {activeTab === "layers"       && <LayerModePage />}
        </main>

        {/* ── Mobile Bottom Nav ── */}
        <nav
          className="show-mobile bottom-nav"
          style={{
            display: "none", /* overridden by CSS on mobile */
            position: "fixed",
            bottom: 0,
            left: 0,
            right: 0,
            height: 64,
            background: "#fff",
            borderTop: "1px solid #E5E7EB",
            zIndex: 20,
            boxShadow: "0 -2px 10px rgba(0,0,0,0.08)",
          }}
        >
          {BOTTOM_NAV.map((item) => {
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => navigate(item.id)}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 2,
                  border: "none",
                  background: "none",
                  cursor: "pointer",
                  color: active ? "#FF4B4B" : "#6B7280",
                  padding: "6px 4px",
                }}
              >
                <span style={{ fontSize: 22, lineHeight: 1 }}>{item.icon}</span>
                <span style={{ fontSize: 10, fontWeight: active ? 700 : 400, lineHeight: 1 }}>
                  {item.label}
                </span>
                {active && (
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      width: 32,
                      height: 3,
                      background: "#FF4B4B",
                      borderRadius: "0 0 3px 3px",
                    }}
                  />
                )}
              </button>
            );
          })}
          {/* More button */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(true)}
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 2,
              border: "none",
              background: "none",
              cursor: "pointer",
              color: "#6B7280",
              padding: "6px 4px",
            }}
          >
            <span style={{ fontSize: 22, lineHeight: 1 }}>⋯</span>
            <span style={{ fontSize: 10, lineHeight: 1 }}>עוד</span>
          </button>
        </nav>
      </div>

      {/* ══════════ MOBILE OVERLAY SIDEBAR ══════════ */}
      {mobileMenuOpen && (
        <div
          style={{ position: "fixed", inset: 0, zIndex: 50, display: "flex" }}
          onClick={() => setMobileMenuOpen(false)}
        >
          {/* Panel */}
          <div
            style={{
              width: 260,
              background: "#1B3A6B",
              height: "100%",
              display: "flex",
              flexDirection: "column",
              overflowY: "auto",
              boxShadow: "4px 0 24px rgba(0,0,0,0.3)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "16px 18px",
                borderBottom: "1px solid rgba(255,255,255,0.1)",
                flexShrink: 0,
              }}
            >
              <div>
                <div style={{ color: "#fff", fontWeight: 700, fontSize: 17 }}>Planex</div>
                <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 10 }}>ניהול פרויקטי בנייה</div>
              </div>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                style={{
                  background: "rgba(255,255,255,0.1)",
                  border: "none",
                  color: "rgba(255,255,255,0.7)",
                  borderRadius: 8,
                  width: 34,
                  height: 34,
                  cursor: "pointer",
                  fontSize: 16,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                ✕
              </button>
            </div>

            <SidebarNav activeTab={activeTab} onNavigate={navigate} />
          </div>

          {/* Backdrop */}
          <div style={{ flex: 1, background: "rgba(0,0,0,0.45)" }} />
        </div>
      )}

      {/* ══════════ RESPONSIVE CSS ══════════ */}
      <style>{`
        @media (max-width: 767px) {
          .hidden-mobile { display: none !important; }
          .show-mobile { display: flex !important; }
          .bottom-nav { display: flex !important; }
          .main-content { padding: 16px 16px 80px !important; }
        }
        .bottom-nav { display: flex; flex-direction: row; }
      `}</style>
    </div>
    </ConfirmProvider>
    </ToastProvider>
  );
};
