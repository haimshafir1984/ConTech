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
  icon: React.ReactNode;
  label: string;
  description: string;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

// ─── SVG Icons ───────────────────────────────────────────────────────────────
const IcoWorkshop = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="12" y2="17"/></svg>;
const IcoPlanning = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>;
const IcoLayers = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>;
const IcoCorrections = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>;
const IcoAreaAnalysis = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M2 20h20"/><path d="M4 20V10l8-8 8 8v10"/><path d="M9 20v-5h6v5"/></svg>;
const IcoDashboard = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>;
const IcoDrawingData = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="12" y2="17"/></svg>;
const IcoInvoices = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>;
const IcoWorker = (s=18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;

const NAV_GROUPS: NavGroup[] = [
  {
    label: "מנהל פרויקט",
    items: [
      { id: "workshop",     icon: IcoWorkshop(),     label: "סדנת עבודה",      description: "העלאה וניתוח תוכניות" },
      { id: "planning",     icon: IcoPlanning(),     label: "הגדרת תכולה",     description: "סימון קירות ופתחים" },
      { id: "layers",       icon: IcoLayers(),       label: "שכבות מנהל",      description: "כמויות לפי חדרים" },
      { id: "corrections",  icon: IcoCorrections(),  label: "תיקונים ידניים",  description: "הוספה/הסרת קירות" },
      { id: "areaAnalysis", icon: IcoAreaAnalysis(), label: "ניתוח שטחים",     description: "חדרים ומדידות" },
    ],
  },
  {
    label: "נתונים ודוחות",
    items: [
      { id: "dashboard",   icon: IcoDashboard(),   label: "דשבורד",       description: "סקירה ודוחות BOQ" },
      { id: "drawingData", icon: IcoDrawingData(), label: "נתוני שרטוט",  description: "יצוא CSV/JSON" },
      { id: "invoices",    icon: IcoInvoices(),    label: "חשבוניות",     description: "חישוב תשלומים" },
    ],
  },
  {
    label: "צד שטח",
    items: [
      { id: "worker", icon: IcoWorker(), label: "ממשק עובד", description: "סימון ודיווח בשטח" },
    ],
  },
];

const ALL_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);

// Bottom nav shown on mobile — most important 4 tabs + "more"
const BOTTOM_NAV: Array<{ id: TabId; icon: React.ReactNode; label: string }> = [
  { id: "workshop",  icon: IcoWorkshop(22),  label: "סדנה" },
  { id: "planning",  icon: IcoPlanning(22),  label: "תכולה" },
  { id: "worker",    icon: IcoWorker(22),    label: "עובד" },
  { id: "dashboard", icon: IcoDashboard(22), label: "דשבורד" },
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
                    background: "var(--orange)",
                    borderRadius: "2px 0 0 2px",
                  }}
                />
              )}
              <span style={{ flexShrink: 0, display: "flex", alignItems: "center", color: active ? "#fff" : "rgba(255,255,255,0.7)" }}>{item.icon}</span>
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
          background: "var(--navy)",
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
          <span style={{ lineHeight: 1, display: "flex", alignItems: "center", color: "var(--navy)" }}>{activeItem.icon}</span>
          <span style={{ fontWeight: 700, fontSize: 14, color: "var(--navy)", flex: 1 }}>
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
          <span style={{ display: "flex", alignItems: "center", color: "var(--navy)" }}>{activeItem.icon}</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: "var(--navy)", lineHeight: 1.2 }}>
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
          {activeTab === "workshop"     && <WorkshopPage onNavigatePlanning={() => setActiveTab("planning")} />}
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
                  color: active ? "var(--orange)" : "#6B7280",
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
                      background: "var(--orange)",
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
              background: "var(--navy)",
              height: "100%",
              display: "flex",
              flexDirection: "column",
              overflowY: "auto",
              boxShadow: "-4px 0 24px rgba(0,0,0,0.3)",
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
