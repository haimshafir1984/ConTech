import React from "react";
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

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState<TabId>("workshop");
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  const activeItem = ALL_ITEMS.find((i) => i.id === activeTab)!;

  return (
    <div
      dir="rtl"
      className="min-h-screen flex"
      style={{ background: "#F4F6F9", fontFamily: "'Heebo', 'Segoe UI', sans-serif" }}
    >
      {/* ===== SIDEBAR ===== */}
      <aside
        style={{
          width: sidebarOpen ? 230 : 64,
          minWidth: sidebarOpen ? 230 : 64,
          background: "#1B3A6B",
          display: "flex",
          flexDirection: "column",
          transition: "width 0.22s ease, min-width 0.22s ease",
          overflow: "hidden",
          position: "sticky",
          top: 0,
          height: "100vh",
          zIndex: 20,
          boxShadow: "2px 0 12px rgba(0,0,0,0.18)",
        }}
      >
        {/* Logo / Brand */}
        <div
          style={{
            padding: sidebarOpen ? "20px 16px 14px 16px" : "20px 0 14px 0",
            display: "flex",
            alignItems: "center",
            justifyContent: sidebarOpen ? "flex-start" : "center",
            gap: 10,
            borderBottom: "1px solid rgba(255,255,255,0.12)",
            flexShrink: 0,
          }}
        >
          <img
            src="/planex_logo.png"
            alt="Planex"
            style={{
              height: 36,
              width: "auto",
              flexShrink: 0,
              filter: "brightness(0) invert(1)",
              objectFit: "contain",
            }}
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
          {sidebarOpen && (
            <div style={{ overflow: "hidden", whiteSpace: "nowrap" }}>
              <div style={{ color: "#fff", fontWeight: 700, fontSize: 18, lineHeight: 1.2 }}>
                Planex
              </div>
              <div style={{ color: "rgba(255,255,255,0.55)", fontSize: 11, marginTop: 2 }}>
                ניהול פרויקטי בנייה
              </div>
            </div>
          )}
        </div>

        {/* Navigation Groups */}
        <nav style={{ flex: 1, overflowY: "auto", overflowX: "hidden", padding: "10px 0" }}>
          {NAV_GROUPS.map((group) => (
            <div key={group.label} style={{ marginBottom: 6 }}>
              {sidebarOpen && (
                <div
                  style={{
                    color: "rgba(255,255,255,0.4)",
                    fontSize: 10,
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    padding: "10px 18px 4px 18px",
                    whiteSpace: "nowrap",
                  }}
                >
                  {group.label}
                </div>
              )}
              {!sidebarOpen && <div style={{ height: 10 }} />}

              {group.items.map((item) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    title={sidebarOpen ? undefined : item.label}
                    onClick={() => setActiveTab(item.id)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      width: sidebarOpen ? "calc(100% - 16px)" : 48,
                      border: "none",
                      cursor: "pointer",
                      padding: sidebarOpen ? "9px 18px" : "9px 0",
                      justifyContent: sidebarOpen ? "flex-start" : "center",
                      background: isActive
                        ? "rgba(255,255,255,0.15)"
                        : "transparent",
                      borderRadius: 8,
                      margin: "1px 8px",
                      transition: "background 0.15s",
                      position: "relative",
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive)
                        (e.currentTarget as HTMLButtonElement).style.background =
                          "rgba(255,255,255,0.08)";
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive)
                        (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                    }}
                  >
                    {/* Active indicator bar */}
                    {isActive && (
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
                    <span style={{ fontSize: 18, flexShrink: 0, lineHeight: 1 }}>{item.icon}</span>
                    {sidebarOpen && (
                      <div style={{ textAlign: "right", overflow: "hidden" }}>
                        <div
                          style={{
                            color: isActive ? "#fff" : "rgba(255,255,255,0.75)",
                            fontSize: 13,
                            fontWeight: isActive ? 600 : 400,
                            whiteSpace: "nowrap",
                            lineHeight: 1.3,
                          }}
                        >
                          {item.label}
                        </div>
                        <div
                          style={{
                            color: "rgba(255,255,255,0.38)",
                            fontSize: 10,
                            whiteSpace: "nowrap",
                            lineHeight: 1.3,
                          }}
                        >
                          {item.description}
                        </div>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        {/* Collapse Toggle */}
        <div
          style={{
            borderTop: "1px solid rgba(255,255,255,0.12)",
            padding: "10px 8px",
            display: "flex",
            justifyContent: sidebarOpen ? "flex-end" : "center",
            flexShrink: 0,
          }}
        >
          <button
            type="button"
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? "כווץ תפריט" : "הרחב תפריט"}
            style={{
              background: "rgba(255,255,255,0.1)",
              border: "none",
              borderRadius: 6,
              color: "rgba(255,255,255,0.7)",
              cursor: "pointer",
              padding: "6px 10px",
              fontSize: 14,
              lineHeight: 1,
              transition: "background 0.15s",
            }}
            onMouseEnter={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.background =
                "rgba(255,255,255,0.2)")
            }
            onMouseLeave={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.background =
                "rgba(255,255,255,0.1)")
            }
          >
            {sidebarOpen ? "◂" : "▸"}
          </button>
        </div>
      </aside>

      {/* ===== MAIN AREA ===== */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
        {/* Top Bar */}
        <header
          style={{
            background: "#fff",
            borderBottom: "1px solid #E5E7EB",
            padding: "0 24px",
            height: 56,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
            position: "sticky",
            top: 0,
            zIndex: 10,
            boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
            flexShrink: 0,
          }}
        >
          {/* Page title */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 22 }}>{activeItem.icon}</span>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: "#1B3A6B", lineHeight: 1.2 }}>
                {activeItem.label}
              </div>
              <div style={{ fontSize: 11, color: "#9CA3AF", lineHeight: 1.2 }}>
                {activeItem.description}
              </div>
            </div>
          </div>

          {/* Quick-nav pills */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-start" }}>
            {[
              { id: "workshop" as TabId,     emoji: "📂" },
              { id: "planning" as TabId,     emoji: "🧱" },
              { id: "layers" as TabId,       emoji: "🎨" },
              { id: "dashboard" as TabId,    emoji: "📊" },
              { id: "invoices" as TabId,     emoji: "💰" },
              { id: "worker" as TabId,       emoji: "👷" },
            ].map(({ id, emoji }) => {
              const item = ALL_ITEMS.find((i) => i.id === id)!;
              const isActive = activeTab === id;
              return (
                <button
                  key={id}
                  type="button"
                  title={item.label}
                  onClick={() => setActiveTab(id)}
                  style={{
                    background: isActive ? "#1B3A6B" : "#F3F4F6",
                    color: isActive ? "#fff" : "#6B7280",
                    border: "none",
                    borderRadius: 20,
                    padding: "4px 10px",
                    fontSize: 13,
                    cursor: "pointer",
                    transition: "all 0.15s",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive)
                      (e.currentTarget as HTMLButtonElement).style.background = "#E5E7EB";
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive)
                      (e.currentTarget as HTMLButtonElement).style.background = "#F3F4F6";
                  }}
                >
                  <span>{emoji}</span>
                  <span style={{ fontSize: 11, fontWeight: 500 }}>{item.label}</span>
                </button>
              );
            })}
          </div>
        </header>

        {/* Page Content */}
        <main style={{ flex: 1, padding: "20px 24px 32px", overflowY: "auto" }}>
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
      </div>
    </div>
  );
};
