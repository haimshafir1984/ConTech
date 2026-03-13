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

const IcoWorkshop = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" /><line x1="8" y1="13" x2="16" y2="13" /><line x1="8" y1="17" x2="12" y2="17" /></svg>;
const IcoPlanning = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><line x1="3" y1="9" x2="21" y2="9" /><line x1="9" y1="21" x2="9" y2="9" /></svg>;
const IcoLayers = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></svg>;
const IcoCorrections = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>;
const IcoAreaAnalysis = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M2 20h20" /><path d="M4 20V10l8-8 8 8v10" /><path d="M9 20v-5h6v5" /></svg>;
const IcoDashboard = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /></svg>;
const IcoDrawingData = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="8" y1="13" x2="16" y2="13" /><line x1="8" y1="17" x2="12" y2="17" /></svg>;
const IcoInvoices = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="5" width="20" height="14" rx="2" /><line x1="2" y1="10" x2="22" y2="10" /></svg>;
const IcoWorker = (s = 18) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>;

const NAV_GROUPS: NavGroup[] = [
  {
    label: "מנהל פרויקט",
    items: [
      { id: "workshop", icon: IcoWorkshop(), label: "סדנת עבודה", description: "קליטת תוכניות" },
      { id: "planning", icon: IcoPlanning(), label: "הגדרת תכולה", description: "סיווג ואישור" },
      { id: "layers", icon: IcoLayers(), label: "שכבות מנהל", description: "פירוק לפי אזורים" },
      { id: "corrections", icon: IcoCorrections(), label: "תיקונים ידניים", description: "עריכה מדויקת" },
      { id: "areaAnalysis", icon: IcoAreaAnalysis(), label: "ניתוח שטחים", description: "חדרים ומדידות" },
    ],
  },
  {
    label: "נתונים ודוחות",
    items: [
      { id: "dashboard", icon: IcoDashboard(), label: "דשבורד", description: "BOQ וביצוע" },
      { id: "drawingData", icon: IcoDrawingData(), label: "נתוני שרטוט", description: "CSV / JSON" },
      { id: "invoices", icon: IcoInvoices(), label: "חשבוניות", description: "תשלומים" },
    ],
  },
  {
    label: "צד שטח",
    items: [
      { id: "worker", icon: IcoWorker(), label: "ממשק עובד", description: "דיווח ובקרה" },
    ],
  },
];

const ALL_ITEMS: NavItem[] = NAV_GROUPS.flatMap((group) => group.items);
const BOTTOM_NAV: Array<{ id: TabId; icon: React.ReactNode; label: string }> = [
  { id: "workshop", icon: IcoWorkshop(20), label: "סדנה" },
  { id: "planning", icon: IcoPlanning(20), label: "תכולה" },
  { id: "worker", icon: IcoWorker(20), label: "שטח" },
  { id: "dashboard", icon: IcoDashboard(20), label: "דשבורד" },
];

const SidebarNav: React.FC<{ activeTab: TabId; onNavigate: (id: TabId) => void }> = ({ activeTab, onNavigate }) => (
  <nav style={{ display: "flex", flexDirection: "column", gap: 18, padding: "14px 10px 20px", overflowY: "auto" }}>
    {NAV_GROUPS.map((group) => (
      <section key={group.label}>
        <div style={{ padding: "0 8px 8px", fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", color: "#94A3B8", textTransform: "uppercase" }}>
          {group.label}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {group.items.map((item) => {
            const active = item.id === activeTab;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                style={{
                  width: "100%",
                  border: `1px solid ${active ? "#CBD5E1" : "transparent"}`,
                  background: active ? "#FFFFFF" : "transparent",
                  borderRadius: 12,
                  padding: "9px 10px",
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  textAlign: "right",
                  cursor: "pointer",
                  transition: "background 0.15s, border-color 0.15s",
                }}
              >
                <span style={{ width: 30, height: 30, borderRadius: 10, display: "grid", placeItems: "center", flexShrink: 0, background: active ? "#111827" : "#E2E8F0", color: active ? "#F8FAFC" : "#334155" }}>
                  {item.icon}
                </span>
                <span style={{ minWidth: 0, flex: 1 }}>
                  <span style={{ display: "block", color: "#0F172A", fontSize: 12, fontWeight: active ? 700 : 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {item.label}
                  </span>
                  <span style={{ display: "block", color: "#64748B", fontSize: 10, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {item.description}
                  </span>
                </span>
                <span style={{ width: 6, height: 6, borderRadius: 999, background: active ? "#10B981" : "#CBD5E1", flexShrink: 0 }} />
              </button>
            );
          })}
        </div>
      </section>
    ))}
  </nav>
);

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState<TabId>("workshop");
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const activeItem = ALL_ITEMS.find((item) => item.id === activeTab)!;

  const navigate = (id: TabId) => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <ToastProvider>
      <ConfirmProvider>
        <div dir="rtl" style={{ minHeight: "100dvh", background: "#E5E7EB", fontFamily: "'Heebo', 'Segoe UI', sans-serif" }}>
          <header style={{ height: 48, background: "#111827", color: "#F9FAFB", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "0 16px 0 18px", borderBottom: "1px solid rgba(255,255,255,0.08)", position: "sticky", top: 0, zIndex: 40 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
              <button type="button" className="show-mobile" onClick={() => setMobileMenuOpen(true)} style={{ display: "none", width: 30, height: 30, borderRadius: 8, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "#F8FAFC", cursor: "pointer", alignItems: "center", justifyContent: "center" }}>
                <span style={{ fontSize: 16, lineHeight: 1 }}>☰</span>
              </button>
              <div style={{ width: 28, height: 28, borderRadius: 8, background: "linear-gradient(135deg, #334155, #0F172A)", display: "grid", placeItems: "center", fontSize: 11, fontWeight: 800, letterSpacing: "0.08em" }}>PX</div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 700, lineHeight: 1.1 }}>Planex Pro</div>
                <div style={{ fontSize: 10, color: "#94A3B8", lineHeight: 1.1 }}>Engineering Workspace</div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0, marginInlineStart: "auto" }}>
              <span style={{ display: "flex", alignItems: "center", color: "#CBD5E1" }}>{activeItem.icon}</span>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1 }}>Workspace / {activeItem.label}</div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#F8FAFC", lineHeight: 1.2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {activeItem.description}
                </div>
              </div>
            </div>
          </header>

          <div style={{ display: "flex", minHeight: "calc(100dvh - 48px)" }}>
            <aside className="hidden-mobile" style={{ width: 250, minWidth: 250, background: "#F8FAFC", borderLeft: "1px solid #D7DEE8", display: "flex", flexDirection: "column", flexShrink: 0 }}>
              <div style={{ padding: "14px 16px 10px", borderBottom: "1px solid #E2E8F0" }}>
                <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>מערכת</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#0F172A" }}>ניווט מהיר</div>
              </div>
              <SidebarNav activeTab={activeTab} onNavigate={navigate} />
            </aside>

            <div style={{ flex: 1, minWidth: 0, background: "#EEF2F6" }}>
              <main className="main-content" style={{ height: "100%", overflowY: "auto", padding: "18px 22px 26px" }}>
                {activeTab === "workshop" && <WorkshopPage onNavigatePlanning={() => setActiveTab("planning")} />}
                {activeTab === "planning" && <PlanningPage />}
                {activeTab === "drawingData" && <DrawingDataPage />}
                {activeTab === "corrections" && <CorrectionsPage />}
                {activeTab === "areaAnalysis" && <AreaAnalysisPage />}
                {activeTab === "dashboard" && <DashboardPage />}
                {activeTab === "invoices" && <InvoicesPage />}
                {activeTab === "worker" && <WorkerPage />}
                {activeTab === "layers" && <LayerModePage />}
              </main>
            </div>
          </div>

          <nav className="show-mobile bottom-nav" style={{ display: "none", position: "fixed", bottom: 0, left: 0, right: 0, height: 58, background: "rgba(255,255,255,0.96)", backdropFilter: "blur(16px)", borderTop: "1px solid #D7DEE8", zIndex: 50 }}>
            {BOTTOM_NAV.map((item) => {
              const active = item.id === activeTab;
              return (
                <button key={item.id} type="button" onClick={() => navigate(item.id)} style={{ flex: 1, border: "none", background: "transparent", color: active ? "#111827" : "#64748B", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 3, cursor: "pointer" }}>
                  <span style={{ display: "flex", alignItems: "center" }}>{item.icon}</span>
                  <span style={{ fontSize: 10, fontWeight: active ? 700 : 500 }}>{item.label}</span>
                </button>
              );
            })}
            <button type="button" onClick={() => setMobileMenuOpen(true)} style={{ flex: 1, border: "none", background: "transparent", color: "#64748B", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 3, cursor: "pointer" }}>
              <span style={{ fontSize: 18, lineHeight: 1 }}>⋯</span>
              <span style={{ fontSize: 10, fontWeight: 500 }}>עוד</span>
            </button>
          </nav>

          {mobileMenuOpen && (
            <div style={{ position: "fixed", inset: 0, zIndex: 60, display: "flex" }} onClick={() => setMobileMenuOpen(false)}>
              <div style={{ width: 260, maxWidth: "82vw", background: "#F8FAFC", borderLeft: "1px solid #D7DEE8", boxShadow: "-16px 0 40px rgba(15,23,42,0.22)", display: "flex", flexDirection: "column" }} onClick={(e) => e.stopPropagation()}>
                <div style={{ padding: "14px 16px", borderBottom: "1px solid #E2E8F0", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "#0F172A" }}>Planex Pro</div>
                    <div style={{ fontSize: 10, color: "#64748B" }}>Navigation</div>
                  </div>
                  <button type="button" onClick={() => setMobileMenuOpen(false)} style={{ width: 30, height: 30, borderRadius: 8, border: "1px solid #CBD5E1", background: "#FFF", color: "#334155", cursor: "pointer" }}>×</button>
                </div>
                <SidebarNav activeTab={activeTab} onNavigate={navigate} />
              </div>
              <div style={{ flex: 1, background: "rgba(15,23,42,0.42)" }} />
            </div>
          )}

          <style>{`
            @media (max-width: 767px) {
              .hidden-mobile { display: none !important; }
              .show-mobile { display: flex !important; }
              .bottom-nav { display: flex !important; }
              .main-content { padding: 14px 14px 76px !important; }
            }
          `}</style>
        </div>
      </ConfirmProvider>
    </ToastProvider>
  );
};
