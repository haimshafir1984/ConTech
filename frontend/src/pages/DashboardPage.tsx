import React from "react";
import { ErrorAlert, SkeletonGrid } from "../components/UiHelpers";
import { listDatabasePlans, listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  getDashboard,
  getWorkerReportSnapshotUrl,
  getWorkerReportsSummary,
  type DashboardResponse,
  type WorkerReportSummaryItem,
  type WorkerReportsSummary,
} from "../api/managerInsightsApi";

// ─── Inject card-expand animation once ───────────────────────────────────────
const _expandStyle = document.createElement("style");
_expandStyle.textContent = `
  @keyframes card-expand {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .report-card-body { animation: card-expand 0.18s ease; }
`;
if (!document.head.querySelector("[data-dashboard-styles]")) {
  _expandStyle.setAttribute("data-dashboard-styles", "1");
  document.head.appendChild(_expandStyle);
}

// ─── HTML escape helper (prevents XSS in print template) ─────────────────────
function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// ─── Print BOQ report ─────────────────────────────────────────────────────────
function printBoqReport(
  dashboard: DashboardResponse,
  summary: WorkerReportsSummary | null,
  planName: string,
  snapshotUrl: string
) {
  const safePlanName = escapeHtml(planName);

  const boqRows = dashboard.boq_progress.map((row) => `
    <tr>
      <td>${escapeHtml(row.label)}</td>
      <td>${row.planned_qty.toFixed(2)} ${escapeHtml(row.unit)}</td>
      <td>${row.built_qty.toFixed(2)} ${escapeHtml(row.unit)}</td>
      <td>${row.remaining_qty.toFixed(2)} ${escapeHtml(row.unit)}</td>
      <td>
        <div style="background:#eee;border-radius:4px;overflow:hidden;height:10px;width:100px;display:inline-block">
          <div style="background:#FF4B4B;height:100%;width:${Math.min(100, row.progress_percent).toFixed(0)}%"></div>
        </div>
        ${row.progress_percent.toFixed(1)}%
      </td>
    </tr>`).join("");

  const reportRows = (summary?.reports ?? []).map((r) => `
    <tr>
      <td>${escapeHtml(r.date)}</td>
      <td>${escapeHtml(r.shift)}</td>
      <td>${r.report_type === "walls" ? "קירות" : "ריצוף"}</td>
      <td>${r.report_type === "walls" ? r.total_length_m.toFixed(2) + " מ'" : r.total_area_m2.toFixed(2) + " מ\"ר"}</td>
      <td>${escapeHtml(r.note || "—")}</td>
    </tr>`).join("");

  const html = `<!doctype html><html dir="rtl" lang="he"><head>
  <meta charset="UTF-8"><title>דוח פרויקט - ${safePlanName}</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 24px 32px; direction: rtl; color: #222; }
    h1 { color: #FF4B4B; margin-bottom: 4px; }
    h2 { color: #444; border-bottom: 2px solid #FF4B4B; padding-bottom: 4px; margin-top: 24px; }
    .meta { color: #666; font-size: 12px; margin-bottom: 16px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
    .kpi { border: 1px solid #ddd; border-radius: 8px; padding: 12px; text-align: center; }
    .kpi-label { font-size: 11px; color: #888; }
    .kpi-value { font-size: 20px; font-weight: bold; color: #FF4B4B; }
    .plan-image { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 12px 0; }
    table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 7px 12px; text-align: right; }
    th { background: #f5f5f5; font-weight: 600; }
    tr:nth-child(even) td { background: #fafafa; }
    .progress-bar { height: 12px; background: #f0f0f0; border-radius: 6px; overflow: hidden; display: inline-block; width: 80px; vertical-align: middle; }
    .progress-fill { height: 100%; background: #FF4B4B; }
    @media print { body { padding: 12px; } }
  </style></head><body>
  <h1>📊 דוח פרויקט: ${safePlanName}</h1>
  <div class="meta">הופק: ${new Date().toLocaleDateString("he-IL")} | סה"כ דיווחים: ${summary?.total_reports ?? 0}</div>

  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">מתוכנן כולל</div><div class="kpi-value">${dashboard.total_planned_m.toFixed(1)} מ'</div></div>
    <div class="kpi"><div class="kpi-label">בוצע בפועל</div><div class="kpi-value">${dashboard.built_m.toFixed(1)} מ'</div></div>
    <div class="kpi"><div class="kpi-label">אחוז ביצוע</div><div class="kpi-value">${dashboard.percent_complete.toFixed(1)}%</div></div>
    <div class="kpi"><div class="kpi-label">עלות מצטברת</div><div class="kpi-value">${dashboard.current_cost_ils.toLocaleString()} ₪</div></div>
  </div>

  <h2>🗺️ תוכנית + סימוני עובד</h2>
  <img src="${snapshotUrl}" class="plan-image" alt="תוכנית עם סימוני עובד" />

  <h2>📋 כתב כמויות</h2>
  <table>
    <thead><tr><th>קטגוריה</th><th>מתוכנן</th><th>בוצע</th><th>נותר</th><th>אחוז ביצוע</th></tr></thead>
    <tbody>${boqRows || '<tr><td colspan="5">אין נתונים</td></tr>'}</tbody>
  </table>

  <h2>👷 פירוט דיווחי עובד</h2>
  <table>
    <thead><tr><th>תאריך</th><th>משמרת</th><th>סוג עבודה</th><th>כמות</th><th>הערה</th></tr></thead>
    <tbody>${reportRows || '<tr><td colspan="5">אין דיווחים</td></tr>'}</tbody>
  </table>

  ${summary ? `<div style="margin-top:16px;font-size:13px;border-top:1px solid #ddd;padding-top:12px;">
    <b>סיכום:</b>
    סה"כ קירות: ${summary.total_walls_m.toFixed(2)} מ' |
    סה"כ ריצוף: ${summary.total_floor_m2.toFixed(2)} מ"ר
  </div>` : ""}
  </body></html>`;

  const w = window.open("", "_blank");
  if (w) {
    w.document.write(html);
    w.document.close();
    // Use the opened window's own setTimeout so the timer belongs to that window's lifecycle
    w.setTimeout(() => { if (!w.closed) w.print(); }, 800);
  }
}

// ─── Export BOQ as CSV ────────────────────────────────────────────────────────
function exportBoqCsv(dashboard: DashboardResponse, planName: string) {
  const rows: string[][] = [
    ["קטגוריה", "יחידה", "מתוכנן", "בוצע", "נותר", "התקדמות %"],
  ];
  for (const row of dashboard.boq_progress) {
    rows.push([
      row.label,
      row.unit,
      row.planned_qty.toFixed(2),
      row.built_qty.toFixed(2),
      row.remaining_qty.toFixed(2),
      row.progress_percent.toFixed(1) + "%",
    ]);
  }
  // Summary row
  rows.push(["", "", "", "", "", ""]);
  rows.push(["סה\"כ קירות מ'", "מ'", dashboard.planned_walls_m.toFixed(2), dashboard.built_walls_m.toFixed(2), (dashboard.planned_walls_m - dashboard.built_walls_m).toFixed(2), ""]);
  rows.push(["סה\"כ ריצוף מ\"ר", "מ\"ר", dashboard.planned_floor_m2.toFixed(2), dashboard.built_floor_m2.toFixed(2), (dashboard.planned_floor_m2 - dashboard.built_floor_m2).toFixed(2), ""]);

  const csv = rows.map((r) => r.map((c) => `"${c.replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `boq-${planName}-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}


// ─── Report card with image preview ──────────────────────────────────────────
const ReportCard: React.FC<{
  report: WorkerReportSummaryItem;
  planId: string;
}> = ({ report, planId }) => {
  const [expanded, setExpanded] = React.useState(false);
  const snapUrl = getWorkerReportSnapshotUrl(planId, report.id);

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 text-sm text-right"
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold">{report.date}</span>
          <span className="text-slate-500">{report.shift}</span>
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${report.report_type === "walls" ? "bg-cyan-100 text-cyan-700" : "bg-orange-100 text-orange-700"}`}>
            {report.report_type === "walls" ? "קירות" : "ריצוף"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-bold text-[#FF4B4B]">
            {report.report_type === "walls"
              ? `${report.total_length_m.toFixed(2)} מ'`
              : `${report.total_area_m2.toFixed(2)} מ"ר`}
          </span>
          <span className="text-slate-400">{expanded ? "▲" : "▼"}</span>
        </div>
      </button>

      {expanded && (
        <div className="report-card-body p-4 space-y-3 bg-white">
          {report.note && (
            <div className="text-xs text-slate-600 bg-slate-50 rounded p-2">הערה: {report.note}</div>
          )}
          <div className="text-xs text-slate-500">פריטים שסומנו: {report.items_count}</div>
          <div className="relative bg-slate-100 rounded border border-slate-200 overflow-hidden" style={{ minHeight: 80 }}>
            <img
              src={snapUrl}
              alt="תוכנית עם סימוני עובד"
              className="w-full rounded"
              loading="lazy"
              style={{ display: "block", transition: "opacity 0.25s", opacity: 1 }}
              onLoad={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "1"; }}
            />
            <div className="absolute top-2 left-2 bg-black/50 text-white text-[10px] px-2 py-0.5 rounded">
              תוכנית + סימונים
            </div>
          </div>
          <a
            href={snapUrl}
            download={`report_${planId}_${report.date}.png`}
            className="inline-flex items-center gap-1 text-xs text-[#FF4B4B] hover:underline"
          >
            ⬇️ הורד תמונה
          </a>
        </div>
      )}
    </div>
  );
};

// ─── Main DashboardPage ───────────────────────────────────────────────────────
export const DashboardPage: React.FC = () => {
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState("");
  const [dashboard, setDashboard] = React.useState<DashboardResponse | null>(null);
  const [summary, setSummary] = React.useState<WorkerReportsSummary | null>(null);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [activeView, setActiveView] = React.useState<"dashboard" | "reports" | "boq">("dashboard");

  React.useEffect(() => {
    void (async () => {
      try {
        // Load from both memory (workshop) and DB
        const [memPlans, dbPlans] = await Promise.allSettled([
          listWorkshopPlans(),
          listDatabasePlans(),
        ]);
        const all: PlanSummary[] = [];
        const seen = new Set<string>();
        for (const res of [memPlans, dbPlans]) {
          if (res.status === "fulfilled") {
            for (const p of res.value) {
              if (!seen.has(p.id)) { seen.add(p.id); all.push(p); }
            }
          }
        }
        setPlans(all);
        if (all.length > 0) setSelectedPlanId(all[0].id);
      } catch (e) { console.error(e); setError("שגיאה בטעינת תוכניות."); }
    })();
  }, []);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    void (async () => {
      try {
        setLoading(true); setError("");
        const [dash, sum] = await Promise.allSettled([
          getDashboard(selectedPlanId),
          getWorkerReportsSummary(selectedPlanId),
        ]);
        if (dash.status === "fulfilled") setDashboard(dash.value);
        else setError("שגיאה בטעינת דשבורד.");
        if (sum.status === "fulfilled") setSummary(sum.value);
      } finally { setLoading(false); }
    })();
  }, [selectedPlanId]);

  const pct = dashboard?.percent_complete ?? 0;
  const snapshotUrl = React.useMemo(() => getWorkerReportSnapshotUrl(selectedPlanId), [selectedPlanId]);
  const selectedPlan = React.useMemo(() => plans.find((p) => p.id === selectedPlanId), [plans, selectedPlanId]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-[#31333F]">📊 דשבורד פרויקט</h2>
          <p className="text-xs text-slate-500 mt-1">מעקב ביצוע, כתב כמויות ודוחות.</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <select
            className="bg-white border border-slate-300 rounded px-2 py-2 text-sm min-w-[200px]"
            value={selectedPlanId}
            onChange={(e) => setSelectedPlanId(e.target.value)}
          >
            {plans.length === 0 && <option value="">אין תוכניות</option>}
            {plans.map((p) => <option key={p.id} value={p.id}>{p.plan_name}</option>)}
          </select>
          {dashboard && (
            <>
              <button
                type="button"
                onClick={() => exportBoqCsv(dashboard, selectedPlan?.plan_name ?? "פרויקט")}
                style={{ background: "var(--s50)", border: "1px solid var(--s300)", color: "var(--s700)", padding: "7px 14px", borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: "pointer" }}
              >
                📥 ייצוא CSV
              </button>
              <button
                type="button"
                onClick={() => printBoqReport(dashboard, summary, selectedPlan?.plan_name ?? "פרויקט", snapshotUrl)}
                style={{ background: "var(--blue)", border: "none", color: "#fff", padding: "7px 14px", borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: "pointer" }}
              >
                🖨️ הדפס / PDF
              </button>
            </>
          )}
        </div>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError("")} />}

      {/* Tab selector */}
      <div className="flex border-b border-[#E6E6EA] bg-white rounded-t-lg px-4 gap-0 shadow-sm">
        {([
          { id: "dashboard", label: "📊 סקירה" },
          { id: "boq", label: "📋 כתב כמויות" },
          { id: "reports", label: "📸 דוחות עובד" },
        ] as { id: typeof activeView; label: string }[]).map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActiveView(t.id)}
            className={`px-5 py-3 text-sm border-b-[3px] -mb-px ${activeView === t.id ? "border-blue-600 text-blue-600 font-semibold" : "border-transparent text-slate-600"}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading && <SkeletonGrid count={4} columns="repeat(2, 1fr)" />}
      {!dashboard && !loading && (
        <div className="bg-white border border-[#E6E6EA] rounded-lg p-8 text-center text-slate-400">בחר תוכנית להצגת נתונים.</div>
      )}

      {/* ── DASHBOARD VIEW ── */}
      {dashboard && activeView === "dashboard" && (
        <>
          {/* KPI grid */}
          <section style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
            {([
              { label: "⚡ התקדמות כללית", value: `${pct.toFixed(1)}%`, sub: `${dashboard.average_daily_m.toFixed(2)} מ'/יום`, topColor: "var(--blue)", extraContent: (
                <div style={{ height: 5, background: "var(--s200)", borderRadius: 3, overflow: "hidden", marginBottom: 8 }}>
                  <div style={{ width: `${Math.min(100, pct)}%`, height: "100%", background: "var(--blue)", borderRadius: 3 }} />
                </div>
              ) },
              { label: "✓ בוצע בפועל", value: `${dashboard.built_m.toFixed(1)} מ'`, sub: `מתוך ${dashboard.total_planned_m.toFixed(1)} מ' מתוכנן`, topColor: "var(--green)", extraContent: null },
              { label: "⏳ נותר לביצוע", value: `${dashboard.remaining_m.toFixed(1)} מ'`, sub: dashboard.days_to_finish != null ? `~${dashboard.days_to_finish.toFixed(0)} ימים` : "", topColor: "var(--amber)", extraContent: null },
              { label: "💰 עלות מצטברת", value: `${dashboard.current_cost_ils.toLocaleString()} ₪`, sub: `שטח: ${dashboard.planned_floor_m2.toFixed(0)} מ"ר`, topColor: "#6366F1", extraContent: null },
            ] as { label: string; value: string; sub: string; topColor: string; extraContent: React.ReactNode }[]).map((c) => (
              <div key={c.label} style={{ background: "#fff", borderTop: `3px solid ${c.topColor}`, borderRadius: "var(--r)", padding: "18px 20px", boxShadow: "var(--sh1)" }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--s400)", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>{c.label}</div>
                <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: -1.5, lineHeight: 1, marginBottom: 8, color: "var(--s900)" }}>{c.value}</div>
                {c.extraContent}
                {c.sub && <div style={{ fontSize: 12, color: "var(--s400)" }}>{c.sub}</div>}
              </div>
            ))}
          </section>

          {/* Dash grid: BOQ table left + side panels right */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 330px", gap: 16 }}>

            {/* BOQ table */}
            <div style={{ background: "#fff", borderRadius: "var(--r)", boxShadow: "var(--sh1)", overflow: "hidden" }}>
              <div style={{ padding: "13px 20px", borderBottom: "1px solid var(--s100)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: 13, fontWeight: 700 }}>📋 ריכוז כמויות — BOQ</span>
                {dashboard.boq_progress.length > 0 && (
                  <button type="button" onClick={() => setActiveView("boq")} style={{ fontSize: 12, color: "var(--blue)", background: "none", border: "none", cursor: "pointer", fontWeight: 600 }}>
                    הצג הכל ←
                  </button>
                )}
              </div>
              {dashboard.boq_progress.length === 0 ? (
                <div style={{ padding: "32px 20px", textAlign: "center", color: "var(--s400)", fontSize: 13 }}>
                  אין נתוני BOQ. הגדר תכולה בעמוד תכנון.
                </div>
              ) : (
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr>
                      {["פריט עבודה", "כמות", "התקדמות", "סטטוס"].map((h) => (
                        <th key={h} style={{ textAlign: "right", padding: "9px 16px", fontSize: 11, fontWeight: 700, color: "var(--s400)", background: "var(--s50)", borderBottom: "1px solid var(--s200)" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dashboard.boq_progress.map((row) => {
                      const pctRow = Math.min(100, row.progress_percent);
                      const barColor = pctRow >= 100 ? "var(--green)" : pctRow >= 50 ? "var(--blue)" : pctRow > 0 ? "var(--amber)" : "var(--s300)";
                      const badge = pctRow >= 100
                        ? { label: "✓ הושלם", bg: "var(--green-50)", color: "var(--green)" }
                        : pctRow > 0
                        ? { label: "בעבודה", bg: "var(--blue-100)", color: "var(--blue)" }
                        : { label: "ממתין", bg: "var(--s100)", color: "var(--s500)" };
                      return (
                        <tr
                          key={row.label}
                          style={{ borderBottom: "1px solid var(--s100)" }}
                          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--s50)")}
                          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                        >
                          <td style={{ padding: "10px 16px", fontWeight: 600, color: "var(--s900)" }}>{row.label}</td>
                          <td style={{ padding: "10px 16px", color: "var(--s500)" }}>{row.planned_qty.toFixed(1)} {row.unit}</td>
                          <td style={{ padding: "10px 16px" }}>
                            <span style={{ fontWeight: 700, color: barColor, marginLeft: 6 }}>{pctRow.toFixed(0)}%</span>
                            <span style={{ width: 80, height: 5, background: "var(--s200)", borderRadius: 3, overflow: "hidden", display: "inline-block", verticalAlign: "middle" }}>
                              <span style={{ display: "block", height: "100%", width: `${pctRow}%`, background: barColor, borderRadius: 3 }} />
                            </span>
                          </td>
                          <td style={{ padding: "10px 16px" }}>
                            <span style={{ display: "inline-flex", alignItems: "center", padding: "3px 8px", borderRadius: 20, fontSize: 10, fontWeight: 700, background: badge.bg, color: badge.color }}>
                              {badge.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>

            {/* Right column: category progress + activity feed */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Category progress */}
              <div style={{ background: "#fff", borderRadius: "var(--r)", boxShadow: "var(--sh1)", overflow: "hidden" }}>
                <div style={{ padding: "13px 20px", borderBottom: "1px solid var(--s100)" }}>
                  <span style={{ fontSize: 13, fontWeight: 700 }}>📊 לפי קטגוריה</span>
                </div>
                <div style={{ padding: "14px 20px" }}>
                  {[
                    { name: "🧱 קירות", pct: dashboard.planned_walls_m > 0 ? Math.min(100, (dashboard.built_walls_m / dashboard.planned_walls_m) * 100) : 0, color: "var(--blue)" },
                    { name: "🪵 ריצוף", pct: dashboard.planned_floor_m2 > 0 ? Math.min(100, (dashboard.built_floor_m2 / dashboard.planned_floor_m2) * 100) : 0, color: "var(--green)" },
                  ].map((cat) => (
                    <div key={cat.name} style={{ marginBottom: 15 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 12 }}>
                        <span style={{ fontWeight: 600 }}>{cat.name}</span>
                        <span style={{ fontWeight: 700, color: cat.color }}>{cat.pct.toFixed(0)}%</span>
                      </div>
                      <div style={{ height: 7, background: "var(--s200)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${cat.pct}%`, background: cat.color, borderRadius: 4 }} />
                      </div>
                    </div>
                  ))}
                  {dashboard.timeline.length > 0 && (
                    <div style={{ marginTop: 8, paddingTop: 12, borderTop: "1px solid var(--s100)", fontSize: 11, color: "var(--s400)" }}>
                      קצב ממוצע: <strong style={{ color: "var(--s700)" }}>{dashboard.average_daily_m.toFixed(2)} מ'/יום</strong>
                    </div>
                  )}
                </div>
              </div>

              {/* Activity feed */}
              <div style={{ background: "#fff", borderRadius: "var(--r)", boxShadow: "var(--sh1)", overflow: "hidden", flex: 1 }}>
                <div style={{ padding: "13px 20px", borderBottom: "1px solid var(--s100)" }}>
                  <span style={{ fontSize: 13, fontWeight: 700 }}>🕐 עדכונים אחרונים</span>
                </div>
                <div style={{ padding: "12px 20px", fontSize: 12 }}>
                  {dashboard.recent_reports.length === 0 ? (
                    <div style={{ textAlign: "center", color: "var(--s400)", padding: "16px 0" }}>אין דיווחים עדיין</div>
                  ) : dashboard.recent_reports.slice(0, 6).map((r) => {
                    const color = r.report_type === "walls" ? "var(--blue)" : "var(--amber)";
                    const qty = r.report_type === "walls"
                      ? `${r.total_length_m.toFixed(2)} מ'`
                      : `${r.total_area_m2.toFixed(2)} מ"ר`;
                    return (
                      <div key={r.id} style={{ display: "flex", gap: 10, padding: "8px 0", borderBottom: "1px solid var(--s100)" }}>
                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: color, marginTop: 3, flexShrink: 0 }} />
                        <div>
                          <div>
                            <span style={{ fontWeight: 600 }}>{r.shift}</span>
                            {" — "}{r.report_type === "walls" ? "קירות" : "ריצוף"} · {qty}
                          </div>
                          <div style={{ fontSize: 11, color: "var(--s400)", marginTop: 2 }}>{r.date}{r.note ? ` · ${r.note}` : ""}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>
          </div>
        </>
      )}

      {/* ── BOQ VIEW ── */}
      {dashboard && activeView === "boq" && (
        <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
          <h3 className="text-sm font-semibold mb-3">📋 כתב כמויות – מתוכנן מול ביצוע</h3>

          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            {[
              { label: 'קירות מתוכנן', value: `${dashboard.planned_walls_m.toFixed(2)} מ'` },
              { label: 'קירות בוצע', value: `${dashboard.built_walls_m.toFixed(2)} מ'` },
              { label: 'ריצוף מתוכנן', value: `${dashboard.planned_floor_m2.toFixed(2)} מ"ר` },
              { label: 'ריצוף בוצע', value: `${dashboard.built_floor_m2.toFixed(2)} מ"ר` },
            ].map((c) => (
              <div key={c.label} className="bg-slate-50 rounded p-3">
                <div className="text-xs text-slate-500">{c.label}</div>
                <div className="font-bold">{c.value}</div>
              </div>
            ))}
          </div>

          {dashboard.boq_progress.length === 0 ? (
            <p className="text-sm text-slate-400">אין נתונים. הגדר תכולה בעמוד תכנון.</p>
          ) : (
            <div className="overflow-auto">
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, minWidth: 560 }}>
                <thead>
                  <tr>
                    {["קטגוריה", "יחידה", "מתוכנן", "בוצע", "נותר", "התקדמות"].map((h) => (
                      <th key={h} style={{ textAlign: "right", padding: "9px 16px", fontSize: 11, fontWeight: 700, color: "var(--s500)", background: "var(--s50)", borderBottom: "2px solid var(--s200)" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dashboard.boq_progress.map((row, i) => {
                    const pctRow = Math.min(100, row.progress_percent);
                    const barColor = pctRow >= 100 ? "var(--green)" : pctRow >= 50 ? "var(--blue)" : "var(--amber)";
                    return (
                      <tr key={row.label} style={{ borderBottom: i < dashboard.boq_progress.length - 1 ? "1px solid var(--s100)" : "none" }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "var(--s50)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        <td style={{ padding: "10px 16px", fontWeight: 600, color: "var(--s900)" }}>{row.label}</td>
                        <td style={{ padding: "10px 16px", color: "var(--s500)" }}>{row.unit}</td>
                        <td style={{ padding: "10px 16px", color: "var(--s700)" }}>{row.planned_qty.toFixed(2)}</td>
                        <td style={{ padding: "10px 16px", color: "var(--green)", fontWeight: 600 }}>{row.built_qty.toFixed(2)}</td>
                        <td style={{ padding: "10px 16px", color: "var(--s500)" }}>{row.remaining_qty.toFixed(2)}</td>
                        <td style={{ padding: "10px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 80, height: 5, background: "var(--s200)", borderRadius: 3, overflow: "hidden", flexShrink: 0, display: "inline-block" }}>
                              <div style={{ height: "100%", width: `${pctRow}%`, background: barColor, borderRadius: 3, transition: "width 0.3s" }} />
                            </div>
                            <span style={{ fontSize: 11, fontWeight: 700, color: barColor }}>{row.progress_percent.toFixed(0)}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* ── REPORTS VIEW ── */}
      {activeView === "reports" && (
        <div className="space-y-3">
          {/* Plan snapshot */}
          {selectedPlanId && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold">🗺️ תוכנית כוללת + כל הסימונים</h3>
                <a href={snapshotUrl} download={`project_${selectedPlanId}.png`} className="text-xs text-[#FF4B4B] hover:underline">
                  ⬇️ הורד תמונה
                </a>
              </div>
              <img
                src={snapshotUrl}
                alt="תוכנית + כל סימוני עובד"
                className="w-full rounded border border-slate-200"
                loading="lazy"
              />
            </section>
          )}

          {/* Summary */}
          {summary && (
            <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">📊 סיכום ביצוע</h3>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-3 text-center">
                  <div className="text-xs text-slate-500">דיווחים</div>
                  <div className="font-bold text-xl">{summary.total_reports}</div>
                </div>
                <div className="bg-slate-50 rounded p-3 text-center">
                  <div className="text-xs text-slate-500">סה"כ קירות</div>
                  <div className="font-bold text-xl text-cyan-600">{summary.total_walls_m.toFixed(2)} מ'</div>
                </div>
                <div className="bg-slate-50 rounded p-3 text-center">
                  <div className="text-xs text-slate-500">סה"כ ריצוף</div>
                  <div className="font-bold text-xl text-orange-500">{summary.total_floor_m2.toFixed(2)} מ"ר</div>
                </div>
              </div>
            </section>
          )}

          {/* Individual reports */}
          {summary && summary.reports.length > 0 ? (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-slate-700 px-1">דיווחים בודדים (לחץ להרחבה + תמונה)</h3>
              {summary.reports.slice().reverse().map((r) => (
                <ReportCard key={r.id} report={r} planId={selectedPlanId} />
              ))}
            </div>
          ) : (
            <div className="bg-white border border-[#E6E6EA] rounded-lg p-8 text-center text-slate-400">
              אין דיווחי עובד עדיין.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
