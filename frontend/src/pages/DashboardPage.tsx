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

// ─── Print BOQ report ─────────────────────────────────────────────────────────
function printBoqReport(
  dashboard: DashboardResponse,
  summary: WorkerReportsSummary | null,
  planName: string,
  snapshotUrl: string
) {
  const boqRows = dashboard.boq_progress.map((row) => `
    <tr>
      <td>${row.label}</td>
      <td>${row.planned_qty.toFixed(2)} ${row.unit}</td>
      <td>${row.built_qty.toFixed(2)} ${row.unit}</td>
      <td>${row.remaining_qty.toFixed(2)} ${row.unit}</td>
      <td>
        <div style="background:#eee;border-radius:4px;overflow:hidden;height:10px;width:100px;display:inline-block">
          <div style="background:#FF4B4B;height:100%;width:${Math.min(100, row.progress_percent).toFixed(0)}%"></div>
        </div>
        ${row.progress_percent.toFixed(1)}%
      </td>
    </tr>`).join("");

  const reportRows = (summary?.reports ?? []).map((r) => `
    <tr>
      <td>${r.date}</td>
      <td>${r.shift}</td>
      <td>${r.report_type === "walls" ? "קירות" : "ריצוף"}</td>
      <td>${r.report_type === "walls" ? r.total_length_m.toFixed(2) + " מ'" : r.total_area_m2.toFixed(2) + " מ\"ר"}</td>
      <td>${r.note || "—"}</td>
    </tr>`).join("");

  const html = `<!doctype html><html dir="rtl" lang="he"><head>
  <meta charset="UTF-8"><title>דוח פרויקט - ${planName}</title>
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
  <h1>📊 דוח פרויקט: ${planName}</h1>
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

// ─── Mini bar chart ───────────────────────────────────────────────────────────
const MiniBarChart: React.FC<{ data: { date: string; value: number; max: number }[] }> = ({ data }) => (
  <div className="space-y-1">
    {data.map((d) => (
      <div key={d.date} className="flex items-center gap-2 text-xs">
        <span className="w-24 text-slate-500 shrink-0 text-left" dir="ltr">{d.date}</span>
        <div className="flex-1 h-4 bg-slate-100 rounded overflow-hidden">
          <div className="h-full bg-[#FF4B4B] rounded" style={{ width: `${Math.min(100, (d.value / Math.max(d.max, 0.001)) * 100)}%` }} />
        </div>
        <span className="w-16 text-right font-semibold">{d.value.toFixed(2)} מ'</span>
      </div>
    ))}
  </div>
);

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
  const progressColor = pct < 30 ? "bg-red-500" : pct < 70 ? "bg-amber-500" : "bg-emerald-500";
  const snapshotUrl = getWorkerReportSnapshotUrl(selectedPlanId);
  const selectedPlan = plans.find((p) => p.id === selectedPlanId);

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
            <button
              type="button"
              onClick={() => printBoqReport(dashboard, summary, selectedPlan?.plan_name ?? "פרויקט", snapshotUrl)}
              className="bg-white border border-[#FF4B4B] text-[#FF4B4B] px-3 py-2 rounded text-sm font-semibold hover:bg-red-50"
            >
              🖨️ הדפס דוח מלא
            </button>
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
            className={`px-5 py-3 text-sm border-b-[3px] -mb-px ${activeView === t.id ? "border-[#FF4B4B] text-[#FF4B4B] font-semibold" : "border-transparent text-slate-600"}`}
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
          <section className="grid grid-cols-2 xl:grid-cols-4 gap-3">
            {[
              { label: "סך מתוכנן", value: `${dashboard.total_planned_m.toFixed(1)} מ'` },
              { label: "בוצע בפועל", value: `${dashboard.built_m.toFixed(1)} מ'`, sub: `${pct.toFixed(1)}%` },
              { label: "נותר", value: `${dashboard.remaining_m.toFixed(1)} מ'`, sub: dashboard.days_to_finish != null ? `~${dashboard.days_to_finish.toFixed(0)} ימים` : "" },
              { label: "עלות מצטברת", value: `${dashboard.current_cost_ils.toLocaleString()} ₪` },
            ].map((c) => (
              <div key={c.label} className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
                <div className="text-xs text-slate-500">{c.label}</div>
                <div className="text-lg font-semibold">{c.value}</div>
                {c.sub && <div className="text-xs text-slate-400">{c.sub}</div>}
              </div>
            ))}
          </section>

          <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
            <h3 className="text-sm font-semibold mb-2">התקדמות כללית</h3>
            <div className="w-full h-7 bg-slate-100 rounded-lg overflow-hidden">
              <div className={`h-full ${progressColor} text-white text-sm font-semibold flex items-center justify-center`} style={{ width: `${Math.min(100, pct)}%` }}>
                {pct > 8 ? `${pct.toFixed(1)}%` : ""}
              </div>
            </div>
            <div className="text-xs text-slate-500 mt-1">קצב ממוצע: {dashboard.average_daily_m.toFixed(2)} מ'/יום</div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">📈 גרף יומי</h3>
              {dashboard.timeline.length === 0
                ? <p className="text-sm text-slate-400">אין דיווחים.</p>
                : <MiniBarChart data={dashboard.timeline.map((p) => ({ date: p.date, value: p.quantity_m, max: dashboard.max_daily_m ?? 1 }))} />
              }
            </div>
            <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
              <h3 className="text-sm font-semibold mb-3">🕒 דיווחים אחרונים</h3>
              {dashboard.recent_reports.length === 0
                ? <p className="text-sm text-slate-400">אין דיווחים.</p>
                : dashboard.recent_reports.map((r) => (
                  <div key={r.id} className="flex justify-between items-center text-xs bg-slate-50 rounded p-2 mb-1">
                    <span>{r.date} | {r.shift} | {r.report_type === "walls" ? "קירות" : "ריצוף"}</span>
                    <span className="font-bold text-[#FF4B4B]">
                      {r.report_type === "walls" ? `${r.total_length_m.toFixed(2)} מ'` : `${r.total_area_m2.toFixed(2)} מ"ר`}
                    </span>
                  </div>
                ))
              }
            </div>
          </section>
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
              <table className="w-full text-sm min-w-[560px]">
                <thead>
                  <tr className="text-slate-500 border-b">
                    <th className="p-2 text-right">קטגוריה</th>
                    <th className="p-2 text-right">מתוכנן</th>
                    <th className="p-2 text-right">בוצע</th>
                    <th className="p-2 text-right">נותר</th>
                    <th className="p-2 w-40">אחוז</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboard.boq_progress.map((row) => (
                    <tr key={row.label} className="border-b last:border-0 hover:bg-slate-50">
                      <td className="p-2 font-semibold">{row.label}</td>
                      <td className="p-2">{row.planned_qty.toFixed(2)} {row.unit}</td>
                      <td className="p-2">{row.built_qty.toFixed(2)} {row.unit}</td>
                      <td className="p-2">{row.remaining_qty.toFixed(2)} {row.unit}</td>
                      <td className="p-2">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-slate-100 rounded overflow-hidden">
                            <div className="h-full bg-[#FF4B4B]" style={{ width: `${Math.min(100, row.progress_percent)}%` }} />
                          </div>
                          <span className="text-xs w-10 text-right">{row.progress_percent.toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
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
