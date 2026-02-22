import React from "react";
import { listDatabasePlans, type PlanSummary } from "../api/managerWorkshopApi";
import {
  calculateInvoice,
  getInvoiceWorkTypes,
  type InvoiceCalculationResponse
} from "../api/managerInsightsApi";

type RangePreset = "week" | "month" | "custom";

function formatDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export const InvoicesPage: React.FC = () => {
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedPlanId, setSelectedPlanId] = React.useState("");
  const [rangePreset, setRangePreset] = React.useState<RangePreset>("month");
  const [startDate, setStartDate] = React.useState(formatDate(new Date(Date.now() - 30 * 24 * 3600 * 1000)));
  const [endDate, setEndDate] = React.useState(formatDate(new Date()));
  const [workTypes, setWorkTypes] = React.useState<string[]>([]);
  const [unitPrices, setUnitPrices] = React.useState<Record<string, number>>({});
  const [contractorName, setContractorName] = React.useState("");
  const [contractorCompany, setContractorCompany] = React.useState("");
  const [contractorVat, setContractorVat] = React.useState("");
  const [contractorAddress, setContractorAddress] = React.useState("");
  const [invoice, setInvoice] = React.useState<InvoiceCalculationResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    void (async () => {
      try {
        const data = await listDatabasePlans();
        setPlans(data);
        if (data.length > 0) setSelectedPlanId(data[0].id);
      } catch (e) {
        console.error(e);
        setError("שגיאה בטעינת תוכניות.");
      }
    })();
  }, []);

  React.useEffect(() => {
    if (!selectedPlanId) return;
    void (async () => {
      try {
        const types = await getInvoiceWorkTypes(selectedPlanId);
        setWorkTypes(types);
        const defaults: Record<string, number> = {};
        for (const type of types) {
          if (type.includes("ריצוף") || type.includes("חיפוי")) defaults[type] = 250;
          else if (type.includes("בטון")) defaults[type] = 1200;
          else if (type.includes("בלוק")) defaults[type] = 600;
          else defaults[type] = 800;
        }
        setUnitPrices(defaults);
      } catch (e) {
        console.error(e);
      }
    })();
  }, [selectedPlanId]);

  React.useEffect(() => {
    const now = new Date();
    if (rangePreset === "week") {
      setEndDate(formatDate(now));
      setStartDate(formatDate(new Date(now.getTime() - 7 * 24 * 3600 * 1000)));
    } else if (rangePreset === "month") {
      setEndDate(formatDate(now));
      setStartDate(formatDate(new Date(now.getTime() - 30 * 24 * 3600 * 1000)));
    }
  }, [rangePreset]);

  const createInvoice = async () => {
    if (!selectedPlanId) return;
    if (!contractorName || !contractorVat) {
      setError("יש למלא שם קבלן ומספר עוסק.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      const data = await calculateInvoice(selectedPlanId, {
        start_date: startDate,
        end_date: endDate,
        unit_prices: unitPrices,
        contractor: {
          name: contractorName,
          company: contractorCompany,
          vat_id: contractorVat,
          address: contractorAddress
        }
      });
      setInvoice(data);
    } catch (e) {
      console.error(e);
      setError("שגיאה ביצירת החשבונית.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm">
        <h2 className="text-lg font-semibold text-[#31333F]">💰 מחולל חשבונות חלקיים</h2>
        <p className="text-xs text-slate-500 mt-1">הפקת חשבון לפי ביצוע בפועל בטווח תאריכים נבחר.</p>
      </div>

      {error && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">{error}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-[1fr,320px] gap-5">
        <main className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm space-y-4">
          <label className="text-xs block">
            פרויקט
            <select
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
              value={selectedPlanId}
              onChange={(e) => setSelectedPlanId(e.target.value)}
            >
              {plans.length === 0 && <option value="">אין תוכניות</option>}
              {plans.map((plan) => (
                <option key={plan.id} value={plan.id}>
                  {plan.plan_name}
                </option>
              ))}
            </select>
          </label>

          <div className="space-y-2">
            <div className="text-sm font-semibold">📅 טווח תאריכים</div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <button
                type="button"
                onClick={() => setRangePreset("week")}
                className={`rounded border px-2 py-2 ${rangePreset === "week" ? "border-[#FF4B4B] text-[#FF4B4B]" : "border-slate-300"}`}
              >
                שבוע אחרון
              </button>
              <button
                type="button"
                onClick={() => setRangePreset("month")}
                className={`rounded border px-2 py-2 ${rangePreset === "month" ? "border-[#FF4B4B] text-[#FF4B4B]" : "border-slate-300"}`}
              >
                חודש אחרון
              </button>
              <button
                type="button"
                onClick={() => setRangePreset("custom")}
                className={`rounded border px-2 py-2 ${rangePreset === "custom" ? "border-[#FF4B4B] text-[#FF4B4B]" : "border-slate-300"}`}
              >
                מותאם
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <label className="block">
                מתאריך
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
                />
              </label>
              <label className="block">
                עד תאריך
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
                />
              </label>
            </div>
          </div>

          <div>
            <div className="text-sm font-semibold mb-2">💵 מחירי יחידה</div>
            <div className="space-y-2">
              {workTypes.length === 0 ? (
                <div className="text-sm text-slate-500">אין סוגי עבודה זמינים.</div>
              ) : (
                workTypes.map((workType) => (
                  <label key={workType} className="grid grid-cols-[1fr,120px] gap-2 items-center text-xs">
                    <span>{workType}</span>
                    <input
                      type="number"
                      value={unitPrices[workType] ?? 0}
                      onChange={(e) =>
                        setUnitPrices((prev) => ({ ...prev, [workType]: Number(e.target.value) }))
                      }
                      className="bg-white border border-slate-300 rounded px-2 py-2"
                    />
                  </label>
                ))
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={() => void createInvoice()}
            disabled={loading}
            className="w-full bg-[#FF4B4B] text-white rounded py-2 text-sm font-semibold disabled:opacity-40"
          >
            {loading ? "מייצר חשבונית..." : "🧾 צור חשבונית"}
          </button>
        </main>

        <aside className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm space-y-3">
          <h3 className="text-sm font-semibold">👷 פרטי קבלן</h3>
          <label className="text-xs block">
            שם הקבלן
            <input
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
              value={contractorName}
              onChange={(e) => setContractorName(e.target.value)}
            />
          </label>
          <label className="text-xs block">
            שם חברה
            <input
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
              value={contractorCompany}
              onChange={(e) => setContractorCompany(e.target.value)}
            />
          </label>
          <label className="text-xs block">
            ח.פ / ע.מ
            <input
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2"
              value={contractorVat}
              onChange={(e) => setContractorVat(e.target.value)}
            />
          </label>
          <label className="text-xs block">
            כתובת
            <textarea
              className="mt-1 w-full bg-white border border-slate-300 rounded px-2 py-2 min-h-[90px]"
              value={contractorAddress}
              onChange={(e) => setContractorAddress(e.target.value)}
            />
          </label>
        </aside>
      </div>

      {invoice && (
        <section className="bg-white border border-[#E6E6EA] rounded-lg p-4 shadow-sm space-y-4">
          <h3 className="text-sm font-semibold">📋 סיכום חשבונית</h3>
          {invoice.items.length === 0 ? (
            <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
              אין דיווחים בטווח התאריכים שנבחר.
            </div>
          ) : (
            <>
              <div className="overflow-auto">
                <table className="w-full text-sm min-w-[640px]">
                  <thead>
                    <tr className="text-slate-500 border-b">
                      <th className="text-right p-2">סוג עבודה</th>
                      <th className="text-right p-2">כמות</th>
                      <th className="text-right p-2">יחידה</th>
                      <th className="text-right p-2">מחיר יחידה</th>
                      <th className="text-right p-2">סה"כ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {invoice.items.map((item) => (
                      <tr key={item.work_type} className="border-b last:border-b-0">
                        <td className="p-2">{item.work_type}</td>
                        <td className="p-2">{item.quantity.toFixed(2)}</td>
                        <td className="p-2">{item.unit}</td>
                        <td className="p-2">{item.unit_price.toLocaleString()} ₪</td>
                        <td className="p-2 font-semibold">{item.subtotal.toLocaleString()} ₪</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="bg-slate-50 rounded p-3 text-sm">
                  <div className="text-slate-500 text-xs">סכום ביניים</div>
                  <div className="font-semibold">{invoice.total_amount.toLocaleString()} ₪</div>
                </div>
                <div className="bg-slate-50 rounded p-3 text-sm">
                  <div className="text-slate-500 text-xs">מע"מ (17%)</div>
                  <div className="font-semibold">{invoice.vat.toLocaleString()} ₪</div>
                </div>
                <div className="bg-slate-50 rounded p-3 text-sm">
                  <div className="text-slate-500 text-xs">סה"כ לתשלום</div>
                  <div className="font-semibold">{invoice.total_with_vat.toLocaleString()} ₪</div>
                </div>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
};
