import React from "react";
import { listWorkshopPlans, type PlanSummary } from "../api/managerWorkshopApi";
import { apiClient } from "../api/client";

// ──────────────────────────────────────────────────────────────────────────────
// Types & Config
// ──────────────────────────────────────────────────────────────────────────────
interface WorkCategory {
  key: string; label: string; color: string; unit: "m" | "m2"; icon: string;
}
const WORK_CATEGORIES: WorkCategory[] = [
  { key: "concrete_wall",  label: "קיר בטון",    color: "#0ea5e9", unit: "m",  icon: "🏗️" },
  { key: "block_wall",     label: "קיר בלוקים", color: "#2563eb", unit: "m",  icon: "🧱" },
  { key: "gypsum_wall",    label: "קיר גבס",    color: "#7c3aed", unit: "m",  icon: "📏" },
  { key: "partition",      label: "מחיצה קלה",  color: "#8b5cf6", unit: "m",  icon: "▫️" },
  { key: "flooring",       label: "ריצוף",       color: "#f97316", unit: "m2", icon: "⬛" },
  { key: "ceiling",        label: "תקרה",        color: "#14b8a6", unit: "m2", icon: "⬆️" },
  { key: "opening_door",   label: "דלת",         color: "#22c55e", unit: "m",  icon: "🚪" },
  { key: "opening_window", label: "חלון",        color: "#84cc16", unit: "m",  icon: "🪟" },
];

function genUid(): string { return Math.random().toString(36).slice(2, 10); }

// ──────────────────────────────────────────────────────────────────────────────
// Room-based types
// ──────────────────────────────────────────────────────────────────────────────
interface RoomAssignment {
  uid: string;
  categoryKey: string;
  quantity: number;
  unit: "m" | "m2";
  note: string;
}
interface Room {
  id: string;
  name: string;
  area_m2: number;
  assignments: RoomAssignment[];
}

// ──────────────────────────────────────────────────────────────────────────────
// Paint-canvas types
// ──────────────────────────────────────────────────────────────────────────────
type DrawMode = "line" | "rect" | "freehand";
interface PaintMark {
  uid: string;
  categoryKey: string;
  roomId: string | null;
  type: DrawMode;
  points: [number, number][];
  measurement: number;
  unit: "m" | "m2";
}

function measureLine(pts: [number, number][], ppm: number): number {
  let t = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i-1][0], dy = pts[i][1] - pts[i-1][1];
    t += Math.sqrt(dx*dx + dy*dy);
  }
  return t / ppm;
}
function measureRect(pts: [number, number][], ppm: number): number {
  if (pts.length < 2) return 0;
  return Math.abs(pts[1][0]-pts[0][0]) * Math.abs(pts[1][1]-pts[0][1]) / (ppm*ppm);
}

// ──────────────────────────────────────────────────────────────────────────────
// RoomPanel — שכבה 3: Room-based assignment
// ──────────────────────────────────────────────────────────────────────────────
interface RoomPanelProps {
  rooms: Room[];
  onRoomsChange: (rooms: Room[]) => void;
}
const RoomPanel: React.FC<RoomPanelProps> = ({ rooms, onRoomsChange }) => {
  const [selectedRoom, setSelectedRoom] = React.useState<string | null>(null);
  const [newRoomName, setNewRoomName] = React.useState("");
  const [newRoomArea, setNewRoomArea] = React.useState("");

  const room = rooms.find(r => r.id === selectedRoom) ?? null;

  const addRoom = () => {
    if (!newRoomName.trim()) return;
    const r: Room = { id: genUid(), name: newRoomName.trim(), area_m2: parseFloat(newRoomArea)||0, assignments: [] };
    onRoomsChange([...rooms, r]);
    setSelectedRoom(r.id);
    setNewRoomName(""); setNewRoomArea("");
  };

  const updateRoom = (updated: Room) => onRoomsChange(rooms.map(r => r.id === updated.id ? updated : r));

  const addAssignment = (categoryKey: string) => {
    if (!room) return;
    const cat = WORK_CATEGORIES.find(c => c.key === categoryKey)!;
    const a: RoomAssignment = { uid: genUid(), categoryKey, quantity: 0, unit: cat.unit, note: "" };
    updateRoom({ ...room, assignments: [...room.assignments, a] });
  };

  const updateAssignment = (assignUid: string, patch: Partial<RoomAssignment>) => {
    if (!room) return;
    updateRoom({ ...room, assignments: room.assignments.map(a => a.uid === assignUid ? { ...a, ...patch } : a) });
  };

  const removeAssignment = (assignUid: string) => {
    if (!room) return;
    updateRoom({ ...room, assignments: room.assignments.filter(a => a.uid !== assignUid) });
  };

  const removeRoom = (id: string) => {
    onRoomsChange(rooms.filter(r => r.id !== id));
    if (selectedRoom === id) setSelectedRoom(null);
  };

  // BOQ summary across all rooms
  const boqByCategory: Record<string, { label: string; unit: string; total: number; color: string }> = {};
  for (const r of rooms) {
    for (const a of r.assignments) {
      const cat = WORK_CATEGORIES.find(c => c.key === a.categoryKey)!;
      if (!boqByCategory[a.categoryKey]) boqByCategory[a.categoryKey] = { label: cat.label, unit: cat.unit, total: 0, color: cat.color };
      boqByCategory[a.categoryKey].total += a.quantity;
    }
  }

  const printBoq = () => {
    const rows = Object.entries(boqByCategory).map(([, v]) =>
      `<tr><td>${v.label}</td><td style="text-align:center">${v.total.toFixed(2)} ${v.unit}</td></tr>`
    ).join("");
    const roomRows = rooms.map(r => {
      const aRows = r.assignments.map(a => {
        const cat = WORK_CATEGORIES.find(c => c.key === a.categoryKey)!;
        return `<tr><td style="padding-right:24px">${cat.icon} ${cat.label}</td><td style="text-align:center">${a.quantity.toFixed(2)} ${a.unit}</td><td>${a.note}</td></tr>`;
      }).join("");
      return `<tr style="background:#f0f4ff"><td colspan="3"><strong>${r.name} — ${r.area_m2} מ"ר</strong></td></tr>${aRows || "<tr><td colspan='3' style='color:#aaa'>אין פריטים</td></tr>"}`;
    }).join("");
    const w = window.open("", "_blank");
    w?.document.write(`<html dir="rtl"><head><title>כתב כמויות</title><style>
      body{font-family:Arial;padding:24px} h2{color:#31333F}
      table{width:100%;border-collapse:collapse;margin-bottom:24px}
      th,td{border:1px solid #ddd;padding:8px;text-align:right}
      th{background:#31333F;color:white}
    </style></head><body>
      <h2>📋 כתב כמויות — לפי חדרים</h2>
      <h3>סיכום כולל</h3>
      <table><thead><tr><th>קטגוריה</th><th>סה"כ</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <h3>פירוט לפי חדרים</h3>
      <table><thead><tr><th>קטגוריה</th><th>כמות</th><th>הערה</th></tr></thead>
      <tbody>${roomRows}</tbody></table>
      <script>window.print()</script>
    </body></html>`);
  };

  return (
    <div className="flex gap-4" style={{ minHeight: 520 }}>
      {/* Room list sidebar */}
      <div className="w-56 flex-shrink-0 flex flex-col gap-2">
        <div className="bg-white border border-[#E6E6EA] rounded-lg p-3">
          <h3 className="font-semibold text-xs mb-2 text-[#31333F]">➕ הוסף חדר</h3>
          <input
            className="w-full border border-[#E6E6EA] rounded px-2 py-1 text-sm mb-1"
            placeholder="שם החדר (כיתה / שירותים...)"
            value={newRoomName}
            onChange={e => setNewRoomName(e.target.value)}
            onKeyDown={e => e.key === "Enter" && addRoom()}
          />
          <input
            className="w-full border border-[#E6E6EA] rounded px-2 py-1 text-sm mb-2"
            placeholder='שטח (מ"ר)'
            type="number"
            value={newRoomArea}
            onChange={e => setNewRoomArea(e.target.value)}
          />
          <button onClick={addRoom} className="w-full bg-[#FF4B4B] text-white rounded py-1.5 text-sm font-medium hover:bg-red-600">
            הוסף חדר
          </button>
        </div>

        <div className="flex-1 overflow-y-auto flex flex-col gap-1">
          {rooms.length === 0 && (
            <p className="text-xs text-slate-400 text-center mt-4 px-2">
              הוסף חדרים מהטופס למעלה,<br />או לחץ על "טען חדרים" מהכרטיסייה
            </p>
          )}
          {rooms.map(r => (
            <div
              key={r.id}
              className={`p-2 rounded-lg border cursor-pointer text-sm transition-colors ${
                selectedRoom === r.id
                  ? "bg-blue-50 border-blue-400"
                  : "bg-white border-[#E6E6EA] hover:bg-slate-50"
              }`}
              onClick={() => setSelectedRoom(r.id)}
            >
              <div className="flex justify-between items-start">
                <span className="font-medium text-sm">{r.name}</span>
                <button
                  onClick={e => { e.stopPropagation(); removeRoom(r.id); }}
                  className="text-red-400 text-xs hover:text-red-600 ml-1"
                >✕</button>
              </div>
              <div className="text-xs text-slate-500">{r.area_m2} מ"ר · {r.assignments.length} סוגי עבודה</div>
            </div>
          ))}
        </div>
      </div>

      {/* Room detail */}
      <div className="flex-1 flex flex-col gap-3">
        {room ? (
          <div className="bg-white border border-[#E6E6EA] rounded-lg p-4 flex-1">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="font-semibold text-base text-[#31333F]">{room.name}</h3>
                <p className="text-sm text-slate-500">שטח: {room.area_m2} מ"ר</p>
              </div>
              <input
                type="number"
                className="border rounded px-2 py-1 text-sm w-24"
                value={room.area_m2}
                onChange={e => updateRoom({ ...room, area_m2: parseFloat(e.target.value)||0 })}
                placeholder='שטח מ"ר'
              />
            </div>

            <p className="text-xs text-slate-500 mb-2 font-medium">בחר סוג עבודה להוספה לחדר:</p>
            <div className="flex flex-wrap gap-1.5 mb-4">
              {WORK_CATEGORIES.map(cat => (
                <button
                  key={cat.key}
                  onClick={() => addAssignment(cat.key)}
                  className="px-2.5 py-1.5 rounded-lg text-xs text-white font-medium hover:opacity-80 transition-opacity"
                  style={{ background: cat.color }}
                >
                  {cat.icon} {cat.label}
                </button>
              ))}
            </div>

            {room.assignments.length === 0 ? (
              <div className="border-2 border-dashed border-slate-200 rounded-lg p-6 text-center text-slate-400 text-sm">
                לחץ על סוג עבודה למעלה כדי להוסיף לחדר זה
              </div>
            ) : (
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="border border-slate-200 p-2 text-right text-xs">קטגוריה</th>
                    <th className="border border-slate-200 p-2 text-right text-xs">כמות</th>
                    <th className="border border-slate-200 p-2 text-right text-xs">הערה</th>
                    <th className="border border-slate-200 p-2 w-8"></th>
                  </tr>
                </thead>
                <tbody>
                  {room.assignments.map(a => {
                    const cat = WORK_CATEGORIES.find(c => c.key === a.categoryKey)!;
                    return (
                      <tr key={a.uid} className="hover:bg-slate-50">
                        <td className="border border-slate-200 p-2">
                          <span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 align-middle" style={{ background: cat.color }}></span>
                          {cat.icon} {cat.label}
                        </td>
                        <td className="border border-slate-200 p-2">
                          <div className="flex items-center gap-1">
                            <input
                              type="number" min="0" step="0.1"
                              className="w-20 border border-slate-200 rounded px-1.5 py-0.5 text-xs"
                              value={a.quantity || ""}
                              placeholder="0"
                              onChange={e => updateAssignment(a.uid, { quantity: parseFloat(e.target.value) || 0 })}
                            />
                            <span className="text-xs text-slate-500">{cat.unit}</span>
                          </div>
                        </td>
                        <td className="border border-slate-200 p-2">
                          <input
                            className="w-full border border-slate-200 rounded px-1.5 py-0.5 text-xs"
                            placeholder="הערה..."
                            value={a.note}
                            onChange={e => updateAssignment(a.uid, { note: e.target.value })}
                          />
                        </td>
                        <td className="border border-slate-200 p-2 text-center">
                          <button onClick={() => removeAssignment(a.uid)} className="text-red-400 hover:text-red-600 text-xs">✕</button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center border-2 border-dashed border-slate-200 rounded-lg">
            <div className="text-center text-slate-400">
              <p className="text-3xl mb-2">🏠</p>
              <p className="text-sm">בחר חדר מהרשימה משמאל<br />או הוסף חדר חדש</p>
            </div>
          </div>
        )}

        {/* BOQ summary */}
        {Object.keys(boqByCategory).length > 0 && (
          <div className="bg-white border border-[#E6E6EA] rounded-lg p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-semibold text-sm text-[#31333F]">📋 כתב כמויות מצטבר</h3>
              <button onClick={printBoq} className="px-3 py-1.5 bg-[#31333F] text-white text-xs rounded-lg hover:bg-slate-700">
                🖨️ הדפס BOQ
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(boqByCategory).map(([key, v]) => (
                <div key={key} className="px-3 py-2 rounded-lg text-white text-xs font-medium" style={{ background: v.color }}>
                  {v.label}: <strong>{v.total.toFixed(2)} {v.unit}</strong>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ──────────────────────────────────────────────────────────────────────────────
// PaintCanvas — שכבה 2: ציור חופשי על גבי תוכנית
// ──────────────────────────────────────────────────────────────────────────────
interface PaintCanvasProps {
  planId: string;
  pxPerMeter: number;
  marks: PaintMark[];
  onMarksChange: (marks: PaintMark[]) => void;
  rooms: Room[];
}
const PaintCanvas: React.FC<PaintCanvasProps> = ({ planId, pxPerMeter, marks, onMarksChange, rooms }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [imgNatW, setImgNatW] = React.useState(1);
  const [imgNatH, setImgNatH] = React.useState(1);
  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const isPanning = React.useRef(false);
  const panStart = React.useRef({ mx: 0, my: 0, px: 0, py: 0 });
  const [drawMode, setDrawMode] = React.useState<DrawMode>("line");
  const [activeCat, setActiveCat] = React.useState<string>(WORK_CATEGORIES[0].key);
  const [activeRoom, setActiveRoom] = React.useState<string | null>(null);
  const isDrawing = React.useRef(false);
  const [curPts, setCurPts] = React.useState<[number, number][]>([]);
  const [visibleCats, setVisibleCats] = React.useState<Set<string>>(
    new Set(WORK_CATEGORIES.map(c => c.key))
  );

  const imgUrl = `${apiClient.defaults.baseURL}/manager/workshop/plans/${encodeURIComponent(planId)}/image`;

  const toImg = (clientX: number, clientY: number): [number, number] => {
    const rect = containerRef.current!.getBoundingClientRect();
    return [
      (clientX - rect.left - pan.x) / zoom,
      (clientY - rect.top - pan.y) / zoom,
    ];
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    const rect = containerRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const newZoom = Math.max(0.3, Math.min(10, zoom * factor));
    setPan(p => ({
      x: mx - (mx - p.x) * (newZoom / zoom),
      y: my - (my - p.y) * (newZoom / zoom),
    }));
    setZoom(newZoom);
  };

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.altKey) {
      isPanning.current = true;
      panStart.current = { mx: e.clientX, my: e.clientY, px: pan.x, py: pan.y };
      return;
    }
    isDrawing.current = true;
    setCurPts([toImg(e.clientX, e.clientY)]);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (isPanning.current) {
      setPan({
        x: panStart.current.px + e.clientX - panStart.current.mx,
        y: panStart.current.py + e.clientY - panStart.current.my,
      });
      return;
    }
    if (!isDrawing.current) return;
    const pt = toImg(e.clientX, e.clientY);
    if (drawMode === "freehand") setCurPts(ps => [...ps, pt]);
    else setCurPts(ps => [ps[0], pt]);
  };

  const onMouseUp = () => {
    isPanning.current = false;
    if (!isDrawing.current || curPts.length < 2) {
      isDrawing.current = false;
      setCurPts([]);
      return;
    }
    isDrawing.current = false;
    const cat = WORK_CATEGORIES.find(c => c.key === activeCat)!;
    const m = drawMode === "rect" ? measureRect(curPts, pxPerMeter) : measureLine(curPts, pxPerMeter);
    const mark: PaintMark = {
      uid: genUid(), categoryKey: activeCat, roomId: activeRoom,
      type: drawMode, points: curPts, measurement: m, unit: cat.unit,
    };
    onMarksChange([...marks, mark]);
    setCurPts([]);
  };

  const displayW = imgNatW * zoom;
  const displayH = imgNatH * zoom;

  const printSummary = () => {
    const bycat: Record<string, { label: string; unit: string; total: number; color: string }> = {};
    for (const m of marks) {
      const cat = WORK_CATEGORIES.find(c => c.key === m.categoryKey)!;
      if (!bycat[m.categoryKey]) bycat[m.categoryKey] = { label: cat.label, unit: cat.unit, total: 0, color: cat.color };
      bycat[m.categoryKey].total += m.measurement;
    }
    const rows = Object.entries(bycat).map(([, v]) =>
      `<tr><td>${v.label}</td><td>${v.total.toFixed(2)} ${v.unit}</td></tr>`
    ).join("");
    const w = window.open("", "_blank");
    w?.document.write(`<html dir="rtl"><head><title>BOQ ציור</title><style>
      body{font-family:Arial;padding:24px}table{width:100%;border-collapse:collapse}
      th,td{border:1px solid #ddd;padding:8px;text-align:right}
      th{background:#31333F;color:white}
    </style></head><body>
      <h2>📋 כתב כמויות — שכבת ציור</h2>
      <table><thead><tr><th>קטגוריה</th><th>סה"כ</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <script>window.print()</script>
    </body></html>`);
  };

  return (
    <div className="flex gap-3" style={{ height: 600 }}>
      {/* Left controls */}
      <div className="w-52 flex-shrink-0 flex flex-col gap-2 overflow-y-auto">
        {/* Draw mode */}
        <div className="bg-white border border-[#E6E6EA] rounded-lg p-3">
          <h4 className="font-semibold text-xs mb-2 text-[#31333F]">📐 מצב ציור</h4>
          {(["line", "rect", "freehand"] as DrawMode[]).map(m => (
            <button key={m} onClick={() => setDrawMode(m)}
              className={`w-full text-right text-xs py-1.5 px-2 rounded mb-1 transition-colors ${
                drawMode === m ? "bg-[#FF4B4B] text-white" : "bg-slate-100 hover:bg-slate-200"
              }`}>
              {m === "line" ? "📏 קו" : m === "rect" ? "▭ מלבן" : "✏️ חופשי"}
            </button>
          ))}
        </div>

        {/* Category selector */}
        <div className="bg-white border border-[#E6E6EA] rounded-lg p-3">
          <h4 className="font-semibold text-xs mb-2 text-[#31333F]">🏗️ קטגוריה פעילה</h4>
          {WORK_CATEGORIES.map(cat => (
            <button key={cat.key} onClick={() => setActiveCat(cat.key)}
              className="w-full text-right text-xs py-1 px-2 rounded mb-1 flex items-center gap-1.5 border transition-all"
              style={{
                background: activeCat === cat.key ? cat.color + "22" : "#f8f8f8",
                borderColor: activeCat === cat.key ? cat.color : "transparent",
                borderWidth: activeCat === cat.key ? 2 : 1,
                fontWeight: activeCat === cat.key ? 700 : 400,
              }}>
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: cat.color }}></span>
              {cat.icon} {cat.label}
            </button>
          ))}
        </div>

        {/* Room association */}
        {rooms.length > 0 && (
          <div className="bg-white border border-[#E6E6EA] rounded-lg p-3">
            <h4 className="font-semibold text-xs mb-2 text-[#31333F]">🏠 שייך לחדר</h4>
            <button onClick={() => setActiveRoom(null)}
              className={`w-full text-right text-xs py-1 px-2 rounded mb-1 ${!activeRoom ? "bg-blue-100 font-bold" : "bg-slate-100 hover:bg-slate-200"}`}>
              ללא שיוך
            </button>
            {rooms.map(r => (
              <button key={r.id} onClick={() => setActiveRoom(r.id)}
                className={`w-full text-right text-xs py-1 px-2 rounded mb-1 ${activeRoom === r.id ? "bg-blue-100 font-bold" : "bg-slate-100 hover:bg-slate-200"}`}>
                {r.name}
              </button>
            ))}
          </div>
        )}

        {/* Visibility toggles */}
        <div className="bg-white border border-[#E6E6EA] rounded-lg p-3">
          <h4 className="font-semibold text-xs mb-2 text-[#31333F]">👁️ הצג שכבות</h4>
          {WORK_CATEGORIES.map(cat => (
            <label key={cat.key} className="flex items-center gap-1.5 text-xs mb-1 cursor-pointer">
              <input type="checkbox" checked={visibleCats.has(cat.key)}
                onChange={() => setVisibleCats(prev => {
                  const s = new Set(prev);
                  if (s.has(cat.key)) s.delete(cat.key); else s.add(cat.key);
                  return s;
                })} />
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: cat.color }}></span>
              {cat.label}
            </label>
          ))}
        </div>

        <button onClick={printSummary} className="w-full bg-[#31333F] text-white text-xs rounded-lg py-2 hover:bg-slate-700">
          🖨️ הדפס BOQ
        </button>
        <p className="text-xs text-slate-400 text-center">
          💡 Alt+גרור = הזזה<br />Scroll = זום
        </p>
      </div>

      {/* Canvas area */}
      <div className="flex-1 flex flex-col gap-2">
        <div
          ref={containerRef}
          className="flex-1 overflow-hidden rounded-lg border border-[#E6E6EA] bg-slate-100 relative select-none"
          style={{ cursor: "crosshair" }}
          onWheel={onWheel}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          <div style={{ transform: `translate(${pan.x}px,${pan.y}px)`, width: displayW, height: displayH, position: "absolute" }}>
            <img
              src={imgUrl} alt="plan" draggable={false}
              onLoad={e => { const i = e.currentTarget; setImgNatW(i.naturalWidth); setImgNatH(i.naturalHeight); }}
              style={{ width: displayW, height: displayH, display: "block", userSelect: "none" }}
            />
            <svg style={{ position: "absolute", top: 0, left: 0, width: displayW, height: displayH, overflow: "visible" }}>
              {/* Saved marks */}
              {marks.filter(m => visibleCats.has(m.categoryKey)).map(m => {
                const cat = WORK_CATEGORIES.find(c => c.key === m.categoryKey)!;
                const pts = m.points.map(([x, y]) => [x * zoom, y * zoom] as [number, number]);
                const mx = pts.length >= 1 ? (pts[0][0] + pts[pts.length-1][0]) / 2 : 0;
                const my2 = pts.length >= 1 ? (pts[0][1] + pts[pts.length-1][1]) / 2 : 0;
                return (
                  <g key={m.uid}>
                    {m.type === "rect" && pts.length >= 2 ? (
                      <rect
                        x={Math.min(pts[0][0], pts[1][0])} y={Math.min(pts[0][1], pts[1][1])}
                        width={Math.abs(pts[1][0] - pts[0][0])} height={Math.abs(pts[1][1] - pts[0][1])}
                        fill={cat.color + "33"} stroke={cat.color} strokeWidth={2 / zoom}
                      />
                    ) : (
                      <polyline
                        points={pts.map(p => p.join(",")).join(" ")}
                        fill="none" stroke={cat.color} strokeWidth={3 / zoom} strokeLinecap="round"
                      />
                    )}
                    <text x={mx} y={my2 - 6 / zoom} fill={cat.color} fontSize={13 / zoom} textAnchor="middle">
                      {m.measurement.toFixed(2)}{m.unit}
                    </text>
                  </g>
                );
              })}
              {/* Current drawing preview */}
              {curPts.length >= 2 && (() => {
                const cat = WORK_CATEGORIES.find(c => c.key === activeCat)!;
                const pts = curPts.map(([x, y]) => [x * zoom, y * zoom] as [number, number]);
                return drawMode === "rect" ? (
                  <rect
                    x={Math.min(pts[0][0], pts[1][0])} y={Math.min(pts[0][1], pts[1][1])}
                    width={Math.abs(pts[1][0] - pts[0][0])} height={Math.abs(pts[1][1] - pts[0][1])}
                    fill={cat.color + "22"} stroke={cat.color} strokeWidth={2 / zoom} strokeDasharray="6,3"
                  />
                ) : (
                  <polyline
                    points={pts.map(p => p.join(",")).join(" ")}
                    fill="none" stroke={cat.color} strokeWidth={3 / zoom} strokeDasharray="6,3"
                  />
                );
              })()}
            </svg>
          </div>

          {/* Zoom controls */}
          <div className="absolute top-2 left-2 flex gap-1">
            <button onClick={() => setZoom(z => Math.min(10, z * 1.2))} className="bg-white border rounded px-2 py-1 text-xs shadow hover:bg-slate-50">+</button>
            <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} className="bg-white border rounded px-2 py-1 text-xs shadow hover:bg-slate-50">↺</button>
            <button onClick={() => setZoom(z => Math.max(0.3, z / 1.2))} className="bg-white border rounded px-2 py-1 text-xs shadow hover:bg-slate-50">−</button>
            <span className="bg-white border rounded px-2 py-1 text-xs shadow">{Math.round(zoom * 100)}%</span>
          </div>
        </div>

        {/* Marks list */}
        {marks.length > 0 && (
          <div className="bg-white border border-[#E6E6EA] rounded-lg p-3 max-h-40 overflow-y-auto">
            <h4 className="text-xs font-semibold mb-2">סימונים ({marks.length})</h4>
            <div className="flex flex-col gap-1">
              {marks.map(m => {
                const cat = WORK_CATEGORIES.find(c => c.key === m.categoryKey)!;
                const roomName = rooms.find(r => r.id === m.roomId)?.name;
                return (
                  <div key={m.uid} className="flex items-center justify-between text-xs border-b border-slate-100 pb-1">
                    <span>
                      <span className="w-2 h-2 rounded-full inline-block mr-1" style={{ background: cat.color }}></span>
                      {cat.icon} {cat.label} — <strong>{m.measurement.toFixed(2)} {m.unit}</strong>
                      {roomName && <span className="text-slate-400 mr-1"> ({roomName})</span>}
                    </span>
                    <button onClick={() => onMarksChange(marks.filter(x => x.uid !== m.uid))} className="text-red-400 hover:text-red-600">✕</button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ──────────────────────────────────────────────────────────────────────────────
// Main LayerModePage
// ──────────────────────────────────────────────────────────────────────────────
export const LayerModePage: React.FC = () => {
  const [plans, setPlans] = React.useState<PlanSummary[]>([]);
  const [selectedId, setSelectedId] = React.useState<string>("");
  const [pxPerMeter, setPxPerMeter] = React.useState(200);
  const [activeTab, setActiveTab] = React.useState<"rooms" | "paint">("rooms");
  const [rooms, setRooms] = React.useState<Room[]>([]);
  const [paintMarks, setPaintMarks] = React.useState<PaintMark[]>([]);

  React.useEffect(() => {
    listWorkshopPlans().then(setPlans).catch(() => {});
  }, []);

  // When a plan is selected, try to set pxPerMeter from plan scale
  React.useEffect(() => {
    if (!selectedId) return;
    const plan = plans.find(p => p.id === selectedId);
    if (plan?.scale_px_per_meter) setPxPerMeter(Math.round(plan.scale_px_per_meter));
  }, [selectedId, plans]);

  const loadKindergartenRooms = () => {
    setRooms([
      { id: genUid(), name: "כיתת גן א'",  area_m2: 63, assignments: [] },
      { id: genUid(), name: "כיתת גן ב'",  area_m2: 63, assignments: [] },
      { id: genUid(), name: "שירותים",      area_m2: 18, assignments: [] },
      { id: genUid(), name: "מחסן",         area_m2: 9,  assignments: [] },
      { id: genUid(), name: "מטבחון",       area_m2: 12, assignments: [] },
      { id: genUid(), name: "מבואה",        area_m2: 20, assignments: [] },
      { id: genUid(), name: "חצר פנימית",  area_m2: 80, assignments: [] },
    ]);
    setActiveTab("rooms");
  };

  const exportAll = () => {
    const data = { plan_id: selectedId, pxPerMeter, rooms, paintMarks, exportedAt: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `layer_marks_${selectedId || "export"}.json`;
    a.click();
  };

  // Combined BOQ from both modes
  const combinedTotals: Record<string, { label: string; unit: string; total: number; color: string }> = {};
  for (const r of rooms) {
    for (const a of r.assignments) {
      const cat = WORK_CATEGORIES.find(c => c.key === a.categoryKey)!;
      if (!combinedTotals[a.categoryKey]) combinedTotals[a.categoryKey] = { label: cat.label, unit: cat.unit, total: 0, color: cat.color };
      combinedTotals[a.categoryKey].total += a.quantity;
    }
  }
  for (const m of paintMarks) {
    const cat = WORK_CATEGORIES.find(c => c.key === m.categoryKey)!;
    if (!combinedTotals[m.categoryKey]) combinedTotals[m.categoryKey] = { label: cat.label, unit: cat.unit, total: 0, color: cat.color };
    combinedTotals[m.categoryKey].total += m.measurement;
  }

  const tabBtn = (id: "rooms" | "paint", label: string) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`px-4 py-2 text-sm rounded-full border transition-colors ${
        activeTab === id
          ? "bg-[#FF4B4B] text-white border-[#FF4B4B]"
          : "bg-white border-[#E6E6EA] hover:bg-slate-50"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="mt-4 flex flex-col gap-4">
      {/* Header */}
      <div className="bg-white border border-[#E6E6EA] rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-4 justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#31333F]">🎨 שכבות מנהל</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              סמן כמויות לפי חדרים, או צייר ישירות על גבי התוכנית
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="text-xs text-slate-500">תוכנית:</label>
            <select
              className="border border-[#E6E6EA] rounded-lg px-3 py-1.5 text-sm"
              value={selectedId}
              onChange={e => setSelectedId(e.target.value)}
            >
              <option value="">— בחר תוכנית —</option>
              {plans.map(p => <option key={p.id} value={p.id}>{p.plan_name || p.filename}</option>)}
            </select>
            <label className="text-xs text-slate-500">px/מטר:</label>
            <input
              type="number" className="border border-[#E6E6EA] rounded-lg px-2 py-1.5 text-sm w-20"
              value={pxPerMeter}
              onChange={e => setPxPerMeter(Number(e.target.value) || 200)}
            />
            <button onClick={exportAll} className="px-3 py-1.5 bg-[#31333F] text-white text-xs rounded-lg hover:bg-slate-700">
              ⬇️ ייצוא JSON
            </button>
          </div>
        </div>
      </div>

      {/* Tabs + quick-load */}
      <div className="flex gap-2 items-center flex-wrap">
        {tabBtn("rooms", "🏠 לפי חדרים")}
        {tabBtn("paint", "✏️ ציור על תוכנית")}
        <div className="mr-auto flex gap-2">
          <button
            onClick={loadKindergartenRooms}
            className="px-3 py-1.5 text-xs border border-dashed border-slate-300 rounded-full hover:bg-slate-50 text-slate-500"
          >
            📋 טען חדרים — גן ילדים
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="bg-white border border-[#E6E6EA] rounded-xl p-4">
        {activeTab === "rooms" ? (
          <RoomPanel rooms={rooms} onRoomsChange={setRooms} />
        ) : !selectedId ? (
          <div className="text-center py-20 text-slate-400">
            <p className="text-4xl mb-3">🗺️</p>
            <p className="text-sm">בחר תוכנית מהתפריט למעלה כדי לצייר עליה שכבות</p>
          </div>
        ) : (
          <PaintCanvas
            planId={selectedId}
            pxPerMeter={pxPerMeter}
            marks={paintMarks}
            onMarksChange={setPaintMarks}
            rooms={rooms}
          />
        )}
      </div>

      {/* Combined summary bar */}
      {Object.keys(combinedTotals).length > 0 && (
        <div className="bg-white border border-[#E6E6EA] rounded-xl p-4">
          <h3 className="font-semibold text-sm text-[#31333F] mb-3">
            📊 סיכום כולל — חדרים + ציור
          </h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(combinedTotals).map(([key, v]) => (
              <div key={key} className="px-3 py-2 rounded-lg text-white text-sm font-medium" style={{ background: v.color }}>
                {v.label}: <strong>{v.total.toFixed(2)} {v.unit}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
