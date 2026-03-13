# CLAUDE CODE PROMPT — שיפוץ UX מלא לפרספקטיבת מנהל עבודה

## בעיה

מנהל עבודה פותח את שלב 3 ורואה:
- תוכנית קטנה בפינה (באג גובה + סיידבר רחב גוזל מקום)
- סיידבר עם: 4 טאבים, סליידר ביטחון, 3 כפתורים, סטטיסטיקה, 2 פילטרים, ו-51 שורות רשימה
- הוא לא מבין מה לעשות ראשון ואיך לאשר

**מה מנהל עבודה צריך:**
1. ראות את התוכנית גדולה וברורה
2. ללחוץ "נתח" — לקבל תוצאה
3. לבדוק ולתקן טעויות (ישירות על התוכנית)
4. לקבל כתב כמויות
5. לאשר ולהמשיך

---

## שינויים נדרשים ב-`frontend/src/pages/PlanningPage.tsx`

### 1. תיקון גובה הקנבס — הבאג הקריטי ביותר

**מצא** `updateDisplaySizeFromImage` (או שם דומה) ו/או את המשתנה `maxW`.

**בעיה**: `maxW=920` מגביל רוחב בלבד — אין הגבלת גובה, אז תמונה גבוהה גורמת לקנבס ענק ולתוכנית שמוצגת בחלק התחתון בלבד.

**תיקון**: הגבל גם לפי גובה הקונטיינר:

```typescript
// מצא את הפונקציה updateDisplaySizeFromImage ושנה את החישוב:
const containerW = canvasContainerRef.current?.clientWidth  ?? 800;
const containerH = canvasContainerRef.current?.clientHeight ?? 600;
const maxW = containerW - 8;   // מרווח קטן
const maxH = containerH - 8;
const scale = Math.min(1, maxW / naturalW, maxH / naturalH);
setDisplaySize({ w: Math.round(naturalW * scale), h: Math.round(naturalH * scale) });
```

**גם**: ודא שה-canvas container עצמו מקבל גובה מלא:
```typescript
// בתוך JSX — הקונטיינר הראשי של שלב 3 (step === 3):
// שנה את style של div האב שמכיל sidebar + canvas:
style={{ display: 'flex', flex: 1, height: 'calc(100vh - 170px)', overflow: 'hidden' }}

// הקונטיינר של הקנבס עצמו:
style={{ flex: 1, position: 'relative', overflow: 'hidden', background: '#1A2744' }}
```

---

### 2. הגדלת קנבס — הסר כפתורי ניווט צפים ממרכז המסך

**הסר לחלוטין** את ה-floating toolbar שבתחתית הקנבס:
```typescript
// מחק את כל הקטע עם כפתורי ניווט: "1 →", icon paint, "התאם", "+", "−"
// הם מסתירים חלק מהתוכנית ומבלבלים
```

**במקומם**, הוסף כפתורי zoom פשוטים בפינה העליונה שמאל של הקנבס:
```typescript
<div style={{ position:'absolute', top:8, left:8, zIndex:20, display:'flex', gap:4 }}>
  <button onClick={() => setZoom(z => Math.min(z+0.2, 3))}
    style={{ background:'rgba(255,255,255,0.9)', border:'1px solid #ccc',
             borderRadius:4, padding:'4px 10px', fontSize:16, cursor:'pointer' }}>+</button>
  <button onClick={() => setZoom(z => Math.max(z-0.2, 0.3))}
    style={{ background:'rgba(255,255,255,0.9)', border:'1px solid #ccc',
             borderRadius:4, padding:'4px 10px', fontSize:16, cursor:'pointer' }}>−</button>
  <button onClick={() => setZoom(1)}
    style={{ background:'rgba(255,255,255,0.9)', border:'1px solid #ccc',
             borderRadius:4, padding:'4px 10px', fontSize:12, cursor:'pointer' }}>100%</button>
</div>
```

---

### 3. עיצוב מחדש של הסיידבר — פשטות למנהל עבודה

**הסר מהסיידבר לחלוטין:**
- 4 הטאבים (אוטו / אזור / ציור / טקסט) — תמיד הצג את תוכן "אוטו" ללא טאבים
- הסליידר "הוסף לפרויקט – מעל X% ביטחון"
- שורת הסטטיסטיקה "51 ממתינים · 51 קירות · 3 אביזרים"
- כפתורי "בחר הכל" / "בטל הכל"
- כפתור "ממתינים בלבד"
- שורת הפילטר הכל/גבוה/לא גבוה
- **רשימת 51 הקטעים הבודדים** (קטעים בסיסיים) — **זה המרכזי ביותר להסיר**
- כפתור "הוסף קטגוריה +"

**מה שנשאר בסיידבר (מלמעלה למטה):**

```typescript
// === A. כותרת תוכנית (קומפקטי) ===
<div style={{ padding:'8px 12px', borderBottom:'1px solid #e5e7eb',
              background:'#f8fafc', fontSize:12, color:'#64748b' }}>
  {autoVisionData?.plan_title || planMeta?.filename || 'תוכנית'}
</div>

// === B. כפתור ניתוח ראשי ===
<div style={{ padding:'12px' }}>
  <button onClick={handleAutoAnalyze}
    disabled={autoAnalyzing}
    style={{
      width:'100%', padding:'12px', borderRadius:8,
      background: autoAnalyzing ? '#94a3b8' : '#1D4ED8',
      color:'white', border:'none', cursor:'pointer',
      fontSize:15, fontWeight:700, display:'flex',
      alignItems:'center', justifyContent:'center', gap:8
    }}>
    {autoAnalyzing ? '⏳ מנתח...' : '🔍 נתח תוכנית'}
  </button>
</div>

// === C. כרטיסי סיכום (מופיעים רק אחרי ניתוח) ===
{autoSegments.length > 0 && (
  <div style={{ padding:'0 12px 12px' }}>
    <div style={{ fontSize:11, color:'#94a3b8', marginBottom:6, fontWeight:600 }}>
      תוצאות ניתוח
    </div>

    {/* קירות חיצוניים */}
    {(() => {
      const ext = autoSegments.filter(s => s.wall_type === 'exterior');
      const extLen = ext.reduce((a,s) => a + (s.length_m || 0), 0);
      if (!ext.length) return null;
      return (
        <div style={{ background:'#EFF6FF', borderRadius:6, padding:'8px 10px',
                      marginBottom:6, borderRight:'3px solid #1D4ED8' }}>
          <div style={{ fontWeight:700, fontSize:13 }}>
            🟦 קירות חיצוניים
          </div>
          <div style={{ fontSize:12, color:'#64748b' }}>
            {ext.length} קטעים · {extLen.toFixed(1)} מ׳
          </div>
        </div>
      );
    })()}

    {/* קירות פנימיים */}
    {(() => {
      const int = autoSegments.filter(s => s.wall_type === 'interior');
      const intLen = int.reduce((a,s) => a + (s.length_m || 0), 0);
      if (!int.length) return null;
      return (
        <div style={{ background:'#ECFDF5', borderRadius:6, padding:'8px 10px',
                      marginBottom:6, borderRight:'3px solid #059669' }}>
          <div style={{ fontWeight:700, fontSize:13 }}>
            🟩 קירות פנימיים
          </div>
          <div style={{ fontSize:12, color:'#64748b' }}>
            {int.length} קטעים · {intLen.toFixed(1)} מ׳
          </div>
        </div>
      );
    })()}

    {/* חלוקות/גבס */}
    {(() => {
      const par = autoSegments.filter(s => s.wall_type === 'partition');
      const parLen = par.reduce((a,s) => a + (s.length_m || 0), 0);
      if (!par.length) return null;
      return (
        <div style={{ background:'#FFFBEB', borderRadius:6, padding:'8px 10px',
                      marginBottom:6, borderRight:'3px solid #D97706' }}>
          <div style={{ fontWeight:700, fontSize:13 }}>
            🟧 גבס / הפרדה
          </div>
          <div style={{ fontSize:12, color:'#64748b' }}>
            {par.length} קטעים · {parLen.toFixed(1)} מ׳
          </div>
        </div>
      );
    })()}

    {/* אביזרים/חדרים מ-vision */}
    {autoVisionData && (
      <div style={{ background:'#F0FDF4', borderRadius:6, padding:'8px 10px',
                    marginBottom:6, borderRight:'3px solid #16a34a' }}>
        <div style={{ fontWeight:700, fontSize:13 }}>📐 מ-Vision</div>
        <div style={{ fontSize:12, color:'#64748b' }}>
          {autoVisionData.rooms?.length || 0} חדרים ·{' '}
          {autoVisionData.elements?.filter(e =>
            e.type?.includes('דלת') || e.type?.includes('door')).length || 0} דלתות ·{' '}
          {autoVisionData.elements?.filter(e =>
            e.type?.includes('חלון') || e.type?.includes('window')).length || 0} חלונות
        </div>
      </div>
    )}

    {/* הוראה */}
    <div style={{ fontSize:11, color:'#94a3b8', textAlign:'center', marginTop:4 }}>
      לחץ על קטע בתוכנית לבחירה · מקש Delete למחיקה
    </div>
  </div>
)}

// === D. כפתור כתב כמויות ===
{autoSegments.length > 0 && (
  <div style={{ padding:'0 12px 8px' }}>
    <button onClick={handleLoadBoq}
      disabled={boqLoading}
      style={{
        width:'100%', padding:'10px', borderRadius:8,
        background:'white', color:'#1D4ED8',
        border:'2px solid #1D4ED8', cursor:'pointer',
        fontSize:14, fontWeight:600
      }}>
      {boqLoading ? '⏳ טוען...' : '📊 כתב כמויות'}
    </button>
  </div>
)}

// === E. ספייסר + כפתור אשר ===
<div style={{ flex:1 }} />

<div style={{ padding:'12px', borderTop:'1px solid #e5e7eb' }}>
  <button
    onClick={handleConfirmAllAndContinue}
    style={{
      width:'100%', padding:'14px', borderRadius:8,
      background:'#16a34a', color:'white',
      border:'none', cursor:'pointer',
      fontSize:15, fontWeight:700
    }}>
    ✅ אשר ועבור לשלב 4 ←
  </button>
</div>
```

**רוחב הסיידבר**: שנה מ-360px ל-**280px** (פחות מקום = יותר קנבס).

---

### 4. שיפור אינטראקציה עם הקנבס

**הוסף**: כאשר המשתמש לוחץ על קטע בתוכנית (SVG rect/line), הקטע נבחר **ומוצג tooltip** עם:
- סוג הקיר
- אורך במטרים

**Delete segment on canvas click**:
```typescript
// בתוך onClickSegment:
const handleCanvasSegmentClick = (seg: AutoAnalyzeSegment, e: React.MouseEvent) => {
  e.stopPropagation();
  setSelectedSegId(prev => prev === seg.id ? null : seg.id);
};

// כפתור מחיקה קטן שמופיע ליד קטע שנבחר:
{selectedSegId && (() => {
  const seg = autoSegments.find(s => s.id === selectedSegId);
  if (!seg) return null;
  // חשב מיקום על הקנבס
  return (
    <div style={{
      position: 'absolute',
      // מיקום קרוב לקטע שנבחר
      top: seg.y1_px * zoomScale + 'px',
      left: seg.x1_px * zoomScale + 'px',
      zIndex: 30,
      background: 'white',
      border: '1px solid #ef4444',
      borderRadius: 6,
      padding: '4px 8px',
      fontSize: 12,
      boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
      display: 'flex',
      gap: 8,
      alignItems: 'center'
    }}>
      <span>{seg.suggested_subtype || seg.suggested_type}</span>
      <span style={{ color:'#64748b' }}>{seg.length_m?.toFixed(1)}מ׳</span>
      <button
        onClick={() => handleDeleteSegment(selectedSegId)}
        style={{ color:'#ef4444', border:'none', background:'none',
                 cursor:'pointer', fontWeight:700, fontSize:14 }}>
        🗑
      </button>
    </div>
  );
})()}
```

---

### 5. תיקון overlay — פחות אטימות

**מצא** `rgba(0,0,0,0.45)` (dimming overlay) ו**שנה** ל:
```typescript
rgba(0,0,0,0.12)
```

כך התוכנית עדיין נראית בצורה ברורה גם כשקטגוריה מסומנת.

---

### 6. הסר: פאנל BOQ מהסיידבר — עבור ל-Modal

כרגע כשלוחצים "כתב כמויות" התוצאה מופיעה בתוך הסיידבר ודוחקת את הכל.

**שנה**: כשה-BOQ נטען, הצג אותו ב-**Modal (dialog)** שמכסה את המסך:

```typescript
{boqVisible && boqData && (
  <div style={{
    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
    background: 'rgba(0,0,0,0.5)', zIndex: 100,
    display: 'flex', alignItems: 'center', justifyContent: 'center'
  }} onClick={() => setBoqVisible(false)}>
    <div style={{
      background: 'white', borderRadius: 12, padding: 24,
      maxWidth: 680, width: '90%', maxHeight: '80vh',
      overflowY: 'auto', position: 'relative'
    }} onClick={e => e.stopPropagation()}>

      {/* כותרת */}
      <div style={{ display:'flex', justifyContent:'space-between',
                    alignItems:'center', marginBottom:16 }}>
        <h2 style={{ margin:0, fontSize:18 }}>
          📊 כתב כמויות — {boqData.plan_title}
        </h2>
        <button onClick={() => setBoqVisible(false)}
          style={{ border:'none', background:'none', fontSize:20,
                   cursor:'pointer', color:'#64748b' }}>✕</button>
      </div>

      {/* סיכום מספרים גדולים */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)',
                    gap:12, marginBottom:20 }}>
        {[
          { label:'שטח בנוי', value: boqData.total_area_m2 + ' מ"ר', color:'#EFF6FF' },
          { label:'אורך קירות', value: boqData.total_wall_length_m + ' מ׳', color:'#ECFDF5' },
          { label:'חדרים', value: boqData.total_rooms, color:'#FFF7ED' },
        ].map(item => (
          <div key={item.label} style={{ background:item.color, borderRadius:8,
                                         padding:'12px', textAlign:'center' }}>
            <div style={{ fontSize:22, fontWeight:700 }}>{item.value}</div>
            <div style={{ fontSize:12, color:'#64748b' }}>{item.label}</div>
          </div>
        ))}
      </div>

      {/* טבלת קירות */}
      <h3 style={{ fontSize:14, marginBottom:8 }}>פירוט קירות</h3>
      <table style={{ width:'100%', borderCollapse:'collapse', marginBottom:16 }}>
        <thead>
          <tr style={{ background:'#f8fafc' }}>
            <th style={{ padding:'8px', textAlign:'right', fontSize:12 }}>סוג קיר</th>
            <th style={{ padding:'8px', textAlign:'center', fontSize:12 }}>קטעים</th>
            <th style={{ padding:'8px', textAlign:'center', fontSize:12 }}>אורך מ׳</th>
            <th style={{ padding:'8px', textAlign:'center', fontSize:12 }}>שטח מ"ר</th>
          </tr>
        </thead>
        <tbody>
          {boqData.walls?.map((w, i) => (
            <tr key={i} style={{ borderTop:'1px solid #f1f5f9' }}>
              <td style={{ padding:'8px', display:'flex', alignItems:'center', gap:6 }}>
                <span style={{ display:'inline-block', width:10, height:10,
                               borderRadius:2, background: w.color || '#6B7280' }} />
                {w.wall_type}
              </td>
              <td style={{ padding:'8px', textAlign:'center' }}>{w.count}</td>
              <td style={{ padding:'8px', textAlign:'center' }}>{w.total_length_m}</td>
              <td style={{ padding:'8px', textAlign:'center' }}>{w.wall_area_m2}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* פתחים */}
      <div style={{ display:'flex', gap:16, flexWrap:'wrap', marginBottom:16 }}>
        {boqData.door_count > 0 && (
          <div style={{ background:'#f8fafc', borderRadius:6, padding:'8px 12px' }}>
            🚪 דלתות: <strong>{boqData.door_count}</strong>
          </div>
        )}
        {boqData.window_count > 0 && (
          <div style={{ background:'#f8fafc', borderRadius:6, padding:'8px 12px' }}>
            🪟 חלונות: <strong>{boqData.window_count}</strong>
          </div>
        )}
        {Object.entries(boqData.fixture_counts || {}).map(([k, v]) => (
          <div key={k} style={{ background:'#f8fafc', borderRadius:6, padding:'8px 12px' }}>
            {k}: <strong>{v as number}</strong>
          </div>
        ))}
      </div>

      {/* חדרים */}
      {boqData.rooms?.length > 0 && (
        <>
          <h3 style={{ fontSize:14, marginBottom:8 }}>חדרים</h3>
          <div style={{ display:'flex', flexWrap:'wrap', gap:8 }}>
            {boqData.rooms.map((r, i) => (
              <div key={i} style={{ background:'#f0fdf4', borderRadius:6,
                                    padding:'6px 10px', fontSize:12 }}>
                {r.name} — {r.area_m2} מ"ר
              </div>
            ))}
          </div>
        </>
      )}

      {/* כפתור סגירה תחתון */}
      <div style={{ marginTop:20, textAlign:'center' }}>
        <button onClick={() => setBoqVisible(false)}
          style={{ padding:'10px 32px', background:'#1D4ED8',
                   color:'white', border:'none', borderRadius:8,
                   cursor:'pointer', fontSize:14, fontWeight:600 }}>
          סגור
        </button>
      </div>
    </div>
  </div>
)}
```

---

## סיכום — מה נוסף ומה הוסר

| מה הוסר | למה |
|---------|-----|
| 4 טאבים (אוטו/אזור/ציור/טקסט) | מבלבל, תמיד היה "אוטו" פעיל |
| סליידר ביטחון % | מנהל עבודה לא מבין "ביטחון 90%" |
| "ממתינים בלבד" / "הכל/גבוה/לא גבוה" | טכני מדי |
| "בחר הכל" / "בטל הכל" | לא ברור מה זה עושה |
| רשימת 51 קטעים בודדים | עומס מידע קיצוני |
| "הוסף קטגוריה +" | לא נצרך בזרימה הרגילה |
| Floating toolbar בקנבס | מסתיר תוכנית |
| BOQ inline בסיידבר | דוחק הכל |

| מה נוסף/שופר | למה |
|---------|-----|
| כרטיסי סיכום ברורים (חיצוני/פנימי/גבס) | מנהל עבודה רואה מה מצאנו בשנייה |
| Tooltip + כפתור מחיקה קטן בקנבס | מחיקה ישירה על המפה |
| BOQ כ-Modal מלא | לא מסתיר את הקנבס |
| "✅ אשר ועבור לשלב 4" בולט תמיד | זרימת עבודה ברורה |
| תיקון גובה קנבס | תוכנית מלאה ולא 30% |
| Zoom בפינה עליונה | ללא floating toolbar |

---

## בדיקה אחרי שינוי

1. פתח step 3 → ה-canvas צריך להציג את **כל** התוכנית (לא רק החלק התחתון)
2. לחץ "🔍 נתח תוכנית" → 3 כרטיסי סיכום צריכים להופיע
3. לחץ על קטע בקנבס → tooltip קטן עם כפתור 🗑
4. לחץ "📊 כתב כמויות" → Modal נפתח עם טבלה
5. לחץ "✅ אשר" → עובר לשלב 4
