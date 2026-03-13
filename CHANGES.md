# 📝 Changes Summary - Metadata Extraction Enhancement

## Quick Overview

**Goal:** Improve architectural plan metadata extraction from 3000 char limit to 20K chars with deterministic pre-parsing + LLM validation.

---

## Files Changed

### 1. ✏️ `analyzer.py` - Enhanced PDF Text Extraction
```python
# BEFORE (line ~498):
meta["raw_text"] = doc[0].get_text()[:3000]  # ❌ Truncated!

# AFTER:
meta["raw_text_full"] = full_text[:20000]     # ✅ Full text
meta["raw_blocks"] = sorted_blocks_with_bbox   # ✅ Structured blocks
meta["normalized_text"] = normalized_order     # ✅ Proper reading order
meta["raw_text"] = full_text[:3000]            # ✅ Kept for compatibility
```

**Impact:** 6.6× more text, proper structure, better context for LLM.

---

### 2. ⭐ `extractor.py` - NEW FILE - Deterministic Pre-Parser
```python
class ArchitecturalTextExtractor:
    """Regex-based extraction of architectural metadata"""
    
    PATTERNS = {
        'room_area': r'(?P<name>...)\s*ר"מ\s*(?P<area>\d+\.?\d*)',
        'scale': r'קנ"מ\s*(?P<ratio>1\s*:\s*\d+)',
        'level_pt': r'פ\.\s*ת\s*(?P<sign>[+\-±]?)\s*(?P<value>\d+\.?\d*)',
        'height_h': r'H\s*=\s*(?P<height>\d+\.?\d*)',
        # ... + 8 more patterns
    }
    
    def extract_candidates(text) -> dict:
        """Returns structured candidates with evidence for each match"""
        return {
            "rooms": [{name, area_m2, evidence}],
            "scale": {value, ratio, evidence},
            "levels": [{label, value_m, evidence}],
            "heights": [{type, value_m, evidence}],
            "document_info": {...},
            "notes": [...],
            "keywords": {...}
        }
```

**Features:**
- ✅ 12 Hebrew architectural patterns
- ✅ Evidence tracking (source text for every match)
- ✅ Confidence scoring
- ✅ Self-test included

**Self-test:**
```bash
$ python extractor.py
=== Self-Test Results ===
Rooms found: 3
  - חדר מורים: 25.5 m²
  - כיתה א': 60.0 m²
  - מסדרון: 12.3 m²
Scale: 1:50
Levels found: 2
  - פ.ת: 2.8m
  - פ.ב: 0.0m
Heights found: 1
```

---

### 3. ✏️ `brain.py` - Enhanced LLM Processing
```python
# NEW FUNCTION:
def safe_process_metadata(raw_text=None, raw_text_full=None, 
                          normalized_text=None, raw_blocks=None, 
                          candidates=None):
    """
    Enhanced extraction with:
    - Full text support (20K chars)
    - Deterministic candidates integration
    - Strict evidence-based prompt
    - Auto-fix for broken JSON
    """
    
    # Build strict prompt with candidates
    prompt = _build_strict_prompt(text, candidates)
    
    # Rules for LLM:
    # 1. No fabrication - evidence required
    # 2. Prefer candidates (deterministic)
    # 3. Confidence + reason for null fields
    # 4. JSON only output
    
    # Try multiple models with fallback
    for model in models:
        response = client.messages.create(...)
        try:
            result = json.loads(clean_response)
        except JSONDecodeError:
            result = _auto_fix_json(client, response)  # ✅ Auto-repair
    
    return result


# KEPT for backward compatibility:
def process_plan_metadata(raw_text):
    """Legacy function → redirects to safe_process_metadata"""
    return safe_process_metadata(raw_text=raw_text)
```

**New features:**
- ✅ Strict prompt with evidence requirements
- ✅ Candidates integration
- ✅ JSON validation + auto-fix
- ✅ Temperature=0.1 for factual extraction
- ✅ Comprehensive error handling

---

### 4. ✏️ `utils.py` - Updated Wrapper
```python
# BEFORE:
def safe_process_metadata(raw_text):
    return process_plan_metadata(raw_text)

# AFTER:
def safe_process_metadata(raw_text=None, meta=None):
    """Enhanced wrapper with full meta dict support"""
    
    if meta:  # New mode: full context
        extractor = ArchitecturalTextExtractor()
        candidates = extractor.extract_candidates(meta["normalized_text"])
        
        return brain_process(
            raw_text_full=meta["raw_text_full"],
            normalized_text=meta["normalized_text"],
            raw_blocks=meta["raw_blocks"],
            candidates=candidates
        )
    else:  # Legacy mode: backward compatible
        return brain_process(raw_text=raw_text)
```

---

### 5. ✏️ `app.py` - Updated Call Site
```python
# BEFORE (line ~103):
if meta.get("raw_text"):
    llm_data = safe_process_metadata(meta["raw_text"])

# AFTER:
if meta.get("raw_text_full") or meta.get("raw_text"):
    llm_data = safe_process_metadata(meta=meta)  # ✅ Pass full meta dict
```

---

## Output Schema

```json
{
  "document": {
    "plan_title": {"value": null, "confidence": 0, "evidence": []},
    "plan_type": {"value": null, "confidence": 0, "evidence": []},
    "scale": {"value": null, "confidence": 0, "evidence": []},
    "date": {"value": null, "confidence": 0, "evidence": []}
  },
  "rooms": [{
    "name": {"value": null, "confidence": 0, "evidence": []},
    "area_m2": {"value": null, "confidence": 0, "evidence": []},
    "ceiling_height_m": {"value": null, "confidence": 0, "evidence": []}
  }],
  "heights_and_levels": {
    "ceiling_levels_m": [{"value": null, "confidence": 0, "evidence": []}]
  },
  "execution_notes": {...},
  "quantities_hint": {...},
  "limitations": [...]
}
```

**Every field includes:**
- `value`: Extracted data
- `confidence`: 0-100 score
- `evidence`: Source text snippet

---

## Backward Compatibility

✅ **Fully backward compatible:**
- Old `raw_text` (3000 chars) still exists
- Old `process_plan_metadata(raw_text)` still works
- If new extraction fails → falls back to old method
- Existing code doesn't break

---

## Testing

### Quick test:
```bash
cd ConTech
python extractor.py  # Test regex extraction
streamlit run app.py  # Test full pipeline
```

### What to verify:
1. Upload a PDF plan
2. Check metadata extraction includes:
   - ✅ Full text (not truncated)
   - ✅ Rooms with areas
   - ✅ Scale (קנ"מ)
   - ✅ Levels (פ.ת, פ.ב)
   - ✅ Evidence for each field
3. Verify JSON is valid

---

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Text length** | 3,000 chars | 20,000 chars |
| **Structure** | Raw text only | Blocks + normalized |
| **Pre-parsing** | None | Regex candidates |
| **Evidence** | None | Full tracking |
| **Confidence** | None | Per-field scores |
| **JSON validation** | Basic | Auto-fix |
| **Hebrew support** | Partial | Full patterns |

---

## Pipeline Flow

```
PDF → Analyzer → Extractor → LLM → Validation → Output
        ↓           ↓           ↓        ↓
     20K text   Candidates  Validate  Auto-fix
     + blocks   + evidence  + enrich  JSON
```

**Philosophy:** 
- Regex finds facts (numbers, dates, measurements)
- LLM adds context and intelligence
- Validation ensures quality

---

**Status:** ✅ Ready for testing
**Compatibility:** ✅ Backward compatible
**Risk:** 🟢 Low (fallbacks in place)
