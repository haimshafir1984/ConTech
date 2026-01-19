"""
extractor.py - Deterministic pre-parser for architectural plan text
Extracts structured candidates using regex before LLM processing
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class EvidenceMatch:
    """Represents a match with its evidence text"""
    value: Any
    evidence: str
    confidence: float = 0.8
    
    def to_dict(self):
        return {
            "value": self.value,
            "evidence": self.evidence,
            "confidence": self.confidence
        }


class ArchitecturalTextExtractor:
    """
    Deterministic extraction of architectural plan metadata using regex.
    Returns structured candidates for LLM validation.
    """
    
    # Regex patterns for Hebrew architectural plans
    PATTERNS = {
        # Room with area: "חדר שינה ר"מ 15.5" or "סלון ר״מ 60"
        'room_area': re.compile(
            r'(?P<name>[\u0590-\u05FF\s\-״׳"\']{2,50}?)\s*'
            r'ר[״"]?מ\s*'
            r'(?P<area>\d+(?:\.\d+)?)',
            re.UNICODE
        ),
        
        # Scale: "קנ"מ 1:50" or "קנה מידה 1:100"
        'scale': re.compile(
            r'(?:קנ[״"]?מ|קנה\s*מידה)\s*'
            r'(?P<ratio>1\s*:\s*\d+)',
            re.UNICODE
        ),
        
        # Height: "H=2.80" or "גובה=3.00"
        'height_h': re.compile(
            r'H\s*=\s*(?P<height>\d+(?:\.\d+)?)'
        ),
        
        # Floor levels: "פ.ת +2.80" or "פ.ב -0.15" or "פ.ר ±0.00"
        'level_pt': re.compile(
            r'פ\.\s*ת\s*(?P<sign>[+\-±]?)\s*(?P<value>\d+(?:\.\d+)?)',
            re.UNICODE
        ),
        
        'level_pb': re.compile(
            r'פ\.\s*ב\s*(?P<sign>[+\-±]?)\s*(?P<value>\d+(?:\.\d+)?)',
            re.UNICODE
        ),
        
        'level_pr': re.compile(
            r'פ\.\s*ר\s*(?P<sign>[+\-±]?)\s*(?P<value>\d+(?:\.\d+)?)',
            re.UNICODE
        ),
        
        # Floor/Level labels: "קומה 2" or "מפלס +3"
        'floor_label': re.compile(
            r'(?:קומה|מפלס)\s*(?P<value>[\u0590-\u05FF\d\s\-+]+)',
            re.UNICODE
        ),
        
        # Plan title: "תכנית ..." or "תוכנית ..."
        'plan_title': re.compile(
            r'(?:תכנית|תוכנית)\s*(?P<title>[\u0590-\u05FF\s\-״׳"\']+?)(?:\s*(?:קומה|מפלס|קנ)|\n|$)',
            re.UNICODE
        ),
        
        # Date patterns: "25/12/2023" or "25.12.23"
        'date': re.compile(
            r'(?P<day>\d{1,2})[/\.](?P<month>\d{1,2})[/\.](?P<year>\d{2,4})'
        ),
        
        # Sheet number: "גליון 5/12" or "דף 3"
        'sheet_number': re.compile(
            r'(?:גליון|דף)\s*(?P<current>\d+)(?:\s*/\s*(?P<total>\d+))?',
            re.UNICODE
        ),
        
        # Ceiling notes: "תקרה אקוסטית" or "גבס"
        'ceiling_keywords': re.compile(
            r'(?P<keyword>תקרה|גבס|אקוסט|תקרת|לוחות\s*מינרלים|ארקליט)',
            re.UNICODE
        ),
        
        # Wall notes: "קיר בטון" or "בלוקים"
        'wall_keywords': re.compile(
            r'(?P<keyword>קיר|בטון|בלוק|קל\s*משקל|בידוד)',
            re.UNICODE
        ),
        
        # Flooring notes: "ריצוף" or "פרקט"
        'flooring_keywords': re.compile(
            r'(?P<keyword>ריצוף|אריח|גרניט|פרקט|שיפוע)',
            re.UNICODE
        ),
    }
    
    def __init__(self):
        self.matches = {}
        
    def extract_candidates(self, text: str) -> Dict[str, Any]:
        """
        Main extraction function - extracts all structured candidates from text.
        
        Args:
            text: Raw text from PDF (preferably full text, not truncated)
            
        Returns:
            Dictionary with structured candidates including evidence
        """
        if not text or len(text.strip()) < 10:
            return self._empty_result()
        
        # Normalize text
        text = self._normalize_text(text)
        
        return {
            "rooms": self._extract_rooms(text),
            "scale": self._extract_scale(text),
            "heights": self._extract_heights(text),
            "levels": self._extract_levels(text),
            "document_info": self._extract_document_info(text),
            "notes": self._extract_notes(text),
            "keywords": self._extract_keywords(text),
            "_raw_text_length": len(text),
            "_extraction_version": "1.0"
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better regex matching"""
        # Replace common Hebrew quote variations
        text = text.replace('"', '"').replace("׳", "'")
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _extract_rooms(self, text: str) -> List[Dict[str, Any]]:
        """Extract rooms with their areas"""
        rooms = []
        
        for match in self.PATTERNS['room_area'].finditer(text):
            name = match.group('name').strip()
            area = float(match.group('area'))
            evidence = self._get_evidence(text, match.start(), match.end())
            
            # Filter out false positives (too short or numeric names)
            if len(name) < 2 or name.isdigit():
                continue
                
            rooms.append({
                "name": EvidenceMatch(name, evidence, 0.85).to_dict(),
                "area_m2": EvidenceMatch(area, evidence, 0.9).to_dict(),
            })
        
        return rooms
    
    def _extract_scale(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract plan scale (קנ"מ)"""
        match = self.PATTERNS['scale'].search(text)
        if match:
            ratio = match.group('ratio').strip()
            evidence = self._get_evidence(text, match.start(), match.end())
            
            # Parse ratio (1:50 -> 50)
            try:
                scale_value = int(ratio.split(':')[-1].strip())
                return {
                    "value": scale_value,
                    "ratio": ratio,
                    "evidence": evidence,
                    "confidence": 0.95
                }
            except:
                pass
        
        return None
    
    def _extract_heights(self, text: str) -> List[Dict[str, Any]]:
        """Extract height measurements (H=...)"""
        heights = []
        
        for match in self.PATTERNS['height_h'].finditer(text):
            height = float(match.group('height'))
            evidence = self._get_evidence(text, match.start(), match.end())
            
            heights.append({
                "type": "H",
                "value_m": height,
                "evidence": evidence,
                "confidence": 0.9
            })
        
        return heights
    
    def _extract_levels(self, text: str) -> List[Dict[str, Any]]:
        """Extract floor/ceiling levels (פ.ת, פ.ב, פ.ר)"""
        levels = []
        
        # Extract פ.ת (finished ceiling)
        for match in self.PATTERNS['level_pt'].finditer(text):
            sign = match.group('sign') or '+'
            value = float(match.group('value'))
            if sign == '-':
                value = -value
            evidence = self._get_evidence(text, match.start(), match.end())
            
            levels.append({
                "label": "פ.ת",
                "type": "finished_ceiling",
                "value_m": value,
                "evidence": evidence,
                "confidence": 0.9
            })
        
        # Extract פ.ב (finished floor)
        for match in self.PATTERNS['level_pb'].finditer(text):
            sign = match.group('sign') or '+'
            value = float(match.group('value'))
            if sign == '-':
                value = -value
            evidence = self._get_evidence(text, match.start(), match.end())
            
            levels.append({
                "label": "פ.ב",
                "type": "finished_floor",
                "value_m": value,
                "evidence": evidence,
                "confidence": 0.9
            })
        
        # Extract פ.ר (rough floor)
        for match in self.PATTERNS['level_pr'].finditer(text):
            sign = match.group('sign') or '+'
            value = float(match.group('value'))
            if sign == '-':
                value = -value
            evidence = self._get_evidence(text, match.start(), match.end())
            
            levels.append({
                "label": "פ.ר",
                "type": "rough_floor",
                "value_m": value,
                "evidence": evidence,
                "confidence": 0.9
            })
        
        return levels
    
    def _extract_document_info(self, text: str) -> Dict[str, Any]:
        """Extract document metadata (title, date, sheet numbers)"""
        info = {}
        
        # Plan title
        match = self.PATTERNS['plan_title'].search(text)
        if match:
            title = match.group('title').strip()
            evidence = self._get_evidence(text, match.start(), match.end())
            info['plan_title'] = {
                "value": title,
                "evidence": evidence,
                "confidence": 0.85
            }
        
        # Date
        match = self.PATTERNS['date'].search(text)
        if match:
            day = match.group('day')
            month = match.group('month')
            year = match.group('year')
            # Expand 2-digit year
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            
            date_str = f"{day}/{month}/{year}"
            evidence = self._get_evidence(text, match.start(), match.end())
            info['date'] = {
                "value": date_str,
                "evidence": evidence,
                "confidence": 0.9
            }
        
        # Sheet numbers
        match = self.PATTERNS['sheet_number'].search(text)
        if match:
            current = match.group('current')
            total = match.group('total') if match.group('total') else None
            evidence = self._get_evidence(text, match.start(), match.end())
            
            sheet_info = {"current": int(current)}
            if total:
                sheet_info['total'] = int(total)
            
            info['sheet_numbers'] = {
                "value": sheet_info,
                "evidence": evidence,
                "confidence": 0.9
            }
        
        # Floor/Level label
        match = self.PATTERNS['floor_label'].search(text)
        if match:
            floor_value = match.group('value').strip()
            evidence = self._get_evidence(text, match.start(), match.end())
            info['floor_or_level'] = {
                "value": floor_value,
                "evidence": evidence,
                "confidence": 0.85
            }
        
        return info
    
    def _extract_notes(self, text: str) -> List[Dict[str, Any]]:
        """Extract general notes and keywords context"""
        notes = []
        
        # Look for "הערות" sections
        note_pattern = re.compile(
            r'(?:הערות?|הערה)[:\s]*(?P<note>[\u0590-\u05FF\s\d\.,\-]{10,200})',
            re.UNICODE
        )
        
        for match in note_pattern.finditer(text):
            note_text = match.group('note').strip()
            evidence = self._get_evidence(text, match.start(), match.end())
            
            notes.append({
                "text": note_text,
                "evidence": evidence,
                "confidence": 0.75
            })
        
        return notes
    
    def _extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract architectural keywords for context"""
        keywords = {
            "ceiling": [],
            "walls": [],
            "flooring": []
        }
        
        # Ceiling keywords
        for match in self.PATTERNS['ceiling_keywords'].finditer(text):
            kw = match.group('keyword')
            if kw not in keywords['ceiling']:
                keywords['ceiling'].append(kw)
        
        # Wall keywords
        for match in self.PATTERNS['wall_keywords'].finditer(text):
            kw = match.group('keyword')
            if kw not in keywords['walls']:
                keywords['walls'].append(kw)
        
        # Flooring keywords
        for match in self.PATTERNS['flooring_keywords'].finditer(text):
            kw = match.group('keyword')
            if kw not in keywords['flooring']:
                keywords['flooring'].append(kw)
        
        return keywords
    
    def _get_evidence(self, text: str, start: int, end: int, context_chars: int = 40) -> str:
        """
        Extract evidence text with context around the match.
        Limited to ~80 chars total.
        """
        # Get context before and after
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(text), end + context_chars)
        
        evidence = text[ctx_start:ctx_end].strip()
        
        # Truncate if too long
        if len(evidence) > 80:
            evidence = evidence[:77] + "..."
        
        return evidence
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "rooms": [],
            "scale": None,
            "heights": [],
            "levels": [],
            "document_info": {},
            "notes": [],
            "keywords": {"ceiling": [], "walls": [], "flooring": []},
            "_raw_text_length": 0,
            "_extraction_version": "1.0"
        }


# Self-test examples
def _self_test():
    """Basic self-test with example Hebrew text"""
    extractor = ArchitecturalTextExtractor()
    
    test_text = """
    תכנית קומה ב' - בית ספר
    קנ"מ 1:50
    תאריך: 15/03/2024
    גליון 3/12
    
    חדר מורים ר"מ 25.5
    כיתה א' ר"מ 60
    מסדרון ר"מ 12.3
    
    פ.ת +2.80
    פ.ב ±0.00
    H=2.70
    
    הערות: תקרה אקוסטית בכיתות
    """
    
    result = extractor.extract_candidates(test_text)
    
    print("=== Self-Test Results ===")
    print(f"Rooms found: {len(result['rooms'])}")
    for room in result['rooms']:
        print(f"  - {room['name']['value']}: {room['area_m2']['value']} m²")
    
    if result['scale']:
        print(f"Scale: 1:{result['scale']['value']}")
    
    print(f"Levels found: {len(result['levels'])}")
    for level in result['levels']:
        print(f"  - {level['label']}: {level['value_m']}m")
    
    print(f"Heights found: {len(result['heights'])}")
    
    return result


if __name__ == "__main__":
    _self_test()
