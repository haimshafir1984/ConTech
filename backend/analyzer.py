from ._compat import load_root_module

_mod = load_root_module("analyzer")

FloorPlanAnalyzer = _mod.FloorPlanAnalyzer
parse_scale = _mod.parse_scale
