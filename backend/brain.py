from ._compat import load_root_module

_mod = load_root_module("brain")

process_plan_metadata = _mod.process_plan_metadata
if hasattr(_mod, "analyze_legend_image"):
    analyze_legend_image = _mod.analyze_legend_image
if hasattr(_mod, "extract_from_architectural_pdf"):
    extract_from_architectural_pdf = _mod.extract_from_architectural_pdf
