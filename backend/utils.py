from ._compat import load_root_module

_mod = load_root_module("utils")

calculate_area_m2 = _mod.calculate_area_m2
clean_metadata_for_json = _mod.clean_metadata_for_json
create_colored_overlay = _mod.create_colored_overlay
extract_segments_from_mask = _mod.extract_segments_from_mask
load_stats_df = _mod.load_stats_df
refine_flooring_mask_with_rooms = _mod.refine_flooring_mask_with_rooms
safe_process_metadata = _mod.safe_process_metadata
