from ._compat import load_root_module

_mod = load_root_module("floor_extractor")

analyze_floor_and_rooms = _mod.analyze_floor_and_rooms
