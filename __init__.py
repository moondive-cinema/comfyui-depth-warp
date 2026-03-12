import importlib.util, os

_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "depth_warp_node",
    os.path.join(_here, "depth_warp_node.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NODE_CLASS_MAPPINGS       = _mod.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _mod.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
