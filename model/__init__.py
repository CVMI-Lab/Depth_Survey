from .marigold import Marigold
from .unidepth import UniDepth
from .depthanythingv2 import DepthAnythingV2
from .geowizard import GeoWizard
from .dpt import DPT
from .leres import LeReS
from .metric3dv2 import Metric3Dv2
from .midasv31 import MiDasV31
from .diffe2eft import DiffE2EFT

try:
    from .genpercept import GenPercept
except:
    print("Failed to import GenPercept. It's fine if not used.")
    pass
