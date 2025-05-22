from .base import BaseCounterfactual
from .attention_based import AB_CF
from .native_guide import NG
from .comte import CoMTE
from .tsevo import TSEvoCF
from .sg_cf import SG_CF
from .timex import TimeX
from .glacier import Glacier
from .latent_cf import LatentCF
from .discox import DiscoX
from .time_cf import Time_CF
from .system_dynamics import SDC
from .dynamic_systems import DynamicSystemCF
from .cels import AutoCELS


# Dictionary mapping method names to their classes
METHOD_MAP = {
    'AB_CF': AB_CF,
    'NG': NG,
    'CoMTE': CoMTE,
    'TSEvoCF': TSEvoCF,
    'SG_CF': SG_CF,
    'TimeX': TimeX,
    'CELS': AutoCELS,  # Updated to use AutoCELS
    'Glacier': Glacier,
    'LatentCF': LatentCF,
    'DiscoX': DiscoX,
    'SDC': SDC,
    'DynamicSystemCF': DynamicSystemCF,
    'Time_CF': Time_CF
}
