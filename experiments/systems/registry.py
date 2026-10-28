"""Registry of the six systems used in the KEWS paper."""


from typing import Dict, List, Optional, Type, Any

from enum import Enum

from .types import BoxModel

from .rate.bautin import BautinConfig, BautinModel

from .rate.saddle_node import SaddleNodeConfig, SaddleNodeModel

from .bifurcation.may_harvesting import MayHarvestingConfig, MayHarvestingModel

from .bifurcation.rosenzweig_macarthur import RosenzweigMacArthurConfig, RosenzweigMacArthurModel

from .bifurcation.amoc import AMOCConfig, AMOCModel

from .rate.amoc import AMOCRateConfig, AMOCRateModel

class SystemType(Enum):
    """Types of tipping mechanisms."""
    BIFURCATION = "bifurcation"
    RATE = "rate"

class SystemSpec:
    """Unified specification for a tipping point system."""

    def __init__(
        self,
        name: str,
        system_type: SystemType,
        description: str,
        config_cls: Type[Any],
        model_cls: Type[BoxModel],
        bifurcation_system: Optional[str] = None,
        rate_system: Optional[str] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.system_type = system_type
        self.description = description
        self.config_cls = config_cls
        self.model_cls = model_cls
        self.bifurcation_system = bifurcation_system
        self.rate_system = rate_system
        self.config_kwargs = config_kwargs or {}

    def default_config(self) -> Any:
        """Create default configuration for this system."""
        return self.config_cls(**self.config_kwargs)

    def create_model(self, config: Any) -> BoxModel:
        """Instantiate the registered model with its configuration."""
        return self.model_cls(config)

SYSTEM_REGISTRY: Dict[str, SystemSpec] = {
    "r_bautin": SystemSpec(
        name="r_bautin",
        system_type=SystemType.RATE,
        rate_system="bautin",
        config_cls=BautinConfig,
        model_cls=BautinModel,
        description="Bautin rate-induced tipping",
    ),
    "r_saddle_node": SystemSpec(
        name="r_saddle_node",
        system_type=SystemType.RATE,
        rate_system="saddle_node",
        config_cls=SaddleNodeConfig,
        model_cls=SaddleNodeModel,
        description="Saddle-node rate-induced tipping",
    ),
    "r_amoc": SystemSpec(
        name="r_amoc",
        system_type=SystemType.RATE,
        rate_system="amoc",
        config_cls=AMOCRateConfig,
        model_cls=AMOCRateModel,
        description="AMOC 2-box model (rate-induced)",
    ),
    "b_harvesting": SystemSpec(
        name="b_harvesting",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="may_harvesting",
        config_cls=MayHarvestingConfig,
        model_cls=MayHarvestingModel,
        description="May's harvesting model (semi-real bifurcation)",
    ),
    "b_rosenzweig_macarthur_hopf": SystemSpec(
        name="b_rosenzweig_macarthur_hopf",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="rosenzweig_macarthur_hopf",
        config_cls=RosenzweigMacArthurConfig,
        model_cls=RosenzweigMacArthurModel,
        description="Rosenzweig-MacArthur consumer-resource model (Hopf bifurcation)",
        config_kwargs={"bifurcation_type": "hopf"},
    ),
    "b_amoc": SystemSpec(
        name="b_amoc",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="amoc",
        config_cls=AMOCConfig,
        model_cls=AMOCModel,
        description="AMOC 2-box model (bifurcation-induced)",
    ),
}

def get_rate_system_spec(rate_system_name: str) -> SystemSpec:
    """Get rate system specification by internal name (e.g., 'bautin' -> 'r_bautin')."""
    for name, spec in SYSTEM_REGISTRY.items():
        if spec.rate_system == rate_system_name:
            return spec
    raise ValueError(
        f"Unknown rate system: {rate_system_name}. "
        f"Available rate systems: {[s.rate_system for s in SYSTEM_REGISTRY.values() if s.rate_system]}"
    )

def get_bifurcation_real_system_spec(real_system_name: str) -> SystemSpec:
    """Get bifurcation real system specification by internal name (e.g., 'may_harvesting' -> 'b_harvesting')."""
    for name, spec in SYSTEM_REGISTRY.items():
        if spec.bifurcation_system == real_system_name:
            return spec
    raise ValueError(
        f"Unknown bifurcation real system: {real_system_name}. "
        f"Available systems: {[s.bifurcation_system for s in SYSTEM_REGISTRY.values() if s.bifurcation_system]}"
    )

