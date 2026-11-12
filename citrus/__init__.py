"""CITRUS: scatter-corrected X-ray CT reconstruction with a deterministic transport solver in the loop.

The scans are cone beams on flat or curved panels, circular or helical, and parallel beams.

The package holds the CITRUS loop (``citrus.citrus``) and its multigrid stages (``citrus.multigrid``), the LBTE
scatter solver (``citrus.scatter_solver``), the MBIRTORCH bridge (``citrus.recon``) and the phantoms, density model,
view angles, synthetic data and plotting helpers (``citrus.utilities``). The main entry points of the loop are
re-exported here.
"""

__version__ = "0.28.0"

from .citrus import CitrusConfig, CitrusState, Schedule, citrus_recon, citrus_step, mbir_recon  # noqa: E402
from .multigrid import Stage  # noqa: E402

__all__ = ["__version__", "CitrusConfig", "CitrusState", "Schedule", "Stage", "citrus_recon", "citrus_step",
           "mbir_recon"]
