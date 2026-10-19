"""CITRUS: scatter-corrected X-ray CT reconstruction with a deterministic transport solver in the loop.

The scans are cone beams on flat or curved panels, circular or helical, and parallel beams.

The package holds the CITRUS loop (``citrus.citrus``), the LBTE scatter solver (``citrus.scatter_solver``), the
MBIRTORCH bridge (``citrus.recon``) and the WLAB phantom, density model, view angles, synthetic data and plotting
helpers (``citrus.utilities``). The main entry points of the loop are re-exported here.
"""

__version__ = "0.18.3"

from .citrus import (CitrusConfig, CitrusState, citrus_recon, citrus_step, mbir_recon)  # noqa: E402

__all__ = ["__version__", "CitrusConfig", "CitrusState", "citrus_recon", "citrus_step", "mbir_recon"]
