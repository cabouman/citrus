"""CITRUS: scatter-corrected cone-beam CT reconstruction with a deterministic transport solver in the loop.

    citrus.citrus           the CITRUS algorithm (citrus_recon, mbir_recon, citrus_step and the per-view operators)
    citrus.scatter_solver   the LBTE source iteration scatter solver (ScatterSolver)
    citrus.recon            MBIRTORCH bridge and synthetic WLAB data
    citrus.utilities        WLAB phantom and geometry, the seven-material density model, view angles, synthetic
                            data, plotting helpers
"""

__version__ = "0.3.0"

from .citrus import (CitrusConfig, CitrusState, citrus_recon, citrus_step, mbir_recon)  # noqa: E402

__all__ = ["__version__", "CitrusConfig", "CitrusState", "citrus_recon", "citrus_step", "mbir_recon"]
