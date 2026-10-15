"""Reconstruction side of CITRUS: the MBIRTORCH bridge and the synthetic WLAB dataset.

    mbir_bridge : ConeBeamModel matching the scatter solver's Grid / ConeGeometry, forward / back projection, FDK, and
                  the warm-started single-VCD-iteration recipe (`mbir_iterations`)
    synthetic   : synthesize_wlab_dataset (primary from MBIRTORCH, scatter from the LBTE solver, Poisson noise)

MBIRTORCH is imported lazily inside mbir_bridge so that `import citrus` never needs it.
"""
