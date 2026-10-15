"""Reconstruction side of CITRUS: the MBIRTORCH bridge.

    mbir_bridge : ConeBeamModel matching the scatter solver's Grid / ConeGeometry, forward / back projection, FDK, and
                  the warm-started single-VCD-iteration recipe (`mbir_iterations`)

MBIRTORCH is imported lazily inside mbir_bridge so that `import citrus` never needs it.
"""
