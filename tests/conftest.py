from pathlib import Path

import pytest
import torch

from citrus.scatter_solver.cross_section_library import load_spectrum_txt

# the sample spectra (copies of demo/spectra): the 125 kVp tube spectrum of the WLAB validation and the 6 MV linac
# spectrum
SPECTRA_DIR = Path(__file__).parent / "data"
SPECTRUM_125KVP = load_spectrum_txt(str(SPECTRA_DIR / "spectrum_125kvp.txt"))
SPECTRUM_6MV = load_spectrum_txt(str(SPECTRA_DIR / "spectrum_6mv.txt"))


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)
