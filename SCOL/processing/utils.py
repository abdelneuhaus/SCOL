"""
Adapted from https://github.com/tmonseigne/palm-tracer
Extraction of localization pipeline and adapted to standalone use.
"""

import sys
import ctypes
import psutil

import numpy as np

from pathlib import Path
from typing import Optional



DLL_PATH = Path(__file__).parent / "DLL"


def load_dll(name: str) -> Optional[ctypes.CDLL]:
	"""
	Charge une DLL, si elle existe.

	:param name: Type de DLL (CPU, GPU).
	:return: Objet Python stockant la DLL chargée.
	"""
	ext = "dll" if sys.platform.startswith("win") else "dylib" if sys.platform == "darwin" else "so"
	path = DLL_PATH / f"PALMTracer_{name}.{ext}"

	try:
		return ctypes.cdll.LoadLibrary(str(path.resolve()))
	except OSError as e:
		print(f"Unable to load the DLL '{path.name}':\n\t{e}")
		return None



def as_c_contig(a: np.ndarray, dtype: np.dtype, *, writeable: bool) -> np.ndarray:
    """
    Retourne un tableau C-contigu du dtype demandé, sans copie si possible.

    :param a: Tableau d'entrée.
    :param dtype: Dtype souhaité.
    :param writeable: Garantit un buffer modifiable si True.
    :return: Tableau C-contigu compatible DLL.
    """
    if not isinstance(a, np.ndarray): a = np.asarray(a)

    if (a.dtype != dtype) or (not a.flags["C_CONTIGUOUS"]) or (writeable and not a.flags["WRITEABLE"]):
        a = np.ascontiguousarray(a, dtype=dtype)
        if writeable: a.setflags(write=True)
    return a



def max_allocation_bytes(fraction_available: float = 0.5, safety_gb: int = 1) -> int:
    """
    Permet de calculer la quantité de mémoire disponible au maximum pour une allocation.

    :param fraction_available: Pourcentage de la RAM disponible à utiliser au maximum.
    :param safety_gb: Marge de sécurité à garder disponible.
    :return: Valeur en byte de l'allocation maximum tolérée.
    """
    giga = 1024 * 1024 * 1024
    avail = psutil.virtual_memory().available
    safety = safety_gb * giga
    budget = int(max(0, avail - safety) * fraction_available)
    return max(budget, giga)
