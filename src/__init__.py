"""Code générique du kit. Importer `src.<module>`.

Deux effets de bord volontaires, à appliquer AVANT tout import de torch.

1. `KMP_DUPLICATE_LIB_OK` : sur une install Anaconda Windows, `import torch` trouve deux copies de
   `libiomp5md.dll` (MKL d'Anaconda + celle de torch) et **avorte le process** (OMP: Error #15).
   C'est le contournement documenté par Intel ; il reste un contournement (perf OpenMP dégradée),
   sans effet sur Linux/le cluster où la variable est simplement ignorée.

2. UTF-8 sur stdout/stderr : sous Windows la console et les redirections `> fichier` utilisent
   cp1252, et tout `print` contenant un accent ou « ≈ » lève UnicodeEncodeError. Équivalent
   manuel : `$env:PYTHONUTF8=1` (PowerShell).
"""
import os as _os
import sys as _sys

if _sys.platform == "win32":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def enable_utf8_stdio():
    """Passe stdout/stderr en UTF-8 (idempotent, sans effet si déjà le cas)."""
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            if _stream is not None and (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # flux non reconfigurable (pipe capturé, notebook) : on laisse tel quel
            pass


enable_utf8_stdio()
