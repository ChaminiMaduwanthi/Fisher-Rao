"""Information-geometric curvature analysis of transformer representations."""

# Must run BEFORE anything imports huggingface_hub, which caches an SSL context
# at import time.  See net.py: this machine's TLS is intercepted by Avast, whose
# root CA is malformed in a way OpenSSL rejects and Windows does not.
from . import net as _net

_net.enable()

from .model import LM, DEFAULT_MODEL  # noqa: E402
from .metrics import (
    manson_metric,
    euclidean_metric,
    fisher_metric,
    truncation_error,
    radial_null_check,
    spectrum,
    gate_a,
)  # noqa: E402
from .trajectory import salience, curvature, layer_trajectory  # noqa: E402

__all__ = [
    "net",
    "LM", "DEFAULT_MODEL",
    "manson_metric", "euclidean_metric", "fisher_metric",
    "truncation_error", "spectrum", "gate_a", "radial_null_check",
    "salience", "curvature", "layer_trajectory",
]
