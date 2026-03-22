"""octane.security — hardware-backed vault and network control."""

from octane.security.vault import VaultManager, VaultError, VaultNotFoundError, VaultLockedError
from octane.security.airgap import AirgapManager, is_airgap_active
from octane.security.provenance import build_provenance, ProvenanceRecord

__all__ = [
    "VaultManager",
    "VaultError",
    "VaultNotFoundError",
    "VaultLockedError",
    "AirgapManager",
    "is_airgap_active",
    "build_provenance",
    "ProvenanceRecord",
]
