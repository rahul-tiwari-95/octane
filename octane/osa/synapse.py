"""Synapse EventBus re-export for OSA.

The actual SynapseEventBus lives in octane.models.synapse.
This module provides convenience access from the OSA package.
"""

from octane.models.synapse import SynapseEvent, SynapseEventBus, SynapseTrace

__all__ = ["SynapseEvent", "SynapseEventBus", "SynapseTrace"]
