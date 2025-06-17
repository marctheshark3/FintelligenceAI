"""
Ergo blockchain utilities for Python development.

This module provides helper functions for interacting with the Ergo blockchain.
"""


import requests


class ErgoNode:
    """Simple Ergo node client."""

    def __init__(self, base_url: str = "http://localhost:9053"):
        self.base_url = base_url.rstrip("/")

    def get_info(self) -> dict:
        """Get node information."""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()

    def get_height(self) -> int:
        """Get current blockchain height."""
        info = self.get_info()
        return info.get("fullHeight", 0)

    def get_unconfirmed_transactions(self) -> list[dict]:
        """Get unconfirmed transactions from mempool."""
        response = requests.get(f"{self.base_url}/transactions/unconfirmed")
        response.raise_for_status()
        return response.json()


def validate_address(address: str) -> bool:
    """Validate an Ergo address format."""
    # Basic validation - in practice use more sophisticated checks
    return len(address) > 40 and address.startswith(("9", "3", "2"))


def nanoerg_to_erg(nanoerg: int) -> float:
    """Convert nanoERG to ERG."""
    return nanoerg / 1_000_000_000


def erg_to_nanoerg(erg: float) -> int:
    """Convert ERG to nanoERG."""
    return int(erg * 1_000_000_000)
