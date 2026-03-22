"""VaultManager — hardware-backed encrypted vaults with Touch ID gate.

Each vault is a named collection of secrets, encrypted with AES-256-GCM.
The encryption key for each vault is stored in macOS Keychain with
`biometryCurrentSet` access control — physically inaccessible without
the enrolled fingerprint.

Architecture:
  - octane-auth (Swift binary)  →  macOS Keychain + Touch ID
  - VaultManager (this file)    →  AES-256-GCM encrypted vault files
  - Vault files                 →  ~/.octane/vaults/<name>.enc

Wire protocol with octane-auth:
  - Binary path  : ~/.octane/bin/octane-auth (built from octane-auth/)
  - All calls    : subprocess.run(), JSON stdout, exit code 0/1
  - Touch ID     : triggered by retrieve/check commands
  - Fallback     : if binary not found, warn but allow vault creation
                   with password-derived key (dev mode only)

Vault file format (binary):
  [4 bytes: magic "OVLT"]
  [1 byte:  version = 1]
  [12 bytes: AES-GCM nonce]
  [N bytes: AES-256-GCM ciphertext of JSON-encoded dict]

Usage:
    vm = VaultManager()
    vm.create("finance")            # creates vault, stores key in Keychain
    vm.write("finance", "api_key", "sk-abc123")   # Touch ID → encrypt → write
    value = vm.read("finance", "api_key")          # Touch ID → decrypt → value
    vm.destroy("finance")           # Touch ID → remove vault + Keychain item
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="security.vault")

# ── Configuration ─────────────────────────────────────────────────────────────

_OCTANE_DIR = Path.home() / ".octane"
_VAULT_DIR  = _OCTANE_DIR / "vaults"
_AUTH_BIN   = _OCTANE_DIR / "bin" / "octane-auth"

_MAGIC   = b"OVLT"
_VERSION = 1
_VALID_VAULTS = {"finance", "health", "research", "code"}

# ── Exceptions ────────────────────────────────────────────────────────────────

class VaultError(Exception):
    """Base vault error."""

class VaultNotFoundError(VaultError):
    """Vault file does not exist."""

class VaultLockedError(VaultError):
    """Touch ID failed or was cancelled."""

class VaultSetupError(VaultError):
    """octane-auth binary not installed — vault unavailable."""

# ── Core helpers ──────────────────────────────────────────────────────────────

def _auth_bin_available() -> bool:
    return _AUTH_BIN.exists() and os.access(_AUTH_BIN, os.X_OK)


def _run_auth(*args: str) -> dict:
    """Run octane-auth with the given arguments. Returns parsed JSON response.

    Raises VaultSetupError if binary not installed.
    Raises VaultLockedError if Touch ID failed.
    Raises VaultError on other failures.
    """
    if not _auth_bin_available():
        raise VaultSetupError(
            f"octane-auth binary not found at {_AUTH_BIN}.\n"
            "Build it first: cd octane-auth && bash build.sh"
        )
    result = subprocess.run(
        [str(_AUTH_BIN), *args],
        capture_output=True, text=True, timeout=30,
    )
    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        raise VaultError(f"octane-auth returned invalid JSON: {result.stdout!r}") from e

    if not data.get("success"):
        error_msg = data.get("error", "Unknown error")
        if "cancelled" in error_msg.lower() or "failed" in error_msg.lower():
            raise VaultLockedError(f"Touch ID: {error_msg}")
        raise VaultError(f"octane-auth error: {error_msg}")
    return data


def _vault_path(name: str) -> Path:
    return _VAULT_DIR / f"{name}.enc"


def _encrypt(key_bytes: bytes, plaintext: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns nonce + ciphertext."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise VaultError(
            "cryptography package not installed. Run: pip install cryptography"
        ) from e
    nonce = os.urandom(12)
    ct = AESGCM(key_bytes).encrypt(nonce, plaintext, None)
    return nonce + ct


def _decrypt(key_bytes: bytes, data: bytes) -> bytes:
    """AES-256-GCM decrypt. data = nonce (12 bytes) + ciphertext."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise VaultError(
            "cryptography package not installed. Run: pip install cryptography"
        ) from e
    nonce, ct = data[:12], data[12:]
    return AESGCM(key_bytes).decrypt(nonce, ct, None)


def _write_vault_file(path: Path, key_bytes: bytes, data: dict) -> None:
    """Serialize *data* dict and write encrypted vault file."""
    plaintext = json.dumps(data).encode()
    encrypted = _encrypt(key_bytes, plaintext)
    # Header: magic(4) + version(1)
    header = _MAGIC + struct.pack("B", _VERSION)
    path.write_bytes(header + encrypted)
    path.chmod(0o600)


def _read_vault_file(path: Path, key_bytes: bytes) -> dict:
    """Read and decrypt vault file. Returns the stored dict."""
    if not path.exists():
        raise VaultNotFoundError(f"Vault file not found: {path}")
    raw = path.read_bytes()
    # Validate magic + version
    if raw[:4] != _MAGIC:
        raise VaultError(f"Invalid vault file format: {path.name}")
    version = struct.unpack("B", raw[4:5])[0]
    if version != _VERSION:
        raise VaultError(f"Unsupported vault version {version}")
    try:
        decrypted = _decrypt(key_bytes, raw[5:])
    except Exception as e:
        raise VaultLockedError("Decryption failed — wrong key or corrupted vault.") from e
    return json.loads(decrypted.decode())


def _get_key(vault_name: str, reason: str | None = None) -> bytes:
    """Retrieve AES key from Keychain via Touch ID. Returns raw key bytes."""
    r = reason or f"Octane needs access to the {vault_name} vault."
    data = _run_auth("retrieve", vault_name, "__aes_key__", r)
    return bytes.fromhex(data["value"])


# ── VaultManager ─────────────────────────────────────────────────────────────

class VaultManager:
    """Manages named encrypted vaults with Touch ID key protection.

    Typical use:
        vm = VaultManager()
        vm.create("finance")
        vm.write("finance", "schwab_token", "Bearer abc123")
        token = vm.read("finance", "schwab_token")
    """

    def __init__(self) -> None:
        _VAULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Vault lifecycle ───────────────────────────────────────────────────────

    def create(self, name: str) -> None:
        """Create a new vault. Generates AES key, stores in Keychain, creates empty vault file.

        Raises VaultError if vault already exists.
        Raises VaultSetupError if octane-auth not installed.
        """
        path = _vault_path(name)
        if path.exists():
            raise VaultError(
                f"Vault '{name}' already exists. Use 'octane vault destroy {name}' first."
            )
        # Generate 32-byte AES-256 key, hex-encode for Keychain storage
        key_hex = os.urandom(32).hex()
        _run_auth("store", name, "__aes_key__", key_hex)

        # Write empty vault (Touch ID needed to re-read key for the file)
        key_bytes = bytes.fromhex(key_hex)
        _write_vault_file(path, key_bytes, {})
        logger.info("vault.created", vault=name)

    def destroy(self, name: str) -> None:
        """Permanently delete vault file and remove key from Keychain.

        Requires Touch ID to confirm identity before deletion.
        """
        # Touch ID confirmation: retrieve key (this also verifies identity)
        _get_key(name, f"Octane: Confirm destruction of {name} vault.")
        # Remove Keychain item
        _run_auth("delete", name, "__aes_key__")
        # Remove encrypted file
        path = _vault_path(name)
        if path.exists():
            path.unlink()
        logger.info("vault.destroyed", vault=name)

    def exists(self, name: str) -> bool:
        return _vault_path(name).exists()

    def list_vaults(self) -> list[str]:
        """Return names of all vaults with existing encrypted files."""
        return sorted(
            p.stem for p in _VAULT_DIR.glob("*.enc") if p.is_file()
        )

    # ── Secret read/write ─────────────────────────────────────────────────────

    def write(self, vault: str, key: str, value: str, reason: str | None = None) -> None:
        """Write (or overwrite) a secret in the vault. Triggers Touch ID."""
        if not self.exists(vault):
            raise VaultNotFoundError(
                f"Vault '{vault}' not found. Run: octane vault create {vault}"
            )
        key_bytes = _get_key(vault, reason)
        path = _vault_path(vault)
        data = _read_vault_file(path, key_bytes)
        data[key] = value
        _write_vault_file(path, key_bytes, data)
        logger.info("vault.write", vault=vault, key=key)

    def read(self, vault: str, key: str, reason: str | None = None) -> str | None:
        """Read a secret from the vault. Triggers Touch ID. Returns None if key missing."""
        if not self.exists(vault):
            raise VaultNotFoundError(
                f"Vault '{vault}' not found. Run: octane vault create {vault}"
            )
        key_bytes = _get_key(vault, reason)
        data = _read_vault_file(_vault_path(vault), key_bytes)
        return data.get(key)

    def read_all(self, vault: str, reason: str | None = None) -> dict[str, str]:
        """Read all secrets from a vault. Triggers Touch ID once."""
        if not self.exists(vault):
            raise VaultNotFoundError(
                f"Vault '{vault}' not found. Run: octane vault create {vault}"
            )
        key_bytes = _get_key(vault, reason)
        return _read_vault_file(_vault_path(vault), key_bytes)

    def delete_key(self, vault: str, key: str, reason: str | None = None) -> None:
        """Remove a single key from the vault. Triggers Touch ID."""
        if not self.exists(vault):
            raise VaultNotFoundError(
                f"Vault '{vault}' not found. Run: octane vault create {vault}"
            )
        key_bytes = _get_key(vault, reason)
        path = _vault_path(vault)
        data = _read_vault_file(path, key_bytes)
        data.pop(key, None)
        _write_vault_file(path, key_bytes, data)
        logger.info("vault.delete_key", vault=vault, key=key)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> list[dict]:
        """Return status of all vaults (no Touch ID required for status check)."""
        vaults = self.list_vaults()
        result = []
        for name in vaults:
            path = _vault_path(name)
            stat = path.stat()
            result.append({
                "name": name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "keychain_protected": _auth_bin_available(),
            })
        return result

    def auth_available(self) -> bool:
        return _auth_bin_available()
