"""Tests for octane.security.vault — VaultManager.

All tests mock the octane-auth subprocess so they run offline without
Touch ID or a compiled Swift binary.  Encryption/decryption is tested
end-to-end using real AES-256-GCM via the cryptography library.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_auth_success(value: str | None = None) -> MagicMock:
    """Return a mock subprocess result that mimics octane-auth success."""
    m = MagicMock()
    d: dict = {"success": True}
    if value is not None:
        d["value"] = value
    m.stdout = json.dumps(d)
    m.returncode = 0
    return m


def _make_auth_failure(error: str = "Touch ID cancelled.") -> MagicMock:
    m = MagicMock()
    m.stdout = json.dumps({"success": False, "error": error})
    m.returncode = 1
    return m


# ── Encryption round-trip ─────────────────────────────────────────────────────

class TestEncryptionRoundTrip:
    """Tests for _encrypt / _decrypt helpers."""

    def test_roundtrip(self):
        from octane.security.vault import _encrypt, _decrypt
        key = os.urandom(32)
        plaintext = b"secret financial data"
        enc = _encrypt(key, plaintext)
        assert len(enc) > 12          # nonce + ciphertext
        dec = _decrypt(key, enc)
        assert dec == plaintext

    def test_wrong_key_raises(self):
        from octane.security.vault import _encrypt, _decrypt
        key1, key2 = os.urandom(32), os.urandom(32)
        enc = _encrypt(key1, b"data")
        with pytest.raises(Exception):
            _decrypt(key2, enc)

    def test_tampered_ciphertext_raises(self):
        from octane.security.vault import _encrypt, _decrypt
        key = os.urandom(32)
        enc = bytearray(_encrypt(key, b"data"))
        enc[-1] ^= 0xFF  # flip last byte
        with pytest.raises(Exception):
            _decrypt(key, bytes(enc))

    def test_unique_nonces(self):
        from octane.security.vault import _encrypt
        key = os.urandom(32)
        enc1 = _encrypt(key, b"same")
        enc2 = _encrypt(key, b"same")
        # Different nonces → different ciphertext
        assert enc1 != enc2


# ── Vault file format ─────────────────────────────────────────────────────────

class TestVaultFileFormat:
    """Tests for _write_vault_file / _read_vault_file helpers."""

    def test_write_and_read(self, tmp_path):
        from octane.security.vault import _write_vault_file, _read_vault_file, _MAGIC, _VERSION
        key = os.urandom(32)
        path = tmp_path / "test.enc"
        data = {"api_key": "secret123", "token": "abc"}
        _write_vault_file(path, key, data)
        result = _read_vault_file(path, key)
        assert result == data

    def test_magic_header(self, tmp_path):
        from octane.security.vault import _write_vault_file, _MAGIC, _VERSION
        key = os.urandom(32)
        path = tmp_path / "test.enc"
        _write_vault_file(path, key, {"x": "y"})
        raw = path.read_bytes()
        assert raw[:4] == _MAGIC
        assert struct.unpack("B", raw[4:5])[0] == _VERSION

    def test_invalid_magic_raises(self, tmp_path):
        from octane.security.vault import _read_vault_file, VaultError
        path = tmp_path / "bad.enc"
        path.write_bytes(b"XXXX\x01" + os.urandom(50))
        with pytest.raises(VaultError, match="Invalid vault file format"):
            _read_vault_file(path, os.urandom(32))

    def test_permissions_restricted(self, tmp_path):
        from octane.security.vault import _write_vault_file
        key = os.urandom(32)
        path = tmp_path / "test.enc"
        _write_vault_file(path, key, {})
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_missing_file_raises(self, tmp_path):
        from octane.security.vault import _read_vault_file, VaultNotFoundError
        path = tmp_path / "missing.enc"
        with pytest.raises(VaultNotFoundError):
            _read_vault_file(path, os.urandom(32))


# ── VaultManager ──────────────────────────────────────────────────────────────

class TestVaultManager:
    """Tests for VaultManager using mocked octane-auth binary."""

    @pytest.fixture
    def vm(self, tmp_path):
        """VaultManager wired to a temp vault directory."""
        from octane.security import vault as vault_mod
        # Redirect vault directory to temp path
        vault_mod._VAULT_DIR = tmp_path
        # Make auth binary appear available
        with patch.object(vault_mod, "_auth_bin_available", return_value=True):
            from octane.security.vault import VaultManager
            vm = VaultManager.__new__(VaultManager)
            vm.__init__()
            yield vm
        # Restore (tests are isolated anyway)
        vault_mod._VAULT_DIR = Path.home() / ".octane" / "vaults"

    def test_create_vault(self, vm, tmp_path):
        key_hex = os.urandom(32).hex()
        with patch("octane.security.vault._run_auth") as mock_auth, \
             patch("octane.security.vault._VAULT_DIR", tmp_path):
            mock_auth.return_value = {"success": True}
            # create() calls _run_auth("store", ...) then needs the key to write
            # We patch _get_key to return a stable key without Touch ID
            with patch("octane.security.vault._get_key", return_value=bytes.fromhex(key_hex)):
                vm.create("finance")
            vault_path = tmp_path / "finance.enc"
            assert vault_path.exists()

    def test_create_duplicate_raises(self, vm, tmp_path):
        from octane.security.vault import VaultError
        vault_path = tmp_path / "finance.enc"
        vault_path.write_bytes(b"dummy")  # simulate existing vault
        with patch("octane.security.vault._VAULT_DIR", tmp_path):
            with pytest.raises(VaultError, match="already exists"):
                vm.create("finance")

    def test_write_and_read(self, tmp_path):
        from octane.security.vault import VaultManager, _write_vault_file
        from octane.security import vault as vault_mod

        key = os.urandom(32)
        key_hex = key.hex()

        # Pre-create an empty vault file
        vault_path = tmp_path / "finance.enc"
        _write_vault_file(vault_path, key, {})

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path), \
             patch("octane.security.vault._get_key", return_value=key):
            vm.write("finance", "api_key", "sk-12345")
            val = vm.read("finance", "api_key")
            assert val == "sk-12345"

    def test_read_missing_key_returns_none(self, tmp_path):
        from octane.security.vault import VaultManager, _write_vault_file

        key = os.urandom(32)
        vault_path = tmp_path / "finance.enc"
        _write_vault_file(vault_path, key, {"existing": "value"})

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path), \
             patch("octane.security.vault._get_key", return_value=key):
            assert vm.read("finance", "nonexistent") is None

    def test_read_all(self, tmp_path):
        from octane.security.vault import VaultManager, _write_vault_file

        key = os.urandom(32)
        vault_path = tmp_path / "code.enc"
        original = {"github_token": "ghp_abc", "npm_token": "npm_xyz"}
        _write_vault_file(vault_path, key, original)

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path), \
             patch("octane.security.vault._get_key", return_value=key):
            result = vm.read_all("code")
            assert result == original

    def test_delete_key(self, tmp_path):
        from octane.security.vault import VaultManager, _write_vault_file, _read_vault_file

        key = os.urandom(32)
        vault_path = tmp_path / "finance.enc"
        _write_vault_file(vault_path, key, {"a": "1", "b": "2"})

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path), \
             patch("octane.security.vault._get_key", return_value=key):
            vm.delete_key("finance", "a")
            remaining = _read_vault_file(vault_path, key)
            assert "a" not in remaining
            assert remaining["b"] == "2"

    def test_vault_not_found_raises(self, tmp_path):
        from octane.security.vault import VaultManager, VaultNotFoundError

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path):
            with pytest.raises(VaultNotFoundError):
                vm.read("nonexistent", "key")

    def test_exists(self, tmp_path):
        from octane.security.vault import VaultManager

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path):
            assert not vm.exists("finance")
            (tmp_path / "finance.enc").write_bytes(b"x")
            assert vm.exists("finance")

    def test_list_vaults(self, tmp_path):
        from octane.security.vault import VaultManager

        (tmp_path / "finance.enc").write_bytes(b"x")
        (tmp_path / "code.enc").write_bytes(b"x")
        (tmp_path / "readme.txt").write_bytes(b"x")  # not a vault

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path):
            vaults = vm.list_vaults()
            assert sorted(vaults) == ["code", "finance"]

    def test_setup_error_when_binary_missing(self, tmp_path):
        from octane.security.vault import VaultManager, VaultSetupError
        from octane.security import vault as vault_mod

        vault_path = tmp_path / "finance.enc"
        vault_path.write_bytes(b"x")

        vm = VaultManager.__new__(VaultManager)
        vm.__init__()

        with patch("octane.security.vault._VAULT_DIR", tmp_path), \
             patch.object(vault_mod, "_auth_bin_available", return_value=False):
            with pytest.raises(VaultSetupError):
                vm.read("finance", "key")
