#!/usr/bin/env bash
# build.sh — Compile octane-auth Swift binary and install to ~/.octane/bin/
# Run from the octane-auth/ directory or from the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.octane/bin"
BINARY_NAME="octane-auth"

echo "Building $BINARY_NAME..."
cd "$SCRIPT_DIR"

swift build -c release 2>&1

BUILD_PATH="$SCRIPT_DIR/.build/release/$BINARY_NAME"
if [[ ! -f "$BUILD_PATH" ]]; then
    echo "Build failed — binary not found at $BUILD_PATH"
    exit 1
fi

mkdir -p "$INSTALL_DIR"
cp "$BUILD_PATH" "$INSTALL_DIR/$BINARY_NAME"
chmod 755 "$INSTALL_DIR/$BINARY_NAME"

echo "Installed to $INSTALL_DIR/$BINARY_NAME"
echo ""
echo "Verify: $INSTALL_DIR/$BINARY_NAME list-vaults"
