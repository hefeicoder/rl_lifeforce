#!/usr/bin/env bash
#
# setup_stable_retro.sh — build stable-retro natively on Apple Silicon (arm64).
#
# Why this script exists: on Apple Silicon there is no working prebuilt
# stable-retro wheel (the published one is a mislabeled x86_64 binary), and a
# naive source build fails because some vendored cores (Genesis/PCE) bundle
# pre-C23 K&R zlib that modern Apple clang rejects. We only need the NES core
# (fceumm) for Life Force, so we build that one and skip the rest.
#
# See docs/macos_arm64_build.md for the full explanation.
#
# Usage:
#   source your project venv first, then:
#     ./scripts/setup_stable_retro.sh
#
#   Override the checkout location with SR_SRC=/path ./scripts/setup_stable_retro.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SR_SRC="${SR_SRC:-$REPO_ROOT/third_party/stable-retro}"
SR_REPO="https://github.com/Farama-Foundation/stable-retro.git"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Preconditions -----------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || die "This script is for macOS. On Linux, 'pip install stable-retro' just works."
[ "$(uname -m)" = "arm64" ]  || die "This script targets Apple Silicon (arm64)."
command -v brew >/dev/null   || die "Homebrew not found. Install it from https://brew.sh"

if ! python -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' 2>/dev/null; then
  printf '\033[1;33mWarning:\033[0m no virtualenv appears active; stable-retro will install into the current Python.\n'
  read -r -p "Continue anyway? [y/N] " ans
  [ "${ans:-N}" = "y" ] || die "Aborted. Activate a venv and re-run."
fi

# --- 1. System dependencies --------------------------------------------------
# lua@5.3 was removed from Homebrew; lua@5.4 works (Life Force uses no Lua).
log "Installing build dependencies via Homebrew"
brew install cmake pkg-config lua@5.4 libzip capnp

# --- 2. Clone stable-retro source -------------------------------------------
if [ ! -d "$SR_SRC/.git" ]; then
  log "Cloning stable-retro into $SR_SRC"
  git clone --depth 1 "$SR_REPO" "$SR_SRC"
else
  log "stable-retro source already present at $SR_SRC"
fi

# --- 3. Patch CMakeLists: build ONLY the NES core ---------------------------
# Comment out every add_core(...) except add_core(nes fceumm). BSD sed is
# idempotent here: already-commented lines start with '#' and won't re-match.
CML="$SR_SRC/CMakeLists.txt"
log "Restricting build to the NES core (fceumm)"
sed -i '' -E \
  's/^([[:space:]]*)add_core\((snes|genesis|atari2600|gb|gba|pce|32x|saturn|ds|fbneo|n64|flycast)/\1# add_core(\2/' \
  "$CML"
echo "  remaining active cores:"
grep -nE '^[[:space:]]*add_core\(' "$CML" | sed 's/^/    /'

# --- 4. Build natively -------------------------------------------------------
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
log "SDKROOT=$SDKROOT"
log "Building stable-retro (this compiles the NES core; takes a few minutes)"
( cd "$SR_SRC" && pip install -e . )

# --- 5. Verify ---------------------------------------------------------------
log "Verifying install"
python - <<'PY'
import stable_retro as retro
assert "LifeForce-Nes-v0" in retro.data.list_games(), "LifeForce integration missing"
print("OK: stable-retro imports and LifeForce-Nes-v0 is recognized")
print("Next: import your legally-owned ROM with:")
print("  python -m retro.import /path/to/your/roms/")
PY

log "Done. stable-retro is built and importable."
