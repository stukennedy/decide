#!/bin/sh
# Installs `decide`: fast local decisions on Apple Silicon (Qwen3.5-4B on MLX).
#   curl -fsSL https://decide.run/install | sh
# Environment overrides:
#   DECIDE_SOURCE      what to install (default: the GitHub repo; a local path works for testing)
#   DECIDE_SKIP_SETUP  set to 1 to skip the ~9 GB model download and test decision
set -eu

DECIDE_SOURCE="${DECIDE_SOURCE:-git+https://github.com/stukennedy/decide}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
fail() { printf 'decide install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] ||
  fail "decide needs an Apple Silicon Mac (M1 or later); it runs the model on MLX."

# Git is needed to fetch the pinned MLX-LM source.
if ! xcode-select -p >/dev/null 2>&1; then
  fail "Git isn't available. Run 'xcode-select --install', then re-run this installer."
fi
git --version >/dev/null 2>&1 ||
  fail "git won't run. If it mentions the Xcode licence, run 'sudo xcodebuild -license accept' and retry."

# uv manages an isolated Python 3.12 environment for the tool.
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
  else
    say "Installing uv (Python package manager from astral.sh)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
  fi
fi

say "Installing decide..."
uv tool install --force --python 3.12 "$DECIDE_SOURCE"

BIN_DIR="$(uv tool dir --bin)"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) uv tool update-shell >/dev/null 2>&1 || true
     PATH="$BIN_DIR:$PATH"
     NEW_SHELL=1 ;;
esac

if [ "${DECIDE_SKIP_SETUP:-0}" != "1" ]; then
  say "Downloading the model and running a test decision..."
  "$BIN_DIR/decide" setup
fi

say "Done."
echo "  decide ask \"Customer: I was charged twice.\" \"Which team?\" billing shipping tech"
echo "  decide --help"
[ "${NEW_SHELL:-0}" = "1" ] && echo "Open a new terminal (or run: export PATH=\"$BIN_DIR:\$PATH\") to use 'decide'."
exit 0
