#!/usr/bin/env sh
set -eu

SYSTEM_INSTALL=0
INSTALL_SERVICE=0
NO_BUILD=0
PREFIX=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --system)
      SYSTEM_INSTALL=1
      ;;
    --install-service)
      INSTALL_SERVICE=1
      ;;
    --no-build)
      NO_BUILD=1
      ;;
    --prefix)
      shift
      PREFIX="${1:-}"
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BINARY="$REPO_ROOT/target/release/hajimi-claw"
SERVICE_TEMPLATE="$REPO_ROOT/packaging/systemd/hajimi-claw.service"

if [ -z "$PREFIX" ]; then
  if [ "$SYSTEM_INSTALL" -eq 1 ]; then
    PREFIX="/usr/local"
  else
    PREFIX="${HOME}/.local"
  fi
fi

BIN_DIR="$PREFIX/bin"
SHARE_DIR="$PREFIX/share/hajimi-claw"
CONFIG_EXAMPLE_TARGET="$SHARE_DIR/config.example.toml"

if [ "$NO_BUILD" -ne 1 ]; then
  echo "Building release binary..."
  cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml"
fi

if [ ! -f "$BINARY" ]; then
  echo "Release binary not found at $BINARY" >&2
  exit 1
fi

mkdir -p "$BIN_DIR" "$SHARE_DIR"
install -m 0755 "$BINARY" "$BIN_DIR/hajimi-claw"
ln -sf "$BIN_DIR/hajimi-claw" "$BIN_DIR/hajimi"
install -m 0644 "$REPO_ROOT/config.example.toml" "$CONFIG_EXAMPLE_TARGET"

if [ "$INSTALL_SERVICE" -eq 1 ]; then
  if [ "$SYSTEM_INSTALL" -ne 1 ]; then
    echo "--install-service requires --system" >&2
    exit 1
  fi
  if [ ! -d /etc/systemd/system ]; then
    echo "systemd directory /etc/systemd/system not found" >&2
    exit 1
  fi
  sed \
    -e "s|@BINARY@|$BIN_DIR/hajimi-claw|g" \
    -e "s|@WORKDIR@|$SHARE_DIR|g" \
    "$SERVICE_TEMPLATE" > /etc/systemd/system/hajimi-claw.service
  echo "Installed systemd service to /etc/systemd/system/hajimi-claw.service"
fi

echo ""
echo "Installed hajimi-claw to $BIN_DIR/hajimi-claw"
echo "Installed command alias to $BIN_DIR/hajimi"
echo "Config example copied to $CONFIG_EXAMPLE_TARGET"
echo "Run \`hajimi onboard\` to create your config, then use \`hajimi\`."

if [ "$SYSTEM_INSTALL" -eq 0 ]; then
  echo "Ensure $BIN_DIR is in PATH."
fi
