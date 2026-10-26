#!/usr/bin/env bash
# Compile the Apple Vision OCR helper and prove it reads a known page.
# Output: tools/macos-vision-ocr/visionocr  (gitignored binary)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
if ! command -v swiftc >/dev/null; then
  echo "swiftc not found (install Xcode command line tools)" >&2
  exit 1
fi
swiftc -O -o "$DIR/visionocr" "$DIR/visionocr.swift" \
  -framework Vision -framework PDFKit -framework AppKit -framework ImageIO -framework CoreText
"$DIR/visionocr" --self-test
echo "built $DIR/visionocr"
