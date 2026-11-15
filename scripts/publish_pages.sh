#!/usr/bin/env bash
# Publica el artículo como sitio estático en la rama `gh-pages`.
#
# La fuente de verdad es `index.html` + `assets/site/` en la rama principal.
# Este script arma el árbol del sitio y lo publica como commit en la rama
# `gh-pages`, de modo que el sitio pueda servirse desde esa rama
# (Settings → Pages → Deploy from a branch → gh-pages / root) sin duplicar
# el contenido a mano ni tocar el índice de la rama principal.
#
# Uso: bash scripts/publish_pages.sh [--no-push]

set -euo pipefail

BRANCH="gh-pages"
PUSH=1
[[ "${1:-}" == "--no-push" ]] && PUSH=0

REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_DIR="$(git rev-parse --absolute-git-dir)"
cd "$REPO_ROOT"

BUILD_DIR="$(mktemp -d)"
TMP_INDEX="$(mktemp -d)/index"
TMP_BRANCH="publish/${BRANCH}-$(date +%s)"

cleanup() {
  git branch -D "$TMP_BRANCH" >/dev/null 2>&1 || true
  rm -rf "$BUILD_DIR" "$(dirname "$TMP_INDEX")"
}
trap cleanup EXIT

echo "→ Componiendo el sitio"
mkdir -p "$BUILD_DIR/assets"
cp index.html "$BUILD_DIR/index.html"
cp -R assets/site "$BUILD_DIR/assets/site"
: > "$BUILD_DIR/.nojekyll"

echo "→ Armando el árbol de objetos"
GIT_INDEX_FILE="$TMP_INDEX" git read-tree --empty
(
  cd "$BUILD_DIR"
  GIT_INDEX_FILE="$TMP_INDEX" git --git-dir="$GIT_DIR" --work-tree="$BUILD_DIR" add -A .
)
TREE="$(GIT_INDEX_FILE="$TMP_INDEX" git write-tree)"

# Se conserva el historial de la rama si ya existe en el remoto.
PARENT="$(git rev-parse --verify --quiet "refs/remotes/origin/${BRANCH}" || true)"
COMMIT="$(printf '%s\n' "docs: publish fraud detection article on GitHub Pages" | git commit-tree "$TREE" ${PARENT:+-p "$PARENT"})"

if [[ "$PUSH" == "1" ]]; then
  echo "→ Publicando en origin/$BRANCH"
  git push origin "${COMMIT}:refs/heads/${BRANCH}"
  echo "✓ Rama $BRANCH actualizada en origin"
else
  git branch "$TMP_BRANCH" "$COMMIT" >/dev/null 2>&1 || true
  echo "✓ Commit local creado ($COMMIT), no publicado"
  git show --stat --oneline "$COMMIT" | head -20
fi
