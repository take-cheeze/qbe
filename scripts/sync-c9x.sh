#!/usr/bin/env bash
# Mirror https://c9x.me/git/qbe.git (git "dumb" HTTP, served by OpenBSD httpd) into
# the bare repo given as $1 (created if missing): refs/heads/* and refs/tags/*.
#
# `git fetch`/`git clone` over that transport is unreliable: the server resets
# connections under the walker's many small requests, and git can leave truncated
# loose objects behind. So download the packs and any missing loose objects with
# curl (retried, each object verified) and set the refs ourselves.
set -euo pipefail
dest=${1:?usage: sync-c9x.sh BARE_REPO_DIR}
base=${UPSTREAM:-https://c9x.me/git/qbe.git}
curl_() { curl -fsS --retry 8 --retry-all-errors --retry-delay 2 "$@"; }

[ -d "$dest" ] || git init -q --bare "$dest"
cd "$dest"

# 1. Packs (the bulk of the history).
curl_ "$base/objects/info/packs" | awk '/^P/{print $2}' | while read -r p; do
  for ext in idx pack; do
    f=objects/pack/${p%.pack}.$ext
    [ -s "$f" ] || { curl_ -o "$f.tmp" "$base/$f" && mv "$f.tmp" "$f"; }
  done
done

# 2. Refs advertised by the server.
curl_ "$base/info/refs" | grep -E $'\trefs/(heads|tags)/[^^]*$' > "$PWD/.refs"

fetch_loose() {  # fetch_loose SHA: download one loose object and verify it
  local o=$1 d=objects/${1:0:2}
  mkdir -p "$d"
  for _ in 1 2 3 4 5; do
    if curl_ -o "$d/${o:2}.tmp" "$base/$d/${o:2}" \
       && mv "$d/${o:2}.tmp" "$d/${o:2}" \
       && git cat-file -p "$o" > /dev/null 2>&1; then return 0; fi
    rm -f "$d/${o:2}" "$d/${o:2}.tmp"; sleep 2
  done
  echo "failed to fetch object $o" >&2; return 1
}

# 3. Loose objects that aren't in the packs: tips first, then whatever is missing.
while IFS=$'\t' read -r sha ref; do
  git cat-file -e "$sha" 2>/dev/null || fetch_loose "$sha"
done < .refs

export -f fetch_loose curl_; export base
for round in $(seq 1 200); do
  missing=$(git rev-list --objects --missing=print $(cut -f1 .refs | sort -u) 2>/dev/null \
            | sed -n 's/^?//p' || true)
  [ -n "$missing" ] || break
  # The server resets connections under heavy load, so keep parallelism modest.
  printf '%s\n' $missing | xargs -P 6 -n 1 bash -c 'fetch_loose "$0"'
done
git rev-list --objects $(cut -f1 .refs | sort -u) > /dev/null  # fails if anything is still missing

# 4. Point refs at the fetched tips (peeled "^{}" lines were filtered out above).
while IFS=$'\t' read -r sha ref; do git update-ref "$ref" "$sha"; done < .refs
rm -f .refs
git fsck --connectivity-only 2>&1 | tail -3
echo "synced: $(git rev-parse --short refs/heads/master) ($(git tag | wc -l) tags)"
