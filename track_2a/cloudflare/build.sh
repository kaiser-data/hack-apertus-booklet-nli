#!/bin/sh
# Assemble the static site: the results explorer at / and the live demo at /demo/.
#   sh cloudflare/build.sh && cd cloudflare && npx wrangler pages deploy public --project-name booklet-nli
set -e
cd "$(dirname "$0")"
rm -rf public && mkdir -p public/demo
cp ../docs/results.html public/index.html
cp ../modal/demo.html public/demo/index.html
