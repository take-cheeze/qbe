#!/bin/sh

# The amd64 test suite with the 256-bit loop vectorizer on (qbe -V -Y): the loops that qualify
# (int32 and float32, element-wise, see vec.c) use AVX2 ymm vectors. Needs an AVX2 host to run.

dir=`dirname "$0"`
tmp=`mktemp -d`
w=$tmp/qbe-VY
printf '#!/bin/sh\nexec "%s/../qbe" -V -Y "$@"\n' "`cd "$dir" && pwd`" > "$w"
chmod +x "$w"
bin="$w" TARGET=${TARGET:-x86_64} sh "$dir/test.sh" "$@"
status=$?
rm -rf "$tmp"
exit $status
