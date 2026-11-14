#!/bin/sh

# The test suite with the vector loop vectorizer on (qbe -V -F -Y): the loops that qualify (int32 and
# float32, element-wise, see vec.c) use AVX2 ymm vectors on amd64 (needs an AVX2 host), and RVV
# v registers on rv64 (run it with TARGET=rv64, under qemu-riscv64).

dir=`dirname "$0"`
tmp=`mktemp -d`
w=$tmp/qbe-VY
printf '#!/bin/sh\nexec "%s/../qbe" -V -F -Y "$@"\n' "`cd "$dir" && pwd`" > "$w"
chmod +x "$w"
bin="$w" TARGET=${TARGET:-x86_64} sh "$dir/test.sh" "$@"
status=$?
rm -rf "$tmp"
exit $status
