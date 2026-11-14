#!/bin/sh

# The amd64 test suite with the loop vectorizer on (qbe -V), so that the loops of every test
# go through vec.c; the plain suite (tools/test.sh) leaves the vectorizer off.

dir=`dirname "$0"`
tmp=`mktemp -d`
w=$tmp/qbe-V
printf '#!/bin/sh\nexec "%s/../qbe" -V -F "$@"\n' "`cd "$dir" && pwd`" > "$w"
chmod +x "$w"
bin="$w" TARGET=x86_64 sh "$dir/test.sh" "$@"
status=$?
rm -rf "$tmp"
exit $status
