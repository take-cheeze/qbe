#!/bin/sh
# Runs the i386 tests (test/i386/*.ssa) for the i386 and
# i386_regparm3 targets: each test's driver (the "# >>> driver"
# section, C) is compiled by gcc -m32 (with -mregparm=3 for the
# regparm target), linked with the QBE output and the
# freestanding runtime test/i386/rt.c (no 32-bit libc needed),
# and run; it must exit with 0 and print the "# >>> output"
# section if there is one.  A 64-bit Linux kernel runs the
# static i386 binaries natively.

dir=`dirname "$0"`
bin=${bin:-$dir/../qbe}
tdir=$dir/../test/i386
tmp=${TMPDIR:-/tmp}/qbe.i386.$$
cc="${CC32:-gcc} -m32 -O1 -ffreestanding -fno-builtin -nostdlib -static -fno-pic -no-pie -fno-stack-protector -I$tdir"

extract() {
	awk "
		/^# >>> $1/ { p = 1; next }
		/^# <<</ { p = 0 }
		p
	" $2 | sed -e 's/# //' -e 's/#$//'
}

once() {
	t=$1 tgt=$2
	case $tgt in
	i386_regparm3) rp=-mregparm=3 ;;
	*) rp= ;;
	esac
	printf "%-14s %-35s" "$tgt" "$(basename $t)..."
	if ! $bin -t $tgt -o $tmp.s $t; then
		echo "[qbe fail]"
		return 1
	fi
	extract driver $t > $tmp.c
	extract output $t > $tmp.out
	if ! $cc $rp -o $tmp.exe $tmp.c $tdir/rt.c $tmp.s; then
		echo "[cc fail]"
		return 1
	fi
	if test -s $tmp.out; then
		$tmp.exe | diff -u - $tmp.out
		ret=$?
		reason=output
	else
		$tmp.exe
		ret=$?
		reason="returned $ret"
	fi
	if test $ret -ne 0; then
		echo "[$reason fail]"
		return 1
	fi
	echo "[ok]"
}

if ! ${CC32:-gcc} -m32 -c -x c -o /dev/null /dev/null 2>/dev/null; then
	echo "no gcc -m32, i386 tests skipped"
	exit 77
fi

# err-*.ssa: QBE must refuse the test with the error of its
# first line ("# error: <text>")
refuse() {
	t=$1 tgt=$2
	printf "%-14s %-35s" "$tgt" "$(basename $t)..."
	want=`sed -n -e '1s/^# error: //p' $t`
	if $bin -t $tgt -o $tmp.s $t 2>$tmp.err; then
		echo "[not refused]"
		return 1
	fi
	if ! grep -F -e "$want" $tmp.err >/dev/null; then
		echo "[wrong error]"
		cat $tmp.err
		return 1
	fi
	echo "[ok]"
}

fail=0
count=0
test $# -eq 0 && set -- $tdir/*.ssa
for t in "$@"; do
	case `basename $t` in
	err-*)
		refuse $t i386
		fail=`expr $fail + $?`
		count=`expr $count + 1`
		continue
		;;
	esac
	for tgt in i386 i386_regparm3; do
		if sed -e 1q $t | grep "skip.* $tgt\( .*\)\?$" >/dev/null; then
			continue
		fi
		once $t $tgt
		fail=`expr $fail + $?`
		count=`expr $count + 1`
	done
done
rm -f $tmp.s $tmp.c $tmp.out $tmp.exe $tmp.err
if test $fail -ge 1; then
	echo
	echo "$fail of $count tests failed!"
else
	echo
	echo "All is fine!"
fi
exit $fail
