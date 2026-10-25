/* A freestanding runtime for the i386 tests (no 32-bit libc is
 * needed): _start calls main and exits with its result through
 * the Linux int $0x80 system calls; output goes to stdout.
 */

int main(int, char **);

static int
sys(int n, int a, int b, int c)
{
	int r;

	__asm__ volatile ("int $0x80" : "=a"(r) : "a"(n), "b"(a), "c"(b), "d"(c) : "memory");
	return r;
}

void
rt_exit(int c)
{
	sys(1, c, 0, 0);
	for (;;);
}

void
rt_puts(const char *s)
{
	int n;

	for (n=0; s[n]; n++)
		;
	sys(4, 1, (int)s, n);
}

void
rt_putd(int v)
{
	char buf[16], *p;
	unsigned u;

	p = &buf[15];
	*p = 0;
	u = v < 0 ? -(unsigned)v : (unsigned)v;
	do
		*--p = '0' + u % 10;
	while (u /= 10);
	if (v < 0)
		*--p = '-';
	rt_puts(p);
}

static int nfail;

void
rt_check(int ok, const char *what, int got, int want)
{
	if (ok)
		return;
	nfail++;
	rt_puts("FAIL ");
	rt_puts(what);
	rt_puts(": got ");
	rt_putd(got);
	rt_puts(", want ");
	rt_putd(want);
	rt_puts("\n");
}

int
rt_failed(void)
{
	return nfail;
}

#define EQ(what, got, want) rt_check((got) == (want), what, got, want)

void *
memcpy(void *d, const void *s, unsigned n)
{
	char *dp = d;
	const char *sp = s;

	while (n--)
		*dp++ = *sp++;
	return d;
}

void *
memset(void *d, int c, unsigned n)
{
	char *dp = d;

	while (n--)
		*dp++ = c;
	return d;
}

__asm__(
	".text\n"
	".globl _start\n"
	"_start:\n"
	"\txorl %ebp, %ebp\n"
	"\tandl $-16, %esp\n"
	"\tsubl $8, %esp\n"
	"\tpushl $0\n"
	"\tpushl $0\n"
	"\tcall rt_main\n"
	"\tud2\n"
);

void
rt_main(void)
{
	int r;

	r = main(0, 0);
	if (r == 0 && nfail)
		r = 1;
	rt_exit(r);
}

/* libgcc's 64-bit divisions (there is no 32-bit libgcc here), by
 * shifts and subtractions so that gcc makes no call itself */
static unsigned long long
udivmod(unsigned long long n, unsigned long long d, unsigned long long *r)
{
	unsigned long long q = 0;
	int i;

	if (d == 0)
		__builtin_trap();
	for (i = 63; i >= 0; i--) {
		if ((d << i >> i) != d)
			continue;
		if (n >= d << i) {
			n -= d << i;
			q |= 1ull << i;
		}
	}
	*r = n;
	return q;
}

unsigned long long __udivdi3(unsigned long long n, unsigned long long d) { unsigned long long r; return udivmod(n, d, &r); }
unsigned long long __umoddi3(unsigned long long n, unsigned long long d) { unsigned long long r; udivmod(n, d, &r); return r; }

long long
__divdi3(long long n, long long d)
{
	unsigned long long r, q;

	q = udivmod(n < 0 ? -(unsigned long long)n : n, d < 0 ? -(unsigned long long)d : d, &r);
	return (n < 0) != (d < 0) ? -q : q;
}

long long
__moddi3(long long n, long long d)
{
	unsigned long long r;

	udivmod(n < 0 ? -(unsigned long long)n : n, d < 0 ? -(unsigned long long)d : d, &r);
	return n < 0 ? -r : r;
}

void
rt_putx64(unsigned long long v)
{
	char buf[17];
	int i;

	for (i = 15; i >= 0; i--, v >>= 4)
		buf[i] = "0123456789abcdef"[v & 15];
	buf[16] = 0;
	rt_puts(buf);
}
