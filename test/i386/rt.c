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
