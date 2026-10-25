void rt_puts(const char *);
void rt_putd(int);
void rt_check(int, const char *, int, int);
#define EQ(what, got, want) rt_check((got) == (want), what, got, want)
void rt_putx64(unsigned long long);
