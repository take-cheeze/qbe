#include "all.h"
#include "config.h"
#include <ctype.h>
#include <getopt.h>

Target T;

int nofp;
int nounwind;
static int code16;   /* -m16: i386 code run in real mode, as gcc -m16 makes it (.code16gcc) */
char debug['Z'+1] = {
	['P'] = 0, /* parsing */
	['M'] = 0, /* memory optimization */
	['N'] = 0, /* ssa construction */
	['C'] = 0, /* copy elimination */
	['G'] = 0, /* gvn/gcm */
	['K'] = 0, /* if-conversion */
	['A'] = 0, /* abi lowering */
	['I'] = 0, /* instruction selection */
	['L'] = 0, /* liveness */
	['S'] = 0, /* spilling */
	['R'] = 0, /* reg. allocation */
};

extern Target T_amd64_sysv;
extern Target T_amd64_apple;
extern Target T_amd64_win;
extern Target T_arm64;
extern Target T_arm64_apple;
extern Target T_rv64;
extern Target T_i386;
extern Target T_i386_regparm3;

static Target *tlist[] = {
	&T_amd64_sysv,
	&T_amd64_apple,
	&T_amd64_win,
	&T_arm64,
	&T_arm64_apple,
	&T_rv64,
	&T_i386,
	&T_i386_regparm3,
	0
};
static FILE *outf;
static int dbg;

static void
data(Dat *d)
{
	if (dbg)
		return;
	emitdat(d, outf);
	if (d->type == DEnd) {
		fputs("/* end data */\n\n", outf);
		freeall();
	}
}

static void
func(Fn *fn)
{
	uint n;

	if (dbg)
		fprintf(stderr, "**** Function %s ****", fn->name);
	if (debug['P']) {
		fprintf(stderr, "\n> After parsing:\n");
		printfn(fn, stderr);
	}
	T.abi0(fn);
	fillcfg(fn);
	filluse(fn);
	promote(fn);
	filluse(fn);
	ssa(fn);
	filluse(fn);
	ssacheck(fn);
	fillalias(fn);
	loadopt(fn);
	filluse(fn);
	fillalias(fn);
	coalesce(fn);
	filluse(fn);
	filldom(fn);
	ssacheck(fn);
	gvn(fn);
	fillcfg(fn);
	simplcfg(fn);
	filluse(fn);
	filldom(fn);
	gcm(fn);
	filluse(fn);
	ssacheck(fn);
	if (T.cansel) {
		ifconvert(fn);
		fillcfg(fn);
		filluse(fn);
		filldom(fn);
		ssacheck(fn);
	}
	T.abi1(fn);
	simpl(fn);
	fillcfg(fn);
	filluse(fn);
	T.isel(fn);
	fillcfg(fn);
	filllive(fn);
	fillloop(fn);
	fillcost(fn);
	spill(fn);
	rega(fn);
	fillcfg(fn);
	simpljmp(fn);
	fillcfg(fn);
	assert(fn->rpo[0] == fn->start);
	for (n=0;; n++)
		if (n == fn->nblk-1) {
			fn->rpo[n]->link = 0;
			break;
		} else
			fn->rpo[n]->link = fn->rpo[n+1];
	if (!dbg) {
		T.emitfn(fn, outf);
		fprintf(outf, "/* end function %s */\n\n", fn->name);
	} else
		fprintf(stderr, "\n");
	freeall();
}

static void
dbgfile(char *fn)
{
	emitdbgfile(fn, outf);
}

int
main(int ac, char *av[])
{
	Target **t;
	FILE *inf, *hf;
	char *f, *sep;
	int c;

	T = Deftgt;
	outf = stdout;
	while ((c = getopt(ac, av, "hd:m:no:t:U")) != -1)
		switch (c) {
		case 'd':
			for (; *optarg; optarg++)
				if (isalpha(*optarg)) {
					debug[toupper(*optarg)] = 1;
					dbg = 1;
				}
			break;
		case 'n':
			nofp = 1;
			break;
		case 'm':
			if (strcmp(optarg, "16") != 0) {
				fprintf(stderr, "unknown option -m%s\n", optarg);
				exit(1);
			}
			code16 = 1;
			break;
		case 'U':
			nounwind = 1;
			break;
		case 'o':
			if (strcmp(optarg, "-") != 0) {
				outf = fopen(optarg, "w");
				if (!outf) {
					fprintf(stderr, "cannot open '%s'\n", optarg);
					exit(1);
				}
			}
			break;
		case 't':
			if (strcmp(optarg, "?") == 0) {
				puts(T.name);
				exit(0);
			}
			for (t=tlist;; t++) {
				if (!*t) {
					fprintf(stderr, "unknown target '%s'\n", optarg);
					exit(1);
				}
				if (strcmp(optarg, (*t)->name) == 0) {
					T = **t;
					break;
				}
			}
			break;
		case 'h':
		default:
			hf = c != 'h' ? stderr : stdout;
			fprintf(hf, "%s [OPTIONS] {file.ssa, -}\n", av[0]);
			fprintf(hf, "\t%-11s prints this help\n", "-h");
			fprintf(hf, "\t%-11s output to file\n", "-o file");
			fprintf(hf, "\t%-11s no floating-point registers in integer code\n", "-n");
			fprintf(hf, "\t%-11s no unwind tables\n", "-U");
			fprintf(hf, "\t%-11s i386 code for real mode (.code16gcc, gcc's -m16)\n", "-m16");
			fprintf(hf, "\t%-11s generate for a target among:\n", "-t <target>");
			fprintf(hf, "\t%-11s ", "");
			for (t=tlist, sep=""; *t; t++, sep=", ") {
				fprintf(hf, "%s%s", sep, (*t)->name);
				if (*t == &Deftgt)
					fputs(" (default)", hf);
			}
			fprintf(hf, "\n");
			fprintf(hf, "\t%-11s dump debug information\n", "-d <flags>");
			exit(c != 'h');
		}

	if (code16) {
		if (!T.ptr32) {
			fprintf(stderr, "-m16 needs an i386 target\n");
			exit(1);
		}
		/* gas adds the operand and address size prefixes */
		fputs(".code16gcc\n", outf);
	}

	do {
		f = av[optind];
		if (!f || strcmp(f, "-") == 0) {
			inf = stdin;
			f = "-";
		} else {
			inf = fopen(f, "r");
			if (!inf) {
				fprintf(stderr, "cannot open '%s'\n", f);
				exit(1);
			}
		}
		parse(inf, f, dbgfile, data, func);
		fclose(inf);
	} while (++optind < ac);

	if (!dbg)
		T.emitfin(outf);

	exit(0);
}
