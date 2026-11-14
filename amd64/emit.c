#include "all.h"


typedef struct E E;

struct E {
	FILE *f;
	Fn *fn;
	int fp;
	uint64_t fsz;
	int nclob;
	int tail;	/* the call is emitted as a jump (a tail call, sysv.c tailcallok) */
};

/* A tail call's target stays in its register through the epilogue, which restores the callee-saved registers and the
   frame pointer: a target in one of them, or a spilled one (in the frame), is a plain call. */
static int
tailtarget(Ins *i)
{
	if (rtype(i->arg[0]) == RCon)
		return 1;
	return isreg(i->arg[0]) && i->arg[0].val != RBP && i->arg[0].val != RSP
		&& i->arg[0].val != RBX && i->arg[0].val != R12 && i->arg[0].val != R13
		&& i->arg[0].val != R14 && i->arg[0].val != R15;
}

#define CMP(X) \
	X(Ciule,      "be", "a") \
	X(Ciult,      "b", "ae") \
	X(Cisle,      "le", "g") \
	X(Cislt,      "l", "ge") \
	X(Cisgt,      "g", "le") \
	X(Cisge,      "ge", "l") \
	X(Ciugt,      "a", "be") \
	X(Ciuge,      "ae", "b") \
	X(Cieq,       "z", "nz") \
	X(Cine,       "nz", "z") \
	X(NCmpI+Cfle, "?" , "?") \
	X(NCmpI+Cflt, "?",  "?") \
	X(NCmpI+Cfgt, "a", "be") \
	X(NCmpI+Cfge, "ae", "b") \
	X(NCmpI+Cfo,  "np", "p") \
	X(NCmpI+Cfuo, "p", "np")

enum {
	SLong = 0,
	SWord = 1,
	SShort = 2,
	SByte = 3,

	Ki = -1, /* matches Kw and Kl */
	Ka = -2, /* matches all classes */
};

/* Instruction format strings:
 *
 * if the format string starts with -, the instruction
 * is assumed to be 3-address and is put in 2-address
 * mode using an extra mov if necessary
 *
 * if the format string starts with +, the same as the
 * above applies, but commutativity is also assumed
 *
 * %k  is used to set the class of the instruction,
 *     it'll expand to "l", "q", "ss", "sd", depending
 *     on the instruction class
 * %0  designates the first argument
 * %1  designates the second argument
 * %=  designates the result
 *
 * if %k is not used, a prefix to 0, 1, or = must be
 * added, it can be:
 *   M - memory reference
 *   L - long  (64 bits)
 *   W - word  (32 bits)
 *   H - short (16 bits)
 *   B - byte  (8 bits)
 *   S - single precision float
 *   D - double precision float
 */
static struct {
	short op;
	short cls;
	char *fmt;
} omap[] = {
	{ Oadd,     Ka, "+add%k %1, %=" },
	{ Osub,     Ka, "-sub%k %1, %=" },
	{ Oand,     Ki, "+and%k %1, %=" },
	{ Oor,      Ki, "+or%k %1, %=" },
	{ Oxor,     Ki, "+xor%k %1, %=" },
	{ Osar,     Ki, "-sar%k %B1, %=" },
	{ Oshr,     Ki, "-shr%k %B1, %=" },
	{ Oshl,     Ki, "-shl%k %B1, %=" },
	{ Omul,     Ki, "+imul%k %1, %=" },
	{ Omul,     Ks, "+mulss %1, %=" },
	{ Omul,     Kd, "+mulsd %1, %=" },
	{ Odiv,     Ka, "-div%k %1, %=" },
	{ Ostorel,  Ka, "movq %L0, %M1" },
	{ Ostorew,  Ka, "movl %W0, %M1" },
	{ Ostoreh,  Ka, "movw %H0, %M1" },
	{ Ostoreb,  Ka, "movb %B0, %M1" },
	{ Ostores,  Ka, "movss %S0, %M1" },
	{ Ostored,  Ka, "movsd %D0, %M1" },
	{ Ovecload,  Ka, "movups %M0, %=" },
	{ Ovecstore, Ka, "movups %0, %M1" },
	{ Ovecaddw,  Kd, "+paddd %1, %=" },
	{ Ovecsubw,  Kd, "-psubd %1, %=" },
	{ Ovecandw,  Kd, "+pand %1, %=" },
	{ Ovecorw,   Kd, "+por %1, %=" },
	{ Ovecxorw,  Kd, "+pxor %1, %=" },
	{ Ovecadds,  Kd, "+addps %1, %=" },
	{ Ovecsubs,  Kd, "-subps %1, %=" },
	{ Ovecmuls,  Kd, "+mulps %1, %=" },
	{ Ovecdivs,  Kd, "-divps %1, %=" },
	{ Ovecaddd,  Kd, "+addpd %1, %=" },
	{ Ovecsubd,  Kd, "-subpd %1, %=" },
	{ Ovecmuld,  Kd, "+mulpd %1, %=" },
	{ Ovecdivd,  Kd, "-divpd %1, %=" },
	{ Ovecmuludq, Kd, "+pmuludq %1, %=" },
	{ Ovecsrlq32, Kd, "-psrlq $32, %=" },
	{ Ovecpack02, Kd, "pshufd $136, %0, %=" },
	{ Ovecunpckl32, Kd, "-punpckldq %1, %=" },
	{ Ovecunpckh32, Kd, "-punpckhdq %1, %=" },
	{ Ovecshufe, Kd, "-shufps $136, %1, %=" },
	{ Ovecshufo, Kd, "-shufps $221, %1, %=" },
	{ Ovecunpcklpd, Kd, "-unpcklpd %1, %=" },
	{ Ovecunpckhpd, Kd, "-unpckhpd %1, %=" },
	{ Ovecceqw, Kd, "+pcmpeqd %1, %=" },
	{ Ovecgtw, Kd, "-pcmpgtd %1, %=" },
	{ Ovecandn, Kd, "-pandn %1, %=" },
	{ Ovecfeqs, Kd, "-cmpps $0, %1, %=" },
	{ Ovecflts, Kd, "-cmpps $1, %1, %=" },
	{ Ovecfles, Kd, "-cmpps $2, %1, %=" },
	{ Ovecfeqd, Kd, "-cmppd $0, %1, %=" },
	{ Ovecfltd, Kd, "-cmppd $1, %1, %=" },
	{ Ovecfled, Kd, "-cmppd $2, %1, %=" },
	{ Ovecload8, Ka, "movq %M0, %=" },
	{ Ovecload256, Ka, "vmovups %M0, %=" },
	{ Ovecstore256, Ka, "vmovups %0, %M1" },
	{ Ovecaddw256, Kd, "vpaddd %1, %0, %=" },
	{ Ovecsubw256, Kd, "vpsubd %1, %0, %=" },
	{ Ovecmulw256, Kd, "vpmulld %1, %0, %=" },
	{ Ovecandw256, Kd, "vpand %1, %0, %=" },
	{ Ovecorw256, Kd, "vpor %1, %0, %=" },
	{ Ovecxorw256, Kd, "vpxor %1, %0, %=" },
	{ Ovecadds256, Kd, "vaddps %1, %0, %=" },
	{ Ovecsubs256, Kd, "vsubps %1, %0, %=" },
	{ Ovecmuls256, Kd, "vmulps %1, %0, %=" },
	{ Ovecdivs256, Kd, "vdivps %1, %0, %=" },
	{ Ovecstore8, Ka, "movq %0, %M1" },
	{ Ovecw2d, Kd, "cvtdq2pd %0, %=" },
	{ Ovecd2w, Kd, "cvttpd2dq %0, %=" },
	{ Oload,    Ka, "mov%k %M0, %=" },
	{ Oloadsw,  Kl, "movslq %M0, %L=" },
	{ Oloadsw,  Kw, "movl %M0, %W=" },
	{ Oloaduw,  Ki, "movl %M0, %W=" },
	{ Oloadsh,  Ki, "movsw%k %M0, %=" },
	{ Oloaduh,  Ki, "movzw%k %M0, %=" },
	{ Oloadsb,  Ki, "movsb%k %M0, %=" },
	{ Oloadub,  Ki, "movzb%k %M0, %=" },
	{ Oextsw,   Kl, "movslq %W0, %L=" },
	{ Oextuw,   Kl, "movl %W0, %W=" },
	{ Oextsh,   Ki, "movsw%k %H0, %=" },
	{ Oextuh,   Ki, "movzw%k %H0, %=" },
	{ Oextsb,   Ki, "movsb%k %B0, %=" },
	{ Oextub,   Ki, "movzb%k %B0, %=" },

	{ Oexts,    Kd, "cvtss2sd %0, %=" },
	{ Otruncd,  Ks, "cvtsd2ss %0, %=" },
	{ Ostosi,   Ki, "cvttss2si%k %0, %=" },
	{ Odtosi,   Ki, "cvttsd2si%k %0, %=" },
	{ Oswtof,   Ka, "cvtsi2%k %W0, %=" },
	{ Osltof,   Ka, "cvtsi2%k %L0, %=" },
	{ Ocast,    Ki, "movq %D0, %L=" },
	{ Ocast,    Ka, "movq %L0, %D=" },

	{ Oaddr,    Ki, "lea%k %M0, %=" },
	{ Oswap,    Ki, "xchg%k %0, %1" },
	{ Olockadd, Ki, "lock xadd%k %0, %M1" },
	{ Olockcmpxchg, Ki, "lock cmpxchg%k %0, %M1" },
	{ Osign,    Kl, "cqto" },
	{ Osign,    Kw, "cltd" },
	{ Oxdiv,    Ki, "div%k %0" },
	{ Oxidiv,   Ki, "idiv%k %0" },
	{ Oxcmp,    Ks, "ucomiss %S0, %S1" },
	{ Oxcmp,    Kd, "ucomisd %D0, %D1" },
	{ Oxcmp,    Ki, "cmp%k %0, %1" },
	{ Oxtest,   Ki, "test%k %0, %1" },
#define X(c, s, _) \
	{ Oflag+c,  Ki, "set" s " %B=\n\tmovzb%k %B=, %=" },
	CMP(X)
#undef X
	{ Oflagfeq, Ki, "setz %B=\n\tmovzb%k %B=, %=" },
	{ Oflagfne, Ki, "setnz %B=\n\tmovzb%k %B=, %=" },
	{ NOp, 0, 0 }
};

static char cmov[][2][16] = {
#define X(c, s0, s1) \
	[c] = { \
		"cmov" s0 " %0, %=", \
		"cmov" s1 " %1, %=", \
	},
	CMP(X)
#undef X
};

static char *rname[][4] = {
	[RAX] = {"rax", "eax", "ax", "al"},
	[RBX] = {"rbx", "ebx", "bx", "bl"},
	[RCX] = {"rcx", "ecx", "cx", "cl"},
	[RDX] = {"rdx", "edx", "dx", "dl"},
	[RSI] = {"rsi", "esi", "si", "sil"},
	[RDI] = {"rdi", "edi", "di", "dil"},
	[RBP] = {"rbp", "ebp", "bp", "bpl"},
	[RSP] = {"rsp", "esp", "sp", "spl"},
	[R8 ] = {"r8" , "r8d", "r8w", "r8b"},
	[R9 ] = {"r9" , "r9d", "r9w", "r9b"},
	[R10] = {"r10", "r10d", "r10w", "r10b"},
	[R11] = {"r11", "r11d", "r11w", "r11b"},
	[R12] = {"r12", "r12d", "r12w", "r12b"},
	[R13] = {"r13", "r13d", "r13w", "r13b"},
	[R14] = {"r14", "r14d", "r14w", "r14b"},
	[R15] = {"r15", "r15d", "r15w", "r15b"},
};


static int
slot(Ref r, E *e)
{
	int s;

	s = rsval(r);
	assert(s <= e->fn->slot);
	/* specific to NAlign == 3 */
	if (s < 0) {
		if (e->fp == RSP)
			return 4*-s - 8 + e->fsz + e->nclob*8;
		else
			return 4*-s;
	}
	else if (e->fp == RSP)
		return 4*s + e->nclob*8;
	else if (e->fn->vararg) {
		if (T.windows)
			return -4 * (e->fn->slot - s);
		else
			return -176 + -4 * (e->fn->slot - s);
	} else
		return -4 * (e->fn->slot - s);
}

static void
emitcon(Con *con, E *e)
{
	char *p, *l;

	switch (con->type) {
	case CAddr:
		l = str(con->sym.id);
		p = l[0] == '"' ? "" : T.assym;
		if (con->sym.type == SThr) {
			assert(!T.apple);
			fprintf(e->f, "%%fs:%s%s@tpoff", p, l);
		} else {
			assert((con->sym.type & ~SExt) == SGlo);
			fprintf(e->f, "%s%s", p, l);
		}
		if (con->bits.i)
			fprintf(e->f, "%+"PRId64, con->bits.i);
		break;
	case CBits:
		fprintf(e->f, "%"PRId64, con->bits.i);
		break;
	default:
		die("unreachable");
	}
}

/* the instruction being emitted works on 256-bit vectors (its xmm registers are ymm ones) */
static int wideins;

static char *
regtoa(int reg, int sz)
{
	static char buf[6];

	assert(reg <= XMM15);
	if (reg >= XMM0) {
		sprintf(buf, wideins ? "ymm%d" : "xmm%d", reg-XMM0);
		return buf;
	} else
		return rname[reg][sz];
}

/* -A (absaddr): the address of a symbol as a sign-extended 32-bit immediate, movq $sym[+off], %reg (gcc's kernel model;
 * QBE's rip-relative lea is the physical address while the kernel runs identity-mapped) */
static void
emitabsaddr(Ins *i, Con *con, E *e)
{
	char *sym = str(con->sym.id);

	fprintf(e->f, "\tmovq $%s%s", sym[0] == '"' ? "" : T.assym, sym);
	if (con->bits.i)
		fprintf(e->f, "%+lld", (long long)con->bits.i);
	fprintf(e->f, ", %%%s\n", regtoa(i->to.val, SLong));
}


static Ref
getarg(char c, Ins *i)
{
	switch (c) {
	case '0':
		return i->arg[0];
	case '1':
		return i->arg[1];
	case '=':
		return i->to;
	default:
		die("invalid arg letter %c", c);
	}
}

static void emitf(char *, Ins *, E *);

/* A jump table (nfcxx, jtab %index, @default, @t0, ...): the index is a register (the isel copy of the index), which the
 * dispatch overwrites: it is zero-extended, checked against the count (the default when not below it), then scaled and
 * added to the table's address, which is held in memory (a relocated quad, so the code needs no relocation and no scratch
 * register). The table is in .data.rel.ro after the function (emitjtabs). */
static void
emitjtab(Blk *b, uint id0, E *e)
{
	FILE *f = e->f;
	int r;

	if (rtype(b->jmp.arg) != RTmp || !isreg(b->jmp.arg))
		die("jtab: the index is not in a register");
	r = b->jmp.arg.val;
	fprintf(f, "\tmovl %%%s, %%%s\n", regtoa(r, SWord), regtoa(r, SWord));
	fprintf(f, "\tcmpl $%u, %%%s\n", b->ntab, regtoa(r, SWord));
	fprintf(f, "\tjae %sbb%d\n", T.asloc, id0+b->s1->id);
	fprintf(f, "\tshlq $3, %%%s\n", regtoa(r, SLong));
	fprintf(f, "\taddq %sjtp%d(%%rip), %%%s\n", T.asloc, id0+b->id, regtoa(r, SLong));
	fprintf(f, "\tjmp *(%%%s)\n", regtoa(r, SLong));
}

/* the tables, after the function (their section is restored to .text) */
static void
emitjtabs(Fn *fn, uint id0, E *e)
{
	Blk *b;
	uint n;

	for (b=fn->start; b; b=b->link)
		if (b->jmp.type == Jjtab) {
			fprintf(e->f, "\t.section .data.rel.ro,\"aw\"\n\t.p2align 3\n");
			fprintf(e->f, "%sjtp%d:\n\t.quad %sjt%d\n", T.asloc, id0+b->id, T.asloc, id0+b->id);
			fprintf(e->f, "%sjt%d:\n", T.asloc, id0+b->id);
			for (n=0; n<b->ntab; n++)
				fprintf(e->f, "\t.quad %sbb%d\n", T.asloc, id0+b->tab[n]->id);
			fputs("\t.text\n", e->f);
		}
}

static void emitins(Ins, E *);

static void
emitcopy(Ref r1, Ref r2, int k, E *e)
{
	Ins icp;
	int w;

	icp.op = Ocopy;
	icp.arg[0] = r2;
	icp.to = r1;
	icp.cls = k;
	w = wideins;	/* the copy sets wideins for itself: the instruction being emitted keeps its own */
	emitins(icp, e);
	wideins = w;
}

static void
emitf(char *s, Ins *i, E *e)
{
	static char clstoa[][3] = {"l", "q", "ss", "sd"};
	char c;
	int sz;
	Ref ref;
	Mem *m;
	Con off;

	switch (*s) {
	case '+':
		if (req(i->arg[1], i->to)) {
			ref = i->arg[0];
			i->arg[0] = i->arg[1];
			i->arg[1] = ref;
		}
		/* fall through */
	case '-':
		assert((!req(i->arg[1], i->to) || req(i->arg[0], i->to)) &&
			"cannot convert to 2-address");
		emitcopy(i->to, i->arg[0], i->cls, e);
		s++;
		break;
	}

	fputc('\t', e->f);
Next:
	while ((c = *s++) != '%')
		if (!c) {
			fputc('\n', e->f);
			return;
		} else
			fputc(c, e->f);
	switch ((c = *s++)) {
	case '%':
		fputc('%', e->f);
		break;
	case 'k':
		fputs(clstoa[i->cls], e->f);
		break;
	case '0':
	case '1':
	case '=':
		sz = KWIDE(i->cls) ? SLong : SWord;
		s--;
		goto Ref;
	case 'D':
	case 'S':
		sz = SLong; /* does not matter for floats */
	Ref:
		c = *s++;
		ref = getarg(c, i);
		switch (rtype(ref)) {
		case RTmp:
			assert(isreg(ref));
			fprintf(e->f, "%%%s", regtoa(ref.val, sz));
			break;
		case RSlot:
			fprintf(e->f, "%d(%%%s)",
				slot(ref, e),
				regtoa(e->fp, SLong)
			);
			break;
		case RMem:
		Mem:
			m = &e->fn->mem[ref.val];
			if (rtype(m->base) == RSlot) {
				off.type = CBits;
				off.bits.i = slot(m->base, e);
				addcon(&m->offset, &off, 1);
				m->base = TMP(e->fp);
			}
			if (m->offset.type != CUndef)
				emitcon(&m->offset, e);
			fputc('(', e->f);
			if (!req(m->base, R))
				fprintf(e->f, "%%%s",
					regtoa(m->base.val, SLong)
				);
			else if (m->offset.type == CAddr)
				fprintf(e->f, "%%rip");
			if (!req(m->index, R))
				fprintf(e->f, ", %%%s, %d",
					regtoa(m->index.val, SLong),
					m->scale
				);
			fputc(')', e->f);
			break;
		case RCon:
			fputc('$', e->f);
			emitcon(&e->fn->con[ref.val], e);
			break;
		default:
			die("unreachable");
		}
		break;
	case 'L':
		sz = SLong;
		goto Ref;
	case 'W':
		sz = SWord;
		goto Ref;
	case 'H':
		sz = SShort;
		goto Ref;
	case 'B':
		sz = SByte;
		goto Ref;
	case 'M':
		c = *s++;
		ref = getarg(c, i);
		switch (rtype(ref)) {
		case RMem:
			goto Mem;
		case RSlot:
			fprintf(e->f, "%d(%%%s)",
				slot(ref, e),
				regtoa(e->fp, SLong)
			);
			break;
		case RCon:
			off = e->fn->con[ref.val];
			emitcon(&off, e);
			if (off.type == CAddr)
			if (off.sym.type != SThr)
				fprintf(e->f, "(%%rip)");
			break;
		case RTmp:
			assert(isreg(ref));
			fprintf(e->f, "(%%%s)", regtoa(ref.val, SLong));
			break;
		default:
			die("unreachable");
		}
		break;
	default:
		die("invalid format specifier %%%c", c);
	}
	goto Next;
}

static bits negmask[4] = {
	[Ks] = 0x80000000,
	[Kd] = 0x8000000000000000,
};

static void
emitins(Ins i, E *e)
{
	Ref r;
	int64_t val;
	int o, t0;
	Ins ineg;
	Con *con;
	char *sym;

	/* the 256-bit vector ops, and the copies of vectors in a function that has them (its slots are 32 bytes) */
	wideins = (i.op >= Ovecbcasts256 && i.op <= Ovecdivs256)
		|| (i.op == Ocopy && i.cls == Kd && e->fn->widevec
		    && (isreg(i.to) || rtype(i.to) == RSlot)
		    && (isreg(i.arg[0]) || rtype(i.arg[0]) == RSlot));

	switch (i.op) {
	default:
		if (isxsel(i.op))
			goto case_Oxsel;
	Table:
		/* most instructions are just pulled out of
		 * the table omap[], some special cases are
		 * detailed below */
		for (o=0;; o++) {
			/* this linear search should really be a binary
			 * search */
			if (omap[o].op == NOp)
				die("no match for %s(%c)",
					optab[i.op].name, "wlsd"[i.cls]);
			if (omap[o].op == i.op)
			if (omap[o].cls == i.cls
			|| (omap[o].cls == Ki && KBASE(i.cls) == 0)
			|| (omap[o].cls == Ka))
				break;
		}
		/* a two-address form whose result is the second source: copy that source aside first (xmm15) */
		if (omap[o].fmt[0] == '-' && KBASE(i.cls) == 1 && req(i.to, i.arg[1]) && !req(i.arg[0], i.to)) {
			emitf("movaps %1, %%xmm15", &i, e);
			i.arg[1] = TMP(XMM0+15);
		}
		emitf(omap[o].fmt, &i, e);
		break;
	case Ovecext0s:
		emitf("movss %0, %=", &i, e);
		break;
	case Ovecext1s:
		emitf("pshufd $1, %0, %%xmm15", &i, e);
		emitf("movss %%xmm15, %=", &i, e);
		break;
	case Ovecext2s:
		emitf("pshufd $2, %0, %%xmm15", &i, e);
		emitf("movss %%xmm15, %=", &i, e);
		break;
	case Ovecext3s:
		emitf("pshufd $3, %0, %%xmm15", &i, e);
		emitf("movss %%xmm15, %=", &i, e);
		break;
	case Ovecext0d:
		emitf("movsd %0, %=", &i, e);
		break;
	case Ovecext1d:
		emitf("pshufd $14, %0, %%xmm15", &i, e);
		emitf("movsd %%xmm15, %=", &i, e);
		break;
	case Ovecext0w:
		emitf("movd %0, %W=", &i, e);
		break;
	case Ovecext1w:
		emitf("pshufd $1, %0, %%xmm15", &i, e);
		emitf("movd %%xmm15, %W=", &i, e);
		break;
	case Ovecext2w:
		emitf("pshufd $2, %0, %%xmm15", &i, e);
		emitf("movd %%xmm15, %W=", &i, e);
		break;
	case Ovecext3w:
		emitf("pshufd $3, %0, %%xmm15", &i, e);
		emitf("movd %%xmm15, %W=", &i, e);
		break;
	case Ovecbcastw:
		/* the scalar into lane 0, then lane 0 into all four (pshufd) */
		emitf("movd %W0, %=", &i, e);
		emitf("pshufd $0, %=, %=", &i, e);
		break;
	case Ovecbcasts:
		emitf("movss %S0, %=", &i, e);
		emitf("pshufd $0, %=, %=", &i, e);
		break;
	case Ovecbcastd:
		/* lanes 0 and 1 (the low quadword) into both quadwords */
		emitf("movsd %D0, %=", &i, e);
		emitf("pshufd $68, %=, %=", &i, e);
		break;
	case Ovecfold256: {
		/* the high 128 bits of the ymm source into the xmm destination: names by hand (the source is wide) */
		char fmt[64], src[8], dst[8];

		wideins = 1;
		strcpy(src, regtoa(i.arg[0].val, SLong));
		wideins = 0;
		strcpy(dst, regtoa(i.to.val, SLong));
		sprintf(fmt, "vextracti128 $1, %%%%%s, %%%%%s", src, dst);
		emitf(fmt, &i, e);
		break;
	}
	case Ovecbcasts256: {
		/* the float to all eight lanes: the source is an xmm (or a slot), the destination a ymm */
		char fmt[64], d[8];

		wideins = 1;
		strcpy(d, regtoa(i.to.val, SLong));
		wideins = 0;
		sprintf(fmt, "vbroadcastss %%S0, %%%%%s", d);
		emitf(fmt, &i, e);
		break;
	}
	case Ovecbcastw256:
		/* the scalar into the low lane of xmm15, then broadcast it to all eight lanes of the ymm */
		emitf("vmovd %W0, %%xmm15", &i, e);
		emitf("vpbroadcastd %%xmm15, %=", &i, e);
		break;
	case Onop:
		/* just do nothing for nops, they are inserted
		 * by some passes */
		break;
	case Omul:
		/* here, we try to use the 3-addresss form
		 * of multiplication when possible */
		if (rtype(i.arg[1]) == RCon) {
			r = i.arg[0];
			i.arg[0] = i.arg[1];
			i.arg[1] = r;
		}
		if (KBASE(i.cls) == 0 /* only available for ints */
		&& rtype(i.arg[0]) == RCon
		&& rtype(i.arg[1]) == RTmp) {
			emitf("imul%k %0, %1, %=", &i, e);
			break;
		}
		goto Table;
	case Osub:
		/* we have to use the negation trick to handle
		 * some 3-address subtractions */
		if (req(i.to, i.arg[1]) && !req(i.arg[0], i.to)) {
			ineg = (Ins){Oneg, i.cls, i.to, {i.to}};
			emitins(ineg, e);
			emitf("add%k %0, %=", &i, e);
			break;
		}
		goto Table;
	case Oneg:
		if (!req(i.to, i.arg[0]))
			emitf("mov%k %0, %=", &i, e);
		if (KBASE(i.cls) == 0)
			emitf("neg%k %=", &i, e);
		else
			fprintf(e->f,
				"\txorp%c %sfp%d(%%rip), %%%s\n",
				"xxsd"[i.cls],
				T.asloc,
				stashbits(negmask[i.cls], 16),
				regtoa(i.to.val, SLong)
			);
		break;
	case Odiv:
		/* use xmm15 to adjust the instruction when the
		 * conversion to 2-address in emitf() would fail */
		if (req(i.to, i.arg[1])) {
			i.arg[1] = TMP(XMM0+15);
			emitf("mov%k %=, %1", &i, e);
			emitf("mov%k %0, %=", &i, e);
			i.arg[0] = i.to;
		}
		goto Table;
	case Ocopy:
		/* copies are used for many things; see my note
		 * to understand how to load big constants:
		 * https://c9x.me/notes/2015-09-19.html */
		assert(rtype(i.to) != RMem);
		if (req(i.to, R) || req(i.arg[0], R))
			break;
		if (req(i.to, i.arg[0]))
			break;
		t0 = rtype(i.arg[0]);
		if (i.cls == Kd
		&& (t0 == RSlot || isreg(i.arg[0]))
		&& (rtype(i.to) == RSlot || isreg(i.to))) {
			/* a double in a slot (spill.c gives it 16 bytes) or a vector: the whole register moves,
			 * movsd would keep only the low lane */
			/* a 256-bit copy (wideins) moves 32 bytes: the slots of such a function are 32 bytes too */
			char *mu = wideins ? "vmovups" : "movups";
			char *ma = wideins ? "vmovaps" : "movaps";
			char fmt[32];
			if (t0 == RSlot && rtype(i.to) == RSlot) {
				i.arg[1] = TMP(XMM0+15);
				sprintf(fmt, "%s %%0, %%1", mu);
				emitf(fmt, &i, e);
				sprintf(fmt, "%s %%1, %%=", mu);
				emitf(fmt, &i, e);
			} else if (isreg(i.to) && isreg(i.arg[0])) {
				sprintf(fmt, "%s %%0, %%=", ma);
				emitf(fmt, &i, e);
			} else {
				sprintf(fmt, "%s %%0, %%=", mu);
				emitf(fmt, &i, e);
			}
			break;
		}
		if (i.cls == Kl
		&& t0 == RCon
		&& e->fn->con[i.arg[0].val].type == CBits) {
			val = e->fn->con[i.arg[0].val].bits.i;
			if (isreg(i.to))
			if (val >= 0 && val <= UINT32_MAX) {
				emitf("movl %W0, %W=", &i, e);
				break;
			}
			if (rtype(i.to) == RSlot)
			if (val < INT32_MIN || val > INT32_MAX) {
				emitf("movl %0, %=", &i, e);
				emitf("movl %0>>32, 4+%=", &i, e);
				break;
			}
		}
		if (isreg(i.to)
		&& t0 == RCon
		&& e->fn->con[i.arg[0].val].type == CAddr) {
			if (absaddr && !(e->fn->con[i.arg[0].val].sym.type & SThr)) {
				emitabsaddr(&i, &e->fn->con[i.arg[0].val], e);   /* -A: a symbol's address, not rip-relative */
				break;
			}
			emitf("lea%k %M0, %=", &i, e);
			break;
		}
		if (rtype(i.to) == RSlot
		&& (t0 == RSlot || t0 == RMem)
		&& nofp && KBASE(i.cls) == 0) {
			/* -n: no %xmm15; %rax, saved below the frame (the
			 * slots are %rbp-relative under -n, emitfn) */
			fputs("\tpushq %rax\n", e->f);
			i.arg[1] = TMP(RAX);
			emitf("mov%k %0, %1", &i, e);
			emitf("mov%k %1, %=", &i, e);
			fputs("\tpopq %rax\n", e->f);
			break;
		}
		if (rtype(i.to) == RSlot
		&& (t0 == RSlot || t0 == RMem)) {
			i.cls = KWIDE(i.cls) ? Kd : Ks;
			i.arg[1] = TMP(XMM0+15);
			emitf("mov%k %0, %1", &i, e);
			emitf("mov%k %1, %=", &i, e);
			break;
		}
		/* conveniently, the assembler knows if it
		 * should use movabsq when reading movq */
		emitf("mov%k %0, %=", &i, e);
		break;
	case Oaddr:
		if (rtype(i.arg[0]) != RCon)
			goto Table;
		con = &e->fn->con[i.arg[0].val];
		assert(isreg(i.to) && con->type == CAddr);
		sym = str(con->sym.id);
		if (absaddr && !(con->sym.type & SThr)) {
			emitabsaddr(&i, con, e);   /* -A: a symbol's address, not rip-relative */
			break;
		}
		if (T.apple && (con->sym.type & SThr)) {
			fprintf(e->f,
				"\tmovq %s%s@tlvp(%%rip), %%%s\n",
				sym[0] == '"' ? "" : T.assym, sym,
				regtoa(i.to.val, SLong));
			break;
		}
		if (T.windows && con->sym.type != SGlo)
			die("extern/thread unsupported on amd64_win");
		switch (con->sym.type) {
		case SThr:
			/* derive the symbol address from the TCB
			 * address at offset 0 of %fs */
			emitf("movq %%fs:0, %L=", &i, e);
			fprintf(e->f, "\tleaq %s%s@tpoff",
				sym[0] == '"' ? "" : T.assym, sym);
			if (con->bits.i)
				fprintf(e->f, "%+"PRId64,
					con->bits.i);
			fprintf(e->f, "(%%%s), %%%s\n",
				regtoa(i.to.val, SLong),
				regtoa(i.to.val, SLong));
			break;
		case SExtThr:
			/* initial-exec TLS: load offset from
			 * GOT, add to thread-base register */
			assert(!con->bits.i);
			emitf("movq %%fs:0, %L=", &i, e);
			fprintf(e->f,
				"\taddq %s%s@gottpoff(%%rip), %%%s\n",
				sym[0] == '"' ? "" : T.assym, sym,
				regtoa(i.to.val, SLong));
			break;
		case SExt:
			if (absaddr) {
				/* -A: the address as gcc's kernel model writes it: a sign-extended 32-bit immediate */
				fprintf(e->f, "\tmovq $%s%s", sym[0] == '"' ? "" : T.assym, sym);
				if (con->bits.i)
					fprintf(e->f, "%+lld", (long long)con->bits.i);
				fprintf(e->f, ", %%%s\n", regtoa(i.to.val, SLong));
				break;
			}
			/* load address from the GOT */
			assert(!con->bits.i);
			fprintf(e->f,
				"\tmovq %s%s@gotpcrel(%%rip), %%%s\n",
				sym[0] == '"' ? "" : T.assym, sym,
				regtoa(i.to.val, SLong));
			break;
		default:
			goto Table;
		}
		break;
	case Ocall:
		/* calls simply have a weird syntax in AT&T
		 * assembly... */
		switch (rtype(i.arg[0])) {
		case RCon:
			con = &e->fn->con[i.arg[0].val];
			fprintf(e->f, e->tail ? "\tjmp " : "\tcallq ");
			emitcon(con, e);
			if (con->type == CAddr
			&& (con->sym.type & SExt)
			&& !T.apple)
				fprintf(e->f, "@plt");
			fprintf(e->f, "\n");
			break;
		case RTmp:
			if (e->tail)
				emitf("jmp *%L0", &i, e);
			else
				emitf("callq *%L0", &i, e);
			break;
		default:
			die("invalid call argument");
		}
		break;
	case Osalloc:
		/* there is no good reason why this is here
		 * maybe we should split Osalloc in 2 different
		 * instructions depending on the result
		 */
		assert(e->fp == RBP);
		emitf("subq %L0, %%rsp", &i, e);
		if (!req(i.to, R))
			emitcopy(i.to, TMP(RSP), Kl, e);
		break;
	case Oswap:
		if (KBASE(i.cls) == 0)
			goto Table;
		/* for floats, there is no swap instruction
		 * so we use xmm15 as a temporary
		 */
		emitcopy(TMP(XMM0+15), i.arg[0], i.cls, e);
		emitcopy(i.arg[0], i.arg[1], i.cls, e);
		emitcopy(i.arg[1], TMP(XMM0+15), i.cls, e);
		break;
	case Odbgloc:
		emitdbgloc(i.arg[0].val, i.arg[1].val, e->f);
		break;
	case_Oxsel:
		if (req(i.to, i.arg[1]))
			emitf(cmov[i.op-Oxsel][0], &i, e);
		else {
			if (!req(i.to, i.arg[0]))
				emitf("mov %0, %=", &i, e);
			emitf(cmov[i.op-Oxsel][1], &i, e);
		}
		break;
	}
}

static void
sysv_framesz(E *e)
{
	uint64_t i, o, f;

	/* specific to NAlign == 3 */
	o = 0;
	if (!e->fn->leaf) {
		for (i=0, o=0; i<NCLR_SYSV; i++)
			o ^= e->fn->reg >> amd64_sysv_rclob[i];
		o &= 1;
	}
	f = e->fn->slot;
	f = (f + 3) & -4;
	if (f > 0
	&& e->fp == RSP
	&& e->fn->salign == 4)
		f += 2;
	e->fsz = 4*f + 8*o + 176*e->fn->vararg;
}

void
amd64_sysv_emitfn(Fn *fn, FILE *f)
{
	static char *ctoa[][2] = {
	#define X(c, s, n) [c] = {s, n},
		CMP(X)
	#undef X
	};
	static int id0;
	Blk *b, *s;
	Ins *i, *last, itmp;
	int *r, c, o, n, lbl, unw;
	uint p;
	E *e;

	e = &(E){.f = f, .fn = fn};
	/* DWARF CFI for functions that make calls: a C++ throw unwinds through them. Leaf functions make no
	   calls, so no unwinder stops inside them. Apple targets use their own unwind format, not emitted here. */
	unw = !T.apple && !fn->leaf && !nounwind;   /* nfcxx -U: no unwind tables */
	emitfnlnk(fn->name, &fn->lnk, f);
	if (unw)
		fputs("\t.cfi_startproc\n", f);
	ehfnbegin(fn, f);
	fputs("\tendbr64\n", f);
	if (!fn->leaf || fn->vararg || fn->dynalloc || nofp) {
		e->fp = RBP;
		fputs("\tpushq %rbp\n\tmovq %rsp, %rbp\n", f);
		if (unw)
			fputs("\t.cfi_def_cfa_offset 16\n\t.cfi_offset %rbp, -16\n\t.cfi_def_cfa_register %rbp\n", f);
	} else
		e->fp = RSP;
	sysv_framesz(e);
	if (e->fsz)
		fprintf(f, "\tsubq $%"PRIu64", %%rsp\n", e->fsz);
	if (fn->vararg) {
		o = -176;
		for (r=amd64_sysv_rsave; r<&amd64_sysv_rsave[6]; r++, o+=8)
			fprintf(f, "\tmovq %%%s, %d(%%rbp)\n", rname[*r][0], o);
		/* -n: no caller passes floating-point arguments
		 * (the kernel's -mno-sse), so none are saved */
		for (n=0; n<8 && !nofp; ++n, o+=16)
			fprintf(f, "\tmovaps %%xmm%d, %d(%%rbp)\n", n, o);
	}
	for (r=amd64_sysv_rclob; r<&amd64_sysv_rclob[NCLR_SYSV]; r++)
		if (fn->reg & BIT(*r)) {
			itmp.arg[0] = TMP(*r);
			emitf("pushq %L0", &itmp, e);
			e->nclob++;
			/* The k-th push is below the frame, at CFA - (16 + fsz + 8k) with the rbp frame. */
			if (unw)
				fprintf(f, "\t.cfi_offset %%%s, %d\n", rname[*r][0],
					-(int)(16 + e->fsz + 8 * e->nclob));
		}

	for (lbl=0, b=fn->start; b; b=b->link) {
		ehlpbegin(fn, b, f);
		if (lbl || b->npred > 1) {
			for (p=0; p<b->npred; p++)
				if (b->pred[p]->id >= b->id)
					break;
			if (p != b->npred)
				fprintf(f, ".p2align 4\n");
			fprintf(f, "%sbb%d:\n", T.asloc, id0+b->id);
		}
		/* a tail call (sysv.c tailcallok): the block's last instruction is its call, which the epilogue precedes */
		last = b->tail && b->jmp.type == Jret0 && b->nins && b->ins[b->nins-1].op == Ocall
			&& tailtarget(&b->ins[b->nins-1]) ? &b->ins[b->nins-1] : 0;
		for (i=b->ins; i!=&b->ins[b->nins]; i++) {
			if (i == last)
				break;
			if (i->op == Ocall)
				ehcsbegin(fn, b, f);
			emitins(*i, e);
			if (i->op == Ocall)
				ehcsend(fn, b, f);
		}
		lbl = 1;
		switch (b->jmp.type) {
		case Jhlt:
			fprintf(f, "\tud2\n");
			break;
		case Jret0:
			/* The epilogue changes the CFA rules; the body's rules are restored after the ret. */
			if (unw)
				fputs("\t.cfi_remember_state\n", f);
			if (fn->dynalloc)
				fprintf(f,
					"\tmovq %%rbp, %%rsp\n"
					"\tsubq $%"PRIu64", %%rsp\n",
					e->fsz + e->nclob * 8);
			for (r=&amd64_sysv_rclob[NCLR_SYSV]; r>amd64_sysv_rclob;)
				if (fn->reg & BIT(*--r)) {
					itmp.arg[0] = TMP(*r);
					emitf("popq %L0", &itmp, e);
				}
			if (e->fp == RBP)
				fputs("\tleave\n", f);
			else if (e->fsz)
				fprintf(f,
					"\taddq $%"PRIu64", %%rsp\n",
					e->fsz);
			if (last) {	/* the tail call: its jump takes the return's place */
				e->tail = 1;
				emitins(*last, e);
				e->tail = 0;
			} else {
				if (e->fn->widevec)
					fputs("\tvzeroupper\n", f);	/* the upper halves of the ymm registers are not the callee's */
				fputs("\tret\n", f);
			}
			if (unw)
				fputs("\t.cfi_restore_state\n", f);
			break;
		case Jjmp:
		Jmp:
			if (b->s1 != b->link)
				fprintf(f, "\tjmp %sbb%d\n",
					T.asloc, id0+b->s1->id);
			else
				lbl = 0;
			break;
		case Jjtab:
			emitjtab(b, id0, e);
			break;
		default:
			c = b->jmp.type - Jjf;
			if (0 <= c && c <= NCmp) {
				if (b->link == b->s2) {
					s = b->s1;
					b->s1 = b->s2;
					b->s2 = s;
					n = 0;
				} else
					n = 1;
				fprintf(f, "\tj%s %sbb%d\n", ctoa[c][n],
					T.asloc, id0+b->s2->id);
				goto Jmp;
			}
			die("unhandled jump %d", b->jmp.type);
		}
	}
	id0 += fn->nblk;
	if (unw)
		fputs("\t.cfi_endproc\n", f);
	if (!T.apple)
		elf_emitfnfin(fn->name, f);
	emitjtabs(fn, id0 - fn->nblk, e);
	ehfnend(fn, f);
}

static void
winabi_framesz(E *e)
{
	uint64_t i, o, f;

	/* specific to NAlign == 3 */
	o = 0;
	if (!e->fn->leaf) {
		for (i=0, o=0; i<NCLR_WIN; i++)
			o ^= e->fn->reg >> amd64_winabi_rclob[i];
		o &= 1;
	}
	f = e->fn->slot;
	f = (f + 3) & -4;
	if (f > 0
	&& e->fp == RSP
	&& e->fn->salign == 4)
		f += 2;
	e->fsz = 4*f + 8*o;
}

void
amd64_winabi_emitfn(Fn *fn, FILE *f)
{
	static char *ctoa[][2] = {
	#define X(c, s, n) [c] = {s, n},
		CMP(X)
	#undef X
	};
	static int id0;
	Blk *b, *s;
	Ins *i, itmp;
	int *r, c, n, lbl;
	E *e;

	e = &(E){.f = f, .fn = fn};
	emitfnlnk(fn->name, &fn->lnk, f);
	fputs("\tendbr64\n", f);
	if (fn->vararg) {
		fprintf(f, "\tmovq %%rcx, 0x8(%%rsp)\n");
		fprintf(f, "\tmovq %%rdx, 0x10(%%rsp)\n");
		fprintf(f, "\tmovq %%r8, 0x18(%%rsp)\n");
		fprintf(f, "\tmovq %%r9, 0x20(%%rsp)\n");
	}
	if (!fn->leaf || fn->vararg || fn->dynalloc) {
		e->fp = RBP;
		fputs("\tpushq %rbp\n\tmovq %rsp, %rbp\n", f);
	} else
		e->fp = RSP;
	winabi_framesz(e);
	if (e->fsz)
		fprintf(f, "\tsubq $%"PRIu64", %%rsp\n", e->fsz);
	for (r=amd64_winabi_rclob; r<&amd64_winabi_rclob[NCLR_WIN]; r++)
		if (fn->reg & BIT(*r)) {
			itmp.arg[0] = TMP(*r);
			emitf("pushq %L0", &itmp, e);
			e->nclob++;
		}

	for (lbl=0, b=fn->start; b; b=b->link) {
		if (lbl || b->npred > 1)
			fprintf(f, "%sbb%d:\n", T.asloc, id0+b->id);
		for (i=b->ins; i!=&b->ins[b->nins]; i++)
			emitins(*i, e);
		lbl = 1;
		switch (b->jmp.type) {
		case Jhlt:
			fprintf(f, "\tud2\n");
			break;
		case Jret0:
			if (fn->dynalloc)
				fprintf(f,
					"\tmovq %%rbp, %%rsp\n"
					"\tsubq $%"PRIu64", %%rsp\n",
					e->fsz + e->nclob * 8);
			for (r=&amd64_winabi_rclob[NCLR_WIN]; r>amd64_winabi_rclob;)
				if (fn->reg & BIT(*--r)) {
					itmp.arg[0] = TMP(*r);
					emitf("popq %L0", &itmp, e);
				}
			if (e->fp == RBP)
				fputs("\tleave\n", f);
			else if (e->fsz)
				fprintf(f,
					"\taddq $%"PRIu64", %%rsp\n",
					e->fsz);
			if (e->fn->widevec)
				fputs("\tvzeroupper\n", f);
			fputs("\tret\n", f);
			break;
		case Jjmp:
		Jmp:
			if (b->s1 != b->link)
				fprintf(f, "\tjmp %sbb%d\n",
					T.asloc, id0+b->s1->id);
			else
				lbl = 0;
			break;
		case Jjtab:
			die("jtab: not on the amd64_win target");
		default:
			c = b->jmp.type - Jjf;
			if (0 <= c && c <= NCmp) {
				if (b->link == b->s2 || c >= NCmpI) {
					s = b->s1;
					b->s1 = b->s2;
					b->s2 = s;
					n = 0;
				} else
					n = 1;
				fprintf(f, "\tj%s %sbb%d\n", ctoa[c][n],
					T.asloc, id0+b->s2->id);
				goto Jmp;
			}
			die("unhandled jump %d", b->jmp.type);
		}
	}
	id0 += fn->nblk;
}
