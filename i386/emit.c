#include "all.h"

/* i386 assembly (AT&T, gas), after amd64/emit.c
 *
 * The frame is always on %ebp:
 *
 *      8(%ebp)  first stack argument, SLOT(-2)
 *      4(%ebp)  return address
 *      0(%ebp)  caller's %ebp
 *     -8(%ebp)  end of the locals: the local slots are below
 *               -8, so that a slot aligned on 16 is aligned
 *               when the caller kept %esp aligned on 16
 *               (%ebp is 8 mod 16)
 *               padding, then the callee-save registers
 *
 * esi, edi and ebp have no 8-bit register: a byte store
 * from them, an 8-bit extension of them and a flag set in
 * them go through a register that has one, exchanged with
 * xchg (which leaves the flags alone).
 */

typedef struct E E;

struct E {
	FILE *f;
	Fn *fn;
	uint64_t fsz;
	int nclob;
};

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
	X(Cine,       "nz", "z")

enum {
	SWord = 0,
	SShort = 1,
	SByte = 2,
};

/* Instruction format strings: as in amd64/emit.c, with
 * %k always "l" (the only class is w) and the prefixes
 * M (memory), W (32 bits), H (16 bits), B (8 bits)
 */
static struct {
	short op;
	char *fmt;
} omap[] = {
	{ Oadd,     "+addl %1, %=" },
	{ Osub,     "-subl %1, %=" },
	{ Oand,     "+andl %1, %=" },
	{ Oor,      "+orl %1, %=" },
	{ Oxor,     "+xorl %1, %=" },
	{ Osar,     "-sarl %B1, %=" },
	{ Oshr,     "-shrl %B1, %=" },
	{ Oshl,     "-shll %B1, %=" },
	{ Omul,     "+imull %1, %=" },
	{ Ostorew,  "movl %W0, %M1" },
	{ Ostoreh,  "movw %H0, %M1" },
	{ Ostoreb,  "movb %B0, %M1" },
	{ Oload,    "movl %M0, %=" },
	{ Oloadsw,  "movl %M0, %=" },
	{ Oloaduw,  "movl %M0, %=" },
	{ Oloadsh,  "movswl %M0, %=" },
	{ Oloaduh,  "movzwl %M0, %=" },
	{ Oloadsb,  "movsbl %M0, %=" },
	{ Oloadub,  "movzbl %M0, %=" },
	{ Oextsh,   "movswl %H0, %=" },
	{ Oextuh,   "movzwl %H0, %=" },
	{ Oextsb,   "movsbl %B0, %=" },
	{ Oextub,   "movzbl %B0, %=" },
	{ Oaddr,    "leal %M0, %=" },
	{ Oswap,    "xchgl %0, %1" },
	{ Osign,    "cltd" },
	{ Oxdiv,    "divl %0" },
	{ Oxidiv,   "idivl %0" },
	{ Oxcmp,    "cmpl %0, %1" },
	{ Oxtest,   "testl %0, %1" },
	{ NOp, 0 }
};

static char *ctoa[][2] = {
#define X(c, s, n) [c] = {s, n},
	CMP(X)
#undef X
};

static char *rname[][3] = {
	[EAX] = {"eax", "ax", "al"},
	[EBX] = {"ebx", "bx", "bl"},
	[ECX] = {"ecx", "cx", "cl"},
	[EDX] = {"edx", "dx", "dl"},
	[ESI] = {"esi", "si", 0},
	[EDI] = {"edi", "di", 0},
	[EBP] = {"ebp", "bp", 0},
	[ESP] = {"esp", "sp", 0},
};

static int
hasbyte(int r)
{
	return r == EAX || r == EBX || r == ECX || r == EDX;
}

static int
slot(Ref r, E *e)
{
	int s;

	s = rsval(r);
	assert(s <= e->fn->slot);
	/* specific to NAlign == 3 */
	if (s < 0)
		return 4*-s;
	else
		return -8 - 4 * (e->fn->slot - s);
}

static void
emitcon(Con *con, int sz, E *e)
{
	char *l;
	int64_t v;

	switch (con->type) {
	case CAddr:
		l = str(con->sym.id);
		assert(con->sym.type == SGlo);
		fputs(l, e->f);
		if ((int32_t)con->bits.i)
			fprintf(e->f, "%+"PRId32, (int32_t)con->bits.i);
		break;
	case CBits:
		v = con->bits.i;
		switch (sz) {
		case SByte: v = (int8_t)v; break;
		case SShort: v = (int16_t)v; break;
		default: v = (int32_t)v; break;
		}
		fprintf(e->f, "%"PRId64, v);
		break;
	default:
		die("unreachable");
	}
}

static char *
regtoa(int reg, int sz)
{
	assert(EAX <= reg && reg <= ESP);
	if (!rname[reg][sz])
		die("no %d-bit name for %%%s", 32 >> sz, rname[reg][0]);
	return rname[reg][sz];
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

static void
emitmem(Mem *m, E *e)
{
	Con off;

	if (rtype(m->base) == RSlot) {
		off.type = CBits;
		off.bits.i = slot(m->base, e);
		addcon(&m->offset, &off, 1);
		m->base = TMP(EBP);
	}
	if (m->offset.type != CUndef)
		emitcon(&m->offset, SWord, e);
	else if (req(m->base, R) && req(m->index, R))
		fputc('0', e->f);
	if (req(m->base, R) && req(m->index, R))
		return;
	fputc('(', e->f);
	if (!req(m->base, R))
		fprintf(e->f, "%%%s", regtoa(m->base.val, SWord));
	if (!req(m->index, R))
		fprintf(e->f, ", %%%s, %d",
			regtoa(m->index.val, SWord),
			m->scale
		);
	fputc(')', e->f);
}

static void emitins(Ins, E *);

static void
emitcopy(Ref r1, Ref r2, E *e)
{
	Ins icp;

	icp.op = Ocopy;
	icp.arg[0] = r2;
	icp.to = r1;
	icp.cls = Kw;
	emitins(icp, e);
}

static void
emitf(char *s, Ins *i, E *e)
{
	char c;
	int sz;
	Ref ref;

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
		emitcopy(i->to, i->arg[0], e);
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
		fputc('l', e->f);
		break;
	case '0':
	case '1':
	case '=':
		sz = SWord;
		s--;
	Ref:
		c = *s++;
		ref = getarg(c, i);
		switch (rtype(ref)) {
		case RTmp:
			assert(isreg(ref));
			fprintf(e->f, "%%%s", regtoa(ref.val, sz));
			break;
		case RSlot:
			fprintf(e->f, "%d(%%ebp)", slot(ref, e));
			break;
		case RMem:
			emitmem(&e->fn->mem[ref.val], e);
			break;
		case RCon:
			fputc('$', e->f);
			emitcon(&e->fn->con[ref.val], sz, e);
			break;
		default:
			die("unreachable");
		}
		break;
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
			emitmem(&e->fn->mem[ref.val], e);
			break;
		case RSlot:
			fprintf(e->f, "%d(%%ebp)", slot(ref, e));
			break;
		case RCon:
			emitcon(&e->fn->con[ref.val], SWord, e);
			break;
		case RTmp:
			assert(isreg(ref));
			fprintf(e->f, "(%%%s)", regtoa(ref.val, SWord));
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

static int
usesreg(Ref r, int reg, E *e)
{
	Mem *m;

	switch (rtype(r)) {
	case RTmp:
		return r.val == (uint)reg;
	case RMem:
		m = &e->fn->mem[r.val];
		return req(m->base, TMP(reg)) || req(m->index, TMP(reg));
	default:
		return 0;
	}
}

static Ref
swapreg(Ref r, int r0, int r1, E *e)
{
	Mem *m;

	switch (rtype(r)) {
	case RTmp:
		if (r.val == (uint)r0)
			return TMP(r1);
		if (r.val == (uint)r1)
			return TMP(r0);
		return r;
	case RMem:
		vgrow(&e->fn->mem, ++e->fn->nmem);
		m = &e->fn->mem[e->fn->nmem-1];
		*m = e->fn->mem[r.val];
		m->base = swapreg(m->base, r0, r1, e);
		m->index = swapreg(m->index, r0, r1, e);
		return MEM(e->fn->nmem-1);
	default:
		return r;
	}
}

/* a byte store of esi or edi: through a byte register
 * that the address does not use, exchanged with it; the
 * address then names it in place of the stored register */
static void
emitstoreb(Ins *i, E *e)
{
	static int br[] = {EAX, ECX, EDX, EBX};
	int b, j, r;
	Ins is;

	r = i->arg[0].val;
	for (j=0;; j++) {
		assert(j < 4);
		b = br[j];
		if (!usesreg(i->arg[1], b, e))
			break;
	}
	fprintf(e->f, "\txchgl %%%s, %%%s\n", rname[b][0], rname[r][0]);
	is = *i;
	is.arg[0] = TMP(b);
	is.arg[1] = swapreg(i->arg[1], r, b, e);
	emitf("movb %B0, %M1", &is, e);
	fprintf(e->f, "\txchgl %%%s, %%%s\n", rname[b][0], rname[r][0]);
}

static void
emitins(Ins i, E *e)
{
	Ref r;
	int o, t0;
	Ins ineg;
	Con *con;

	switch (i.op) {
	default:
		if (i.op >= Oflag && i.op <= Oflag1) {
			o = i.op - Oflag;
			if (o >= NCmpI)
				die("no floating point on i386");
			assert(isreg(i.to));
			if (hasbyte(i.to.val)) {
				fprintf(e->f, "\tset%s %%%s\n\tmovzbl %%%s, %%%s\n",
					ctoa[o][0], rname[i.to.val][SByte],
					rname[i.to.val][SByte], rname[i.to.val][0]);
			} else {
				fprintf(e->f,
					"\txchgl %%eax, %%%s\n"
					"\tset%s %%al\n"
					"\tmovzbl %%al, %%eax\n"
					"\txchgl %%eax, %%%s\n",
					rname[i.to.val][0], ctoa[o][0],
					rname[i.to.val][0]);
			}
			break;
		}
	Table:
		if (i.cls != Kw && i.op != Ostorew && i.op != Ostoreh
		&& i.op != Ostoreb && i.op != Oxdiv && i.op != Oxidiv)
			die("class %c on i386 (%s)", "wlsd"[i.cls],
				optab[i.op].name);
		for (o=0;; o++) {
			if (omap[o].op == NOp)
				die("no match for %s(%c)",
					optab[i.op].name, "wlsd"[i.cls]);
			if (omap[o].op == i.op)
				break;
		}
		emitf(omap[o].fmt, &i, e);
		break;
	case Onop:
		break;
	case Ostoreb:
		if (rtype(i.arg[0]) == RTmp && !hasbyte(i.arg[0].val)) {
			emitstoreb(&i, e);
			break;
		}
		goto Table;
	case Oextsb:
	case Oextub:
		if (rtype(i.arg[0]) == RTmp && !hasbyte(i.arg[0].val)) {
			if (!req(i.to, i.arg[0]))
				emitf("movl %0, %=", &i, e);
			if (i.op == Oextub)
				emitf("andl $255, %=", &i, e);
			else
				emitf("shll $24, %=\n\tsarl $24, %=", &i, e);
			break;
		}
		goto Table;
	case Omul:
		/* here, we try to use the 3-addresss form
		 * of multiplication when possible */
		if (rtype(i.arg[1]) == RCon) {
			r = i.arg[0];
			i.arg[0] = i.arg[1];
			i.arg[1] = r;
		}
		if (rtype(i.arg[0]) == RCon
		&& rtype(i.arg[1]) == RTmp) {
			emitf("imull %0, %1, %=", &i, e);
			break;
		}
		goto Table;
	case Osub:
		/* we have to use the negation trick to handle
		 * some 3-address subtractions */
		if (req(i.to, i.arg[1]) && !req(i.arg[0], i.to)) {
			ineg = (Ins){Oneg, i.cls, i.to, {i.to}};
			emitins(ineg, e);
			emitf("addl %0, %=", &i, e);
			break;
		}
		goto Table;
	case Oneg:
		if (!req(i.to, i.arg[0]))
			emitf("movl %0, %=", &i, e);
		emitf("negl %=", &i, e);
		break;
	case Ocopy:
		assert(rtype(i.to) != RMem);
		if (req(i.to, R) || req(i.arg[0], R))
			break;
		if (req(i.to, i.arg[0]))
			break;
		t0 = rtype(i.arg[0]);
		if (rtype(i.to) == RSlot
		&& (t0 == RSlot || t0 == RMem)) {
			/* memory to memory, through the stack */
			emitf("pushl %0", &i, e);
			emitf("popl %=", &i, e);
			break;
		}
		emitf("movl %0, %=", &i, e);
		break;
	case Ocall:
		switch (rtype(i.arg[0])) {
		case RCon:
			con = &e->fn->con[i.arg[0].val];
			fprintf(e->f, "\tcall ");
			emitcon(con, SWord, e);
			fprintf(e->f, "\n");
			break;
		case RTmp:
			emitf("call *%W0", &i, e);
			break;
		default:
			die("invalid call argument");
		}
		break;
	case Osalloc:
		emitf("subl %W0, %%esp", &i, e);
		if (!req(i.to, R))
			emitcopy(i.to, TMP(ESP), e);
		break;
	case Odbgloc:
		emitdbgloc(i.arg[0].val, i.arg[1].val, e->f);
		break;
	}
}

static void
framesz(E *e)
{
	Fn *fn;
	uint64_t f;
	int *r;

	fn = e->fn;
	e->nclob = 0;
	for (r=i386_rclob; *r>=0; r++)
		if (fn->reg & BIT(*r))
			e->nclob++;
	f = 4 * (uint64_t)fn->slot;
	if (f)
		f += 8;
	if (!fn->leaf || fn->dynalloc) {
		/* keep %esp aligned on 16 at calls: %ebp is
		 * 8 mod 16, then the frame and the saves */
		f += (8 - (f + 4*e->nclob)) & 15;
	}
	e->fsz = f;
}

void
i386_emitfn(Fn *fn, FILE *f)
{
	static int id0;
	Blk *b, *s;
	Ins *i;
	int *r, c, n, lbl;
	E *e;

	e = &(E){.f = f, .fn = fn};
	emitfnlnk(fn->name, &fn->lnk, f);
	fputs("\tpushl %ebp\n\tmovl %esp, %ebp\n", f);
	framesz(e);
	if (e->fsz)
		fprintf(f, "\tsubl $%"PRIu64", %%esp\n", e->fsz);
	for (r=i386_rclob; *r>=0; r++)
		if (fn->reg & BIT(*r))
			fprintf(f, "\tpushl %%%s\n", rname[*r][0]);

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
			if (fn->dynalloc && e->nclob)
				fprintf(f, "\tleal -%"PRIu64"(%%ebp), %%esp\n",
					e->fsz + 4 * e->nclob);
			for (r=&i386_rclob[NCLR]; r>i386_rclob;)
				if (fn->reg & BIT(*--r))
					fprintf(f, "\tpopl %%%s\n", rname[*r][0]);
			fputs("\tleave\n", f);
			/* cdecl: the callee pops the hidden
			 * argument of an aggregate return */
			if (fn->retty >= 0 && !T.regparm)
				fputs("\tret $4\n", f);
			else
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
		default:
			c = b->jmp.type - Jjf;
			if (0 <= c && c < NCmpI) {
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
	elf_emitfnfin(fn->name, f);
}
