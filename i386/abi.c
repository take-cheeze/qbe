#include "all.h"

/* The System V i386 ABI, as gcc implements it on Linux:
 *
 * - arguments are words on the stack, the first at 4(%esp)
 *   on entry; an aggregate is copied there, its size rounded
 *   up to 4 (gcc places it on 4 bytes, as here)
 * - a word is returned in eax; an aggregate always in memory
 *   (gcc's default -fpcc-struct-return on Linux): the caller
 *   passes the address as a hidden first argument and the
 *   callee returns it in eax; with the cdecl convention, the
 *   callee pops the hidden argument (ret $4)
 * - eax, ecx, edx are caller-save; ebx, esi, edi, ebp
 *   callee-save
 * - with gcc's -mregparm=3 (T.regparm), the first three
 *   words of arguments go in eax, edx, ecx: the hidden
 *   pointer first, then each argument of n words while n
 *   registers remain (a small aggregate too, its words in
 *   order); the first argument that does not fit goes on the
 *   stack with all the following ones.  A variadic function
 *   takes all its arguments on the stack.
 *
 * The callers keep %esp aligned on 16 at calls (the Linux
 * ABI since gcc 4.5); code built with gcc's
 * -mpreferred-stack-boundary=2, as the Linux kernel is, only
 * keeps it on 4, so locals aligned on more than 4 are only
 * aligned under 16-aligned callers.
 */

typedef struct AClass AClass;
typedef struct RAlloc RAlloc;

struct AClass {
	Typ *type;
	int inmem;
	uint size;  /* bytes in memory, a multiple of 4 */
	int nreg;   /* words in registers */
	int reg;    /* first register (index in argreg[]) */
	uint off;   /* stack offset when inmem */
	Ref ref[3];
};

struct RAlloc {
	Ins i;
	RAlloc *link;
};

static int argreg[] = {EAX, EDX, ECX};

int i386_rsave[] = {EAX, ECX, EDX, -1};
int i386_rclob[] = {EBX, ESI, EDI, -1};

MAKESURE(i386_arrays_ok,
	sizeof i386_rsave == (NGPS+1) * sizeof(int) &&
	sizeof i386_rclob == (NCLR+1) * sizeof(int)
);

/* layout of call's second argument (RCall)
 *
 *  29     8    4  2  0
 *  |0...00|xxxx|xx|xx|
 *              |  ` gp regs returned  (0..1)
 *              ` gp regs passed       (0..3)
 */

bits
i386_retregs(Ref r, int p[2])
{
	bits b;
	int ni;

	assert(rtype(r) == RCall);
	ni = r.val & 3;
	b = 0;
	if (ni >= 1)
		b |= BIT(EAX);
	if (ni >= 2)
		b |= BIT(EDX);
	if (p) {
		p[0] = ni;
		p[1] = 0;
	}
	return b;
}

bits
i386_argregs(Ref r, int p[2])
{
	bits b;
	int j, ni;

	assert(rtype(r) == RCall);
	ni = (r.val >> 4) & 15;
	assert(ni <= 3);
	b = 0;
	for (j=0; j<ni; j++)
		b |= BIT(argreg[j]);
	if (p) {
		p[0] = ni;
		p[1] = 0;
	}
	return b;
}

/* classifies the arguments i0..i1 (op is Oarg or Opar);
 * returns the number of registers used, *stk gets the
 * stack bytes, *hreg whether the hidden return pointer
 * is in eax (retmem)
 */
static int
argsclass(Ins *i0, Ins *i1, AClass *ac, int op, int retmem, int varc,
	uint *stk, int *hreg)
{
	int nreg, ni, words;
	AClass *a;
	Ins *i;

	nreg = varc ? 0 : T.regparm;
	ni = 0;
	*stk = 0;
	*hreg = 0;
	if (retmem) {
		if (nreg > 0) {
			*hreg = 1;
			nreg--;
			ni++;
		} else
			*stk = 4;
	}
	for (i=i0, a=ac; i<i1; i++, a++) {
		memset(a, 0, sizeof *a);
		switch (i->op - op + Oarg) {
		case Oarg:
			a->size = 4;
			break;
		case Oargc:
			a->type = &typ[i->arg[0].val];
			if (a->type->size > UINT_MAX - 3)
				err("aggregate argument too large");
			a->size = (a->type->size + 3) & -4;
			break;
		case Oargv:
			continue;
		case Oarge:
			err("environment arguments are not supported on %s", T.name);
		default:
			die("unreachable");
		}
		words = a->size / 4;
		if (words == 0)
			continue;
		if (words <= nreg) {
			a->nreg = words;
			a->reg = ni;
			ni += words;
			nreg -= words;
		} else {
			/* gcc: no register for the arguments that follow */
			nreg = 0;
			a->inmem = 1;
			a->off = *stk;
			*stk += a->size;
		}
	}
	return ni;
}

static int
hasvarg(Ins *i0, Ins *i1)
{
	Ins *i;

	for (i=i0; i<i1; i++)
		if (i->op == Oargv)
			return 1;
	return 0;
}

static Ref
offaddr(Ref base, uint off, Fn *fn)
{
	Ref r;

	if (off == 0)
		return base;
	r = newtmp("abi", Kw, fn);
	/* emitted before its use (emit() goes backwards):
	 * the caller emits the use first */
	emit(Oadd, Kw, r, base, getcon(off, fn));
	return r;
}

/* loads the word at off of the aggregate at p (of tsz bytes)
 * in the register or temporary r, without reading past the
 * aggregate's end
 */
static void
ldword(Ref r, Ref p, uint off, uint64_t tsz, Fn *fn)
{
	uint n;
	Ref t0, t1, t2;
	Ins *i;

	n = tsz - off >= 4 ? 4 : tsz - off;
	switch (n) {
	case 4:
		emit(Oload, Kw, r, R, R);
		break;
	case 2:
		emit(Oloaduh, Kw, r, R, R);
		break;
	case 1:
		emit(Oloadub, Kw, r, R, R);
		break;
	case 3:
		t0 = newtmp("abi", Kw, fn);
		t1 = newtmp("abi", Kw, fn);
		t2 = newtmp("abi", Kw, fn);
		emit(Oor, Kw, r, t0, t2);
		emit(Oshl, Kw, t2, t1, getcon(16, fn));
		emit(Oloadub, Kw, t1, R, R);
		i = curi;
		i->arg[0] = offaddr(p, off+2, fn);
		emit(Oloaduh, Kw, t0, R, R);
		break;
	default:
		die("unreachable");
	}
	i = curi;
	i->arg[0] = offaddr(p, off, fn);
}

static void
selret(Blk *b, Fn *fn)
{
	int j;
	Ref r0;

	j = b->jmp.type;
	if (!isret(j) || j == Jret0)
		return;
	r0 = b->jmp.arg;
	b->jmp.type = Jret0;
	if (j == Jretc) {
		assert(rtype(fn->retr) == RTmp);
		emit(Ocopy, Kw, TMP(EAX), fn->retr, R);
		emit(Oblit1, 0, R, INT(typ[fn->retty].size), R);
		emit(Oblit0, 0, R, r0, fn->retr);
	} else {
		assert(j == Jretw);
		emit(Ocopy, Kw, TMP(EAX), r0, R);
	}
	b->jmp.arg = CALL(1);
}

static void
selcall(Fn *fn, Ins *i0, Ins *i1, RAlloc **rap)
{
	Ins *i, ains[3];
	AClass *ac, *a;
	int ca, ni, hreg, retmem, j, n;
	uint stk, stk0, pop;
	Ref r, r1, pad;
	Typ *rt;
	RAlloc *ra;

	ac = alloc((i1-i0+1) * sizeof ac[0]);
	retmem = !req(i1->arg[1], R);
	ni = argsclass(i0, i1, ac, Oarg, retmem, hasvarg(i0, i1),
		&stk0, &hreg);
	ca = ni << 4;
	stk = (stk0 + 15) & -16;
	/* the cdecl callee pops the hidden argument */
	pop = retmem && !T.regparm ? 4 : 0;

	if (stk - pop)
		emit(Osalloc, Kw, R, getcon(-(int64_t)(stk - pop), fn), R);

	pad = R;
	if (retmem) {
		assert(rtype(i1->arg[1]) == RType);
		rt = &typ[i1->arg[1].val];
		pad = newtmp("abi", Kw, fn);
		emit(Ocopy, Kw, i1->to, TMP(EAX), R);
		n = stkalloc(ains, pad, rt->align, rt->size ? rt->size : 1, fn);
		for (j=0; j<n; j++) {
			ra = alloc(sizeof *ra);
			ra->i = ains[j];
			ra->link = *rap;
			*rap = ra;
		}
	} else
		emit(Ocopy, i1->cls, i1->to, TMP(EAX), R);
	ca += 1;

	emit(Ocall, i1->cls, R, i1->arg[0], CALL(ca));

	/* arguments in registers */
	if (hreg)
		emit(Ocopy, Kw, TMP(EAX), pad, R);
	for (i=i0, a=ac; i<i1; i++, a++) {
		if (a->nreg == 0)
			continue;
		if (i->op == Oargc)
			for (j=0; j<a->nreg; j++)
				ldword(TMP(argreg[a->reg+j]), i->arg[1],
					4*j, a->type->size, fn);
		else
			emit(Ocopy, Kw, TMP(argreg[a->reg]), i->arg[0], R);
	}

	/* arguments on the stack */
	if (!stk)
		return;
	r = newtmp("abi", Kw, fn);
	for (i=i0, a=ac; i<i1; i++, a++) {
		if (!a->inmem)
			continue;
		r1 = newtmp("abi", Kw, fn);
		if (i->op == Oargc) {
			emit(Oblit1, 0, R, INT(a->type->size), R);
			emit(Oblit0, 0, R, i->arg[1], r1);
		} else
			emit(Ostorew, 0, R, i->arg[0], r1);
		emit(Oadd, Kw, r1, r, getcon(a->off, fn));
	}
	if (retmem && !hreg)
		emit(Ostorew, 0, R, pad, r);
	emit(Osalloc, Kw, r, getcon(stk, fn), R);
}

static uint
selpar(Fn *fn, Ins *i0, Ins *i1)
{
	AClass *ac, *a;
	Ins *i, ains[3];
	int ni, hreg, retmem, j, n;
	uint stk, s;
	Ref r;

	ac = alloc((i1-i0+1) * sizeof ac[0]);
	curi = &insb[NIns];
	retmem = fn->retty >= 0;
	ni = argsclass(i0, i1, ac, Opar, retmem, fn->vararg, &stk, &hreg);
	fn->reg = i386_argregs(CALL(ni << 4), 0);

	/* aggregates in registers are stored to a local slot,
	 * after all the registers are read */
	for (i=i0, a=ac; i<i1; i++, a++) {
		if (i->op != Oparc || a->inmem)
			continue;
		for (j=a->nreg-1; j>=0; j--) {
			a->ref[j] = newtmp("abi", Kw, fn);
			r = i->to;
			if (j) {
				r = newtmp("abi", Kw, fn);
				emit(Ostorew, 0, R, a->ref[j], r);
				emit(Oadd, Kw, r, i->to, getcon(4*j, fn));
			} else
				emit(Ostorew, 0, R, a->ref[j], r);
		}
		n = stkalloc(ains, i->to, a->type->align, a->size ? a->size : 4, fn);
		for (; n>0; n--)
			emiti(ains[n-1]);
	}

	/* the first argument is at 8(%ebp), SLOT(-2) */
	s = 2;
	if (retmem) {
		r = newtmp("abi", Kw, fn);
		if (hreg)
			emit(Ocopy, Kw, r, TMP(EAX), R);
		else
			emit(Oload, Kw, r, SLOT(-2), R);
		fn->retr = r;
	}
	for (i=i0, a=ac; i<i1; i++, a++) {
		if (a->inmem) {
			if (i->op == Oparc)
				fn->tmp[i->to.val].slot = -(s + a->off/4);
			else
				emit(Oload, Kw, i->to, SLOT(-(s + a->off/4)), R);
			continue;
		}
		if (a->nreg == 0)
			continue;
		if (i->op == Oparc)
			for (j=0; j<a->nreg; j++)
				emit(Ocopy, Kw, a->ref[j], TMP(argreg[a->reg+j]), R);
		else
			emit(Ocopy, Kw, i->to, TMP(argreg[a->reg]), R);
	}
	return stk;
}

static void
selvastart(Fn *fn, uint stk, Ref ap)
{
	Ref r;

	/* va_list is a pointer to the next argument */
	r = newtmp("abi", Kw, fn);
	emit(Ostorew, Kw, R, r, ap);
	emit(Oadd, Kw, r, TMP(EBP), getcon(8 + stk, fn));
}

static void
selvaarg(Fn *fn, Ins *i)
{
	Ref r0, r1;

	assert(i->cls == Kw);
	r0 = newtmp("abi", Kw, fn);
	r1 = newtmp("abi", Kw, fn);
	emit(Ostorew, Kw, R, r1, i->arg[0]);
	emit(Oadd, Kw, r1, r0, getcon(4, fn));
	emit(Oload, Kw, i->to, r0, R);
	emit(Oload, Kw, r0, i->arg[0], R);
}

/* what the i386 target cannot compile yet is an error, never
 * code that does something else: values of class l (64-bit
 * integers), s and d (no x87 floating point), thread-local
 * symbols, landing pads
 */
static void
refuse(Fn *fn)
{
	static char *msg = "the i386 target does not support %s yet"
		" (function %s)";
	Blk *b;
	Ins *i;
	Tmp *t;
	Con *c;
	int n, k;

	for (t=&fn->tmp[Tmp0]; t<&fn->tmp[fn->ntmp]; t++)
		switch (t->cls) {
		case Kl:
			err(msg, "64-bit integers (class l)", fn->name);
		case Ks:
		case Kd:
			err(msg, "floating point", fn->name);
		}
	for (b=fn->start; b; b=b->link) {
		for (i=b->ins; i<&b->ins[b->nins]; i++) {
			if (i->op == Oargc || i->op == Oparc)
				continue;
			for (n=0; n<2; n++) {
				if (req(i->arg[n], R)
				|| rtype(i->arg[n]) == RType)
					continue;
				k = argcls(i, n);
				if (k == Kl)
					err(msg, "64-bit integers (class l)",
						fn->name);
				if (k == Ks || k == Kd)
					err(msg, "floating point", fn->name);
			}
		}
		switch (b->jmp.type) {
		case Jretl:
			err(msg, "64-bit integers (class l)", fn->name);
		case Jrets:
		case Jretd:
			err(msg, "floating point", fn->name);
		}
	}
	for (c=fn->con; c<&fn->con[fn->ncon]; c++)
		if (c->type == CAddr) {
			if (c->sym.type & SThr)
				err(msg, "thread-local symbols", fn->name);
			/* non-PIC: an extern symbol is a direct
			 * reference, which the linker resolves */
			c->sym.type &= ~SExt;
		}
	if (fn->eh_lp)
		err(msg, "landing pads", fn->name);
}

/* abi0: the refusals, then the sub-word arguments and
 * returns, extended by the caller and the callee as gcc
 * does; a sub-word parameter is a word (its high bits are
 * not trusted, as on amd64)
 */
void
i386_abi0(Fn *fn)
{
	Blk *b;
	Ins *i0, *i1, *i;
	int j, op;
	Ref r;

	refuse(fn);
	for (b=fn->start; b; b=b->link) {
		curi = &insb[NIns];
		j = b->jmp.type;
		if (isretbh(j)) {
			r = newtmp("abi", Kw, fn);
			op = Oextsb + (j - Jretsb);
			emit(op, Kw, r, b->jmp.arg, R);
			b->jmp.arg = r;
			b->jmp.type = Jretw;
		}
		for (i=&b->ins[b->nins]; i>b->ins;) {
			emiti(*--i);
			if (isparbh(i->op))
				curi->op = Opar;
			if (i->op != Ocall)
				continue;
			for (i0=i1=i; i0>b->ins; i0--)
				if (!isarg((i0-1)->op))
					break;
			for (i=i1; i>i0;) {
				emiti(*--i);
				if (isargbh(i->op)) {
					i->to = newtmp("abi", Kw, fn);
					curi->op = Oarg;
					curi->arg[0] = i->to;
				}
			}
			for (i=i1; i>i0;)
				if (isargbh((--i)->op)) {
					op = Oextsb + (i->op - Oargsb);
					emit(op, Kw, i->to, i->arg[0], R);
				}
		}
		idup(b, curi, &insb[NIns]-curi);
	}
}

void
i386_abi(Fn *fn)
{
	Blk *b;
	Ins *i, *i0;
	RAlloc *ral;
	int n0, n1, ioff;
	uint stk;

	for (b=fn->start; b; b=b->link)
		b->visit = 0;

	/* lower parameters */
	for (b=fn->start, i=b->ins; i<&b->ins[b->nins]; i++)
		if (!ispar(i->op))
			break;
	stk = selpar(fn, b->ins, i);
	n0 = &insb[NIns] - curi;
	ioff = i - b->ins;
	n1 = b->nins - ioff;
	vgrow(&b->ins, n0+n1);
	icpy(b->ins+n0, b->ins+ioff, n1);
	icpy(b->ins, curi, n0);
	b->nins = n0+n1;

	/* lower calls, returns, and vararg instructions */
	ral = 0;
	b = fn->start;
	do {
		if (!(b = b->link))
			b = fn->start; /* do it last */
		if (b->visit)
			continue;
		curi = &insb[NIns];
		selret(b, fn);
		for (i=&b->ins[b->nins]; i!=b->ins;)
			switch ((--i)->op) {
			default:
				emiti(*i);
				break;
			case Ocall:
				for (i0=i; i0>b->ins; i0--)
					if (!isarg((i0-1)->op))
						break;
				selcall(fn, i0, i, &ral);
				i = i0;
				break;
			case Ovastart:
				selvastart(fn, stk, i->arg[0]);
				break;
			case Ovaarg:
				selvaarg(fn, i);
				break;
			case Oarg:
			case Oargc:
				die("unreachable");
			}
		if (b == fn->start)
			for (; ral; ral=ral->link)
				emiti(ral->i);
		idup(b, curi, &insb[NIns]-curi);
	} while (b != fn->start);

	if (debug['A']) {
		fprintf(stderr, "\n> After ABI lowering:\n");
		printfn(fn, stderr);
	}
}
