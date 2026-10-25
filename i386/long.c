#include "all.h"

/* 64-bit integers (class l) on i386, as pairs of words
 *
 * Runs first in i386_isel, on the SSA form after the ABI
 * lowering: each temporary of class l gets two temporaries
 * of class w, its low and high words, defined where it is
 * (a phi of class l becomes two phis), and each instruction
 * of class l, or with an operand of class l, becomes word
 * instructions:
 *
 * - add and sub carry with an unsigned comparison of the low
 *   words; mul is a widening mull of the low words (Oxmul,
 *   eax:edx) plus the two cross products
 * - shifts by a constant are moves between the words; by a
 *   variable, the word shifts are combined with masks on bit
 *   5 of the amount (no branch)
 * - comparisons: on the high words, then the low words
 *   (unsigned) when the high words are equal
 * - loads and stores are two word accesses, low word first
 * - a word instruction (or jnz) using a long takes its low
 *   word, as QBE's w class does
 *
 * Divisions were calls to libgcc's __divdi3 and friends since
 * i386_abi0, as gcc makes them; i386_abi passes and returns
 * longs as two words (eax:edx).
 */

static Fn *fn;
static Ref *lo, *hi;   /* the words of each long temporary */
static char *inphi;    /* the long temporaries that phis use */
static Ins *ib;        /* the new instructions of a block */
static uint nib;

static Ref
add(int op, Ref to, Ref a0, Ref a1)
{
	Ins *i;

	vgrow(&ib, ++nib);
	i = &ib[nib-1];
	*i = (Ins){.op = op, .cls = Kw, .to = to, .arg = {a0, a1}};
	return to;
}

static Ref
tmp(void)
{
	return newtmp("long", Kw, fn);
}

static int conval(Ref, int64_t *);

/* a word set to op a0, a1: a new temporary, or a constant or
 * an operand when the operation folds */
static Ref
op2(int op, Ref a0, Ref a1)
{
	int64_t x, y;
	Con c;

	if (op != Ocopy && rtype(a0) == RCon
	&& fn->con[a0.val].type == CBits
	&& (req(a1, R) || (rtype(a1) == RCon
	 && fn->con[a1.val].type == CBits))
	&& optab[op].canfold
	&& !foldint(&c, op, 0, &fn->con[a0.val],
		&fn->con[req(a1, R) ? CON_Z.val : a1.val]))
		return getcon((uint32_t)c.bits.i, fn);
	if (conval(a1, &y) && y == 0)
		switch (op) {
		case Oor: case Oxor: case Oadd: case Osub:
		case Oshl: case Oshr: case Osar:
			return a0;
		case Oand: case Omul:
			return CON_Z;
		}
	if (conval(a0, &x) && x == 0)
		switch (op) {
		case Oor: case Oxor: case Oadd:
			return a1;
		case Oand: case Omul: case Oshl: case Oshr: case Osar:
			return CON_Z;
		}
	return add(op, tmp(), a0, a1);
}

static int
islong(Ref r)
{
	return rtype(r) == RTmp && r.val >= Tmp0
		&& fn->tmp[r.val].cls == Kl;
}

/* the word w (0 low, 1 high) of the long operand r */
static Ref
word(Ref r, int w)
{
	Con *c;

	switch (rtype(r)) {
	case RTmp:
		if (!islong(r))
			die("word temporary %%%s used as a long",
				fn->tmp[r.val].name);
		return w ? hi[r.val] : lo[r.val];
	case RCon:
		c = &fn->con[r.val];
		switch (c->type) {
		case CUndef:
			return r;
		case CBits:
			if (w)
				return getcon(c->bits.i >> 32, fn);
			return getcon((uint32_t)c->bits.i, fn);
		case CAddr:
			/* an address is below 4G */
			return w ? CON_Z : r;
		}
		/* fall through */
	default:
		die("invalid long operand");
	}
}

/* the low word of r, used where a word is expected */
static Ref
low(Ref r)
{
	if (islong(r))
		return lo[r.val];
	return r;
}

/* the words of the long to: its uses other than by phis come
 * after it in the reverse post-order of the blocks, they are
 * the words directly */
static void
setw(Ref to, Ref vlo, Ref vhi)
{
	if (!inphi[to.val]) {
		lo[to.val] = vlo;
		hi[to.val] = vhi;
		return;
	}
	add(Ocopy, lo[to.val], vlo, R);
	add(Ocopy, hi[to.val], vhi, R);
}

static int
conval(Ref r, int64_t *v)
{
	return isconbits(fn, r, v);
}

static void
shift(Ins *i)
{
	Ref a0, a1, n, s, s1, m, nm, t0, t1, t2, t3, sign;
	int64_t v;
	int op;

	op = i->op;
	a0 = word(i->arg[0], 0);
	a1 = word(i->arg[0], 1);
	n = low(i->arg[1]);
	if (conval(n, &v)) {
		v &= 63;
		if (v == 0) {
			setw(i->to, a0, a1);
			return;
		}
		if (v >= 32) {
			v -= 32;
			switch (op) {
			case Oshl:
				setw(i->to, CON_Z, v ? op2(Oshl, a0, getcon(v, fn)) : a0);
				break;
			case Oshr:
				setw(i->to, v ? op2(Oshr, a1, getcon(v, fn)) : a1, CON_Z);
				break;
			case Osar:
				setw(i->to, v ? op2(Osar, a1, getcon(v, fn)) : a1,
					op2(Osar, a1, getcon(31, fn)));
				break;
			}
			return;
		}
		s = getcon(v, fn);
		s1 = getcon(32 - v, fn);
		switch (op) {
		case Oshl:
			t0 = op2(Oshl, a1, s);
			t1 = op2(Oshr, a0, s1);
			setw(i->to, op2(Oshl, a0, s), op2(Oor, t0, t1));
			break;
		case Oshr:
		case Osar:
			t0 = op2(Oshr, a0, s);
			t1 = op2(Oshl, a1, s1);
			setw(i->to, op2(Oor, t0, t1), op2(op, a1, s));
			break;
		}
		return;
	}
	/* s = n & 31; m = -((n >> 5) & 1): all ones when n >= 32
	 * (n & 63); the bits crossing between the words are
	 * shifted in two steps, 1 then 31-s, so that s = 0 moves
	 * none */
	s = op2(Oand, n, getcon(31, fn));
	t0 = op2(Oshr, n, getcon(5, fn));
	t0 = op2(Oand, t0, getcon(1, fn));
	m = op2(Osub, CON_Z, t0);
	nm = op2(Oxor, m, getcon(-1, fn));
	s1 = op2(Osub, getcon(31, fn), s);
	switch (op) {
	case Oshl:
		t0 = op2(Oshl, a0, s);           /* low word shifted */
		t1 = op2(Oshr, a0, getcon(1, fn));
		t1 = op2(Oshr, t1, s1);
		t2 = op2(Oshl, a1, s);
		t2 = op2(Oor, t2, t1);           /* high word, s < 32 */
		t3 = op2(Oand, t0, nm);
		t2 = op2(Oand, t2, nm);
		t1 = op2(Oand, t0, m);
		setw(i->to, t3, op2(Oor, t2, t1));
		break;
	case Oshr:
	case Osar:
		t0 = op2(op, a1, s);             /* high word shifted */
		t1 = op2(Oshl, a1, getcon(1, fn));
		t1 = op2(Oshl, t1, s1);
		t2 = op2(Oshr, a0, s);
		t2 = op2(Oor, t2, t1);           /* low word, s < 32 */
		t2 = op2(Oand, t2, nm);
		t3 = op2(Oand, t0, m);
		t2 = op2(Oor, t2, t3);
		t0 = op2(Oand, t0, nm);
		if (op == Osar) {
			sign = op2(Osar, a1, getcon(31, fn));
			sign = op2(Oand, sign, m);
			t0 = op2(Oor, t0, sign);
		}
		setw(i->to, t2, t0);
		break;
	}
}

/* the widening product of two words: *plo, *phi */
static void
umul(Ref a, Ref b, Ref *plo, Ref *phi)
{
	Ref r;

	if (rtype(b) != RTmp) {
		r = b;
		b = a;
		a = r;
	}
	if (rtype(b) != RTmp)
		b = op2(Ocopy, b, R);
	/* eax and edx are live across the mull, so that
	 * neither holds b: the copies from them stay even
	 * when unused (isel keeps a copy from a register) */
	add(Ocopy, TMP(EAX), a, R);
	add(Ocopy, TMP(EDX), CON_Z, R);
	add(Oxmul, R, b, R);
	*phi = add(Ocopy, tmp(), TMP(EDX), R);
	*plo = add(Ocopy, tmp(), TMP(EAX), R);
}

static void
cmp(Ins *i, int c)
{
	Ref a0, a1, b0, b1, t0, t1, t2;
	int chi, clo;

	a0 = word(i->arg[0], 0);
	a1 = word(i->arg[0], 1);
	b0 = word(i->arg[1], 0);
	b1 = word(i->arg[1], 1);
	switch (c) {
	case Cieq:
	case Cine:
		t0 = op2(Oxor, a0, b0);
		t1 = op2(Oxor, a1, b1);
		t0 = op2(Oor, t0, t1);
		add(c == Cieq ? Oceqw : Ocnew, i->to, t0, CON_Z);
		return;
	}
	/* strict on the high words, the low words decide
	 * (unsigned) when they are equal */
	switch (c) {
	case Cislt: chi = Cislt, clo = Ciult; break;
	case Cisle: chi = Cislt, clo = Ciule; break;
	case Cisgt: chi = Cisgt, clo = Ciugt; break;
	case Cisge: chi = Cisgt, clo = Ciuge; break;
	case Ciult: chi = Ciult, clo = Ciult; break;
	case Ciule: chi = Ciult, clo = Ciule; break;
	case Ciugt: chi = Ciugt, clo = Ciugt; break;
	case Ciuge: chi = Ciugt, clo = Ciuge; break;
	default: die("unreachable");
	}
	t0 = op2(Ocmpw+chi, a1, b1);
	t1 = op2(Oceqw, a1, b1);
	t2 = op2(Ocmpw+clo, a0, b0);
	t1 = op2(Oand, t1, t2);
	add(Oor, i->to, t0, t1);
}

static Ref
addr4(Ref p)
{
	return op2(Oadd, p, getcon(4, fn));
}

static void
lins(Ins *i)
{
	Ref a0, a1, b0, b1, t0, t1, t2, t3;
	int k, c, n;

	if (iscmp(i->op, &k, &c) && k == Kl) {
		cmp(i, c);
		return;
	}
	if (i->op == Ostorel) {
		t0 = word(i->arg[0], 0);
		t1 = word(i->arg[0], 1);
		add(Ostorew, R, t0, i->arg[1]);
		add(Ostorew, R, t1, addr4(i->arg[1]));
		return;
	}
	if (i->cls != Kl || i->op == Ocall || i->op == Onop
	|| isstore(i->op) || i->op == Odbgloc) {
		/* a word instruction: a long operand is its low word */
		for (n=0; n<2; n++)
			if (islong(i->arg[n])) {
				if (argcls(i, n) == Kl)
					die("unhandled long operand in %s",
						optab[i->op].name);
				i->arg[n] = lo[i->arg[n].val];
			}
		vgrow(&ib, ++nib);
		ib[nib-1] = *i;
		ib[nib-1].cls = i->cls == Kl ? Kw : i->cls;
		return;
	}
	if (rtype(i->to) != RTmp)
		die("long %s without a result", optab[i->op].name);
	switch (i->op) {
	case Ocopy:
		setw(i->to, word(i->arg[0], 0), word(i->arg[0], 1));
		break;
	case Oadd:
	case Osub:
		a0 = word(i->arg[0], 0);
		a1 = word(i->arg[0], 1);
		b0 = word(i->arg[1], 0);
		b1 = word(i->arg[1], 1);
		t0 = op2(i->op, a0, b0);
		if (i->op == Oadd)
			t1 = op2(Ocultw, t0, a0);    /* carry */
		else
			t1 = op2(Ocultw, a0, b0);    /* borrow */
		t2 = op2(i->op, a1, b1);
		setw(i->to, t0, op2(i->op, t2, t1));
		break;
	case Oneg:
		a0 = word(i->arg[0], 0);
		a1 = word(i->arg[0], 1);
		t0 = op2(Oneg, a0, R);
		t1 = op2(Ocnew, a0, CON_Z);
		t2 = op2(Oneg, a1, R);
		setw(i->to, t0, op2(Osub, t2, t1));
		break;
	case Oand:
	case Oor:
	case Oxor:
		t0 = op2(i->op, word(i->arg[0], 0), word(i->arg[1], 0));
		t1 = op2(i->op, word(i->arg[0], 1), word(i->arg[1], 1));
		setw(i->to, t0, t1);
		break;
	case Omul:
		a0 = word(i->arg[0], 0);
		a1 = word(i->arg[0], 1);
		b0 = word(i->arg[1], 0);
		b1 = word(i->arg[1], 1);
		/* the cross products first: fewer words are live
		 * across the mull, which takes eax and edx */
		t2 = op2(Omul, a0, b1);
		t3 = op2(Omul, a1, b0);
		t2 = op2(Oadd, t2, t3);
		umul(a0, b0, &t0, &t1);
		setw(i->to, t0, op2(Oadd, t1, t2));
		break;
	case Oshl:
	case Oshr:
	case Osar:
		shift(i);
		break;
	case Oextsw:
	case Oextuw:
		a0 = low(i->arg[0]);
		t1 = i->op == Oextsw ? op2(Osar, a0, getcon(31, fn)) : CON_Z;
		setw(i->to, a0, t1);
		break;
	case Oextsb:
	case Oextub:
	case Oextsh:
	case Oextuh:
		t0 = op2(i->op, low(i->arg[0]), R);
		if (i->op == Oextsb || i->op == Oextsh)
			t1 = op2(Osar, t0, getcon(31, fn));
		else
			t1 = CON_Z;
		setw(i->to, t0, t1);
		break;
	case Oload:
		t0 = op2(Oload, i->arg[0], R);
		t1 = op2(Oload, addr4(i->arg[0]), R);
		setw(i->to, t0, t1);
		break;
	case Oloadsw:
	case Oloaduw:
	case Oloadsb:
	case Oloadub:
	case Oloadsh:
	case Oloaduh:
		t0 = op2(i->op, i->arg[0], R);
		if (i->op == Oloadsw || i->op == Oloadsb || i->op == Oloadsh)
			t1 = op2(Osar, t0, getcon(31, fn));
		else
			t1 = CON_Z;
		setw(i->to, t0, t1);
		break;
	default:
		die("long %s is not supported on i386", optab[i->op].name);
	}
}

void
i386_long(Fn *f)
{
	Blk *b;
	Phi *p, *p1;
	Ins *i;
	uint n, nt, t, r;
	int any;

	fn = f;
	any = 0;
	nt = fn->ntmp;
	for (t=Tmp0; t<nt; t++)
		if (fn->tmp[t].cls == Kl)
			any = 1;
	if (!any)
		return;
	lo = emalloc(nt * sizeof lo[0]);
	hi = emalloc(nt * sizeof hi[0]);
	inphi = emalloc(nt);
	for (b=fn->start; b; b=b->link)
		for (p=b->phi; p; p=p->link)
			for (n=0; n<p->narg; n++)
				if (islong(p->arg[n]))
					inphi[p->arg[n].val] = 1;
	for (t=Tmp0; t<nt; t++)
		if (fn->tmp[t].cls == Kl) {
			lo[t] = newtmp("lo", Kw, fn);
			hi[t] = newtmp("hi", Kw, fn);
		}
	ib = vnew(0, sizeof ib[0], PFn);
	for (r=0; r<fn->nblk; r++) {
		b = fn->rpo[r];
		for (p=b->phi; p; p=p->link) {
			if (p->cls != Kl) {
				for (n=0; n<p->narg; n++)
					p->arg[n] = low(p->arg[n]);
				continue;
			}
			p1 = alloc(sizeof *p1);
			*p1 = *p;
			p1->arg = vnew(p->narg, sizeof p->arg[0], PFn);
			p1->blk = vnew(p->narg, sizeof p->blk[0], PFn);
			memcpy(p1->blk, p->blk, p->narg * sizeof p->blk[0]);
			for (n=0; n<p->narg; n++) {
				p1->arg[n] = word(p->arg[n], 1);
				p->arg[n] = word(p->arg[n], 0);
			}
			p1->to = hi[p->to.val];
			p->to = lo[p->to.val];
			p->cls = p1->cls = Kw;
			p1->link = p->link;
			p->link = p1;
			p = p1;
		}
		nib = 0;
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			lins(i);
		b->jmp.arg = low(b->jmp.arg);
		if (b->jmp.type == Jjnz && rtype(b->jmp.arg) == RCon)
			b->jmp.arg = add(Ocopy, tmp(), b->jmp.arg, R);
		idup(b, ib, nib);
	}
	for (t=Tmp0; t<nt; t++)
		if (fn->tmp[t].cls == Kl)
			fn->tmp[t].cls = Kw; /* now unused */
	free(lo);
	free(hi);
	free(inphi);
	filluse(fn);
	if (debug['I']) {
		fprintf(stderr, "\n> After splitting the longs:\n");
		printfn(fn, stderr);
	}
}
