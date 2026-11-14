/* Loop vectorizer (nfcxx, -V; amd64_sysv with SSE2).
 *
 * It runs after the SSA passes and before instruction selection, and changes a loop
 * whose body is element-wise over arrays into a vector loop in front of the scalar one:
 *
 *	P:   ...                              the entry of H from outside the loop
 *	     jmp H                            (P -> Chk -> Hv when the loop stores to an array)
 *	Chk: ov = overlap of the stored arrays with the others; jnz ov, H, Hv
 *	Hv:  iv = phi(P: i0, Bv: ivn)         the vector header: runs while i + VF <= n
 *	     tt = add iv, VF
 *	     cc = cslew tt, n
 *	     jnz cc, Bv, H
 *	Bv:  the body on lanes iv .. iv+VF-1
 *	     ivn = add iv, VF
 *	     jmp Hv
 *	H:   i = phi(.. Hv: iv ..)            the scalar loop, unchanged: it runs the rest
 *	     c = csltw i, n
 *	     jnz c, B, E
 *	B:   ... the scalar body ...
 *	L:   i2 = add i, 1
 *	     jmp H
 *
 * Recognized loops (anything else is left alone):
 *   - H: one phi i (class w, the only phi), one instruction c = csltw i, n with n a
 *     temporary of class w defined outside the loop, or a constant; jnz c, B, E.
 *   - H has two predecessors: P outside the loop, and the latch L.
 *   - B up to L is a chain of blocks, each with one predecessor, no phi, and a jmp; L jumps to H.
 *   - Body instructions are one of:
 *       index arithmetic on i: extsw, extuw, mul or shl by a constant (class l);
 *       an address add base, off with base a loop-invariant temporary (class l) and off an
 *       index whose step per i equals the element size of the access;
 *       a load of a w, s or d element at such an address, giving a vector lane;
 *       w add, sub, and, or, xor, or s or d add, sub, mul, div on vector lanes, invariant
 *       scalars (broadcast) and constants, with at least one lane operand;
 *       a store of a lane of the same width at such an address;
 *       the latch i2 = add i, 1.
 *   - No other instruction, phi, call, or memory access. A value defined in the body is
 *     used only in the body (the latch aside: H's phi uses it).
 *   - All lanes of a loop have one width: 2 for doubles, else 4; w/s with d is refused.
 *   - An array (base) accessed with two different element sizes is refused.
 *   - When the loop stores and touches more than one base, Chk tests at run time that the
 *     index range [base, base + n*size) of each stored base does not overlap any other base's
 *     range; if one does, the scalar loop runs. Ranges start at the base: a superset of the
 *     accessed elements, so the test is conservative.
 *
 * The vector loop runs while iv + VF <= n: each lane i < n of the scalar loop. A bound near
 * INT_MAX can overflow iv + VF; the scalar loop has the same limit at i + 1.
 */

#include "all.h"

/* lane kinds of a vector value */
enum {
	LNone,
	LW,	/* 4 x int32 */
	LS,	/* 4 x float */
	LD,	/* 2 x double */
};

enum { NBREF = 8 };	/* the distinct bases of a loop */

typedef struct Loop Loop;
struct Loop {
	Blk *H, *P, *L;
	Blk *body[17];
	uint nbody;
	Phi *ph;
	int ia;		/* index in ph of the value from P; 1 - ia is the latch's */
	Ref i;		/* the induction temporary (ph->to) */
	Ref lat;	/* the latch value i2 */
	Ref n;		/* the bound */
	int vf;		/* lanes per vector */
	int nst;	/* stores */
	uint nt;	/* fn->ntmp when the loop was found */
	char *inl;	/* per temporary: defined in the loop */
	char *dch;	/* per temporary: defined in the body (not the latch) */
	char *idx;	/* per temporary: an index derived from i */
	char *adr;	/* per temporary: an address (base + index) */
	char *lane;	/* per temporary: a vector value, its lane kind */
	int64_t *step;	/* per index or address: its step in bytes per i */
	Ref *base;	/* per address: its base */
	char *inb;	/* per block id: in the loop (header or body) */
	Ref *map;	/* build: per temporary, its vector value or clone */
	int nbref;	/* the distinct bases of the accesses */
	Ref bref[NBREF];
	int bsz[NBREF];	/* per base: the element size */
	int bst[NBREF];	/* per base: stored to */
};

struct Stats {
	int nld, nst;
	int hasd, hasws;
	int nvec;
};

/* a loop the pass refuses: under -d V, the line of the refusal */
static int
rej(int line)
{
	if (debug['V'])
		fprintf(stderr, "vec: refused at vec.c:%d\n", line);
	return 0;
}

static int
conint(Fn *fn, Ref r, int64_t *v)
{
	if (rtype(r) != RCon || fn->con[r.val].type != CBits)
		return 0;
	*v = fn->con[r.val].bits.i;
	return 1;
}

static int
isone(Fn *fn, Ref r)
{
	int64_t v;

	return conint(fn, r, &v) && v == 1;
}

static int
isidx(Loop *lp, Ref r)
{
	return rtype(r) == RTmp && lp->idx[r.val];
}

static int
isadr(Loop *lp, Ref r)
{
	return rtype(r) == RTmp && lp->adr[r.val];
}

/* a loop-invariant temporary of class k */
static int
isinv(Fn *fn, Loop *lp, Ref r, int k)
{
	return rtype(r) == RTmp && !lp->inl[r.val] && fn->tmp[r.val].cls == k;
}

static int
vecop(int op, int kind)
{
	switch (kind) {
	case LW:
		switch (op) {
		case Oadd: return Ovecaddw;
		case Osub: return Ovecsubw;
		case Oand: return Ovecandw;
		case Oor:  return Ovecorw;
		case Oxor: return Ovecxorw;
		}
		break;
	case LS:
		switch (op) {
		case Oadd: return Ovecadds;
		case Osub: return Ovecsubs;
		case Omul: return Ovecmuls;
		case Odiv: return Ovecdivs;
		}
		break;
	case LD:
		switch (op) {
		case Oadd: return Ovecaddd;
		case Osub: return Ovecsubd;
		case Omul: return Ovecmuld;
		case Odiv: return Ovecdivd;
		}
		break;
	}
	return -1;
}

/* an element access at address a: the address steps by the element size, and the base has one size */
static int
access(Loop *lp, Ref a, int size, int kind, int store, struct Stats *st)
{
	Ref b;
	int j;

	if (!isadr(lp, a) || lp->step[a.val] != size)
		return 0;
	b = lp->base[a.val];
	for (j=0; j<lp->nbref; j++)
		if (req(lp->bref[j], b))
			break;
	if (j == lp->nbref) {
		if (lp->nbref == NBREF)
			return 0;
		lp->bref[j] = b;
		lp->bsz[j] = size;
		lp->bst[j] = 0;
		lp->nbref++;
	} else if (lp->bsz[j] != size)
		return 0;
	if (store)
		lp->bst[j] = 1;
	if (kind == LD)
		st->hasd = 1;
	else
		st->hasws = 1;
	st->nvec++;
	return 1;
}

/* check one body instruction, and record what it defines */
static int
classify(Fn *fn, Loop *lp, Ins *ins, struct Stats *st)
{
	int64_t c;
	int kind, size, k, j, nlane;
	Ref a0, a1, to, a[2];
	int tn;

	to = ins->to;
	tn = rtype(to) == RTmp ? to.val : -1;
	a0 = ins->arg[0];
	a1 = ins->arg[1];

	/* index arithmetic on i */
	if ((ins->op == Oextsw || ins->op == Oextuw) && ins->cls == Kl && isidx(lp, a0)) {
		if (tn < 0) return 0;
		lp->idx[tn] = 1;
		lp->step[tn] = lp->step[a0.val];
		lp->dch[tn] = 1;
		return 1;
	}
	if (ins->op == Omul && ins->cls == Kl) {
		if (isidx(lp, a0) && conint(fn, a1, &c))
			;
		else if (isidx(lp, a1) && conint(fn, a0, &c))
			a0 = a1;
		else
			return 0;
		if (tn < 0 || c > (1 << 20) || c < -(1 << 20))
			return 0;
		lp->idx[tn] = 1;
		lp->step[tn] = lp->step[a0.val] * c;
		lp->dch[tn] = 1;
		return 1;
	}
	if (ins->op == Oshl && ins->cls == Kl && isidx(lp, a0) && conint(fn, a1, &c)) {
		if (tn < 0 || c < 0 || c > 3)
			return 0;
		lp->idx[tn] = 1;
		lp->step[tn] = lp->step[a0.val] << c;
		lp->dch[tn] = 1;
		return 1;
	}
	/* an address: an index plus a loop-invariant base */
	if (ins->op == Oadd && ins->cls == Kl) {
		if (tn < 0)
			return 0;
		if (isidx(lp, a0) && isinv(fn, lp, a1, Kl)) {
			lp->step[tn] = lp->step[a0.val];
			lp->base[tn] = a1;
		} else if (isidx(lp, a1) && isinv(fn, lp, a0, Kl)) {
			lp->step[tn] = lp->step[a1.val];
			lp->base[tn] = a0;
		} else
			return 0;
		lp->adr[tn] = 1;
		lp->dch[tn] = 1;
		return 1;
	}
	/* element stores: the value is a lane of the same width */
	if (ins->op == Ostorew || ins->op == Ostores || ins->op == Ostored) {
		kind = ins->op == Ostorew ? LW : ins->op == Ostores ? LS : LD;
		size = kind == LD ? 8 : 4;
		if (rtype(a0) != RTmp || lp->lane[a0.val] != kind)
			return 0;
		if (!access(lp, a1, size, kind, 1, st))
			return 0;
		st->nst++;
		return 1;
	}
	/* element loads */
	if (ins->op == Oloadsw && ins->cls == Kw) {
		kind = LW; size = 4;
	} else if (ins->op == Oload && ins->cls == Ks) {
		kind = LS; size = 4;
	} else if (ins->op == Oload && ins->cls == Kd) {
		kind = LD; size = 8;
	} else
		kind = LNone;
	if (kind != LNone) {
		if (tn < 0 || !access(lp, a0, size, kind, 0, st))
			return 0;
		st->nld++;
		lp->lane[tn] = kind;
		lp->dch[tn] = 1;
		return 1;
	}
	/* arithmetic on lanes */
	k = ins->cls;
	kind = k == Kw ? LW : k == Ks ? LS : k == Kd ? LD : LNone;
	if (kind == LNone || vecop(ins->op, kind) < 0 || tn < 0)
		return 0;
	a[0] = a0;
	a[1] = a1;
	nlane = 0;
	for (j=0; j<2; j++) {
		if (rtype(a[j]) == RTmp && lp->lane[a[j].val] == kind)
			nlane++;
		else if (isinv(fn, lp, a[j], k) || (rtype(a[j]) == RCon && conint(fn, a[j], &c)))
			;
		else
			return 0;
	}
	if (nlane == 0)
		return 0;
	lp->lane[tn] = kind;
	lp->dch[tn] = 1;
	st->nvec++;
	if (kind == LD)
		st->hasd = 1;
	else
		st->hasws = 1;
	return 1;
}

/* find and check the loop headed by H; fills lp, true when H heads a loop to vectorize */
static int
findloop(Fn *fn, Blk *H, Loop *lp)
{
	Blk *B, *L, *P, *b;
	Phi *ph;
	Ins *cmp, *ins;
	struct Stats st;
	uint k, t, nt, nb, j;
	int ia, found;
	Use *u;

	memset(&st, 0, sizeof st);
	if (H->jmp.type != Jjnz || H->s1 == H || H->s2 == H || H->npred != 2 || H->nins != 1 || !H->phi)
		return rej(__LINE__);
	ph = H->phi;
	if (ph->link || ph->narg != 2 || ph->cls != Kw)
		return rej(__LINE__);
	cmp = &H->ins[0];
	if (cmp->op != Ocsltw || cmp->cls != Kw || !req(H->jmp.arg, cmp->to) || !req(cmp->arg[0], ph->to))
		return rej(__LINE__);
	if (fn->tmp[cmp->to.val].nuse != 1)
		return rej(__LINE__);
	if (rtype(cmp->arg[1]) == RCon && fn->con[cmp->arg[1].val].type != CBits)
		return rej(__LINE__);
	if (rtype(cmp->arg[1]) != RTmp && rtype(cmp->arg[1]) != RCon)
		return rej(__LINE__);

	/* the body chain: one predecessor each, no phi, a jmp; the last one jumps back to H */
	lp->nbody = 0;
	B = H->s1;
	for (;;) {
		if (!B || B == H || B->npred != 1 || B->phi || B->jmp.type != Jjmp || lp->nbody >= 16)
			return rej(__LINE__);
		lp->body[lp->nbody++] = B;
		if (B->s1 == H)
			break;
		B = B->s1;
	}
	L = lp->body[lp->nbody-1];
	if (H->pred[0] == L)
		P = H->pred[1];
	else if (H->pred[1] == L)
		P = H->pred[0];
	else
		return rej(__LINE__);
	if (P->jmp.type == Jjtab || (P->s1 != H && P->s2 != H))
		return rej(__LINE__);
	if (ph->blk[0] == P)
		ia = 0;
	else if (ph->blk[1] == P)
		ia = 1;
	else
		return rej(__LINE__);
	if (ph->blk[1-ia] != L)
		return rej(__LINE__);

	lp->H = H;
	lp->P = P;
	lp->L = L;
	lp->ph = ph;
	lp->ia = ia;
	lp->i = ph->to;
	lp->lat = ph->arg[1-ia];
	lp->n = cmp->arg[1];
	lp->vf = 0;
	lp->nbref = 0;
	nt = fn->ntmp;
	nb = fn->nblk;
	lp->nt = nt;
	lp->inl = emalloc(nt);
	lp->dch = emalloc(nt);
	lp->idx = emalloc(nt);
	lp->adr = emalloc(nt);
	lp->lane = emalloc(nt);
	lp->step = emalloc(nt * sizeof lp->step[0]);
	lp->base = emalloc(nt * sizeof lp->base[0]);
	lp->inb = emalloc(nb);
	lp->map = emalloc(nt * sizeof lp->map[0]);

	for (k=0; k<lp->nbody; k++)
		lp->inb[lp->body[k]->id] = 1;
	lp->inb[H->id] = 1;

	lp->inl[lp->i.val] = 1;
	lp->inl[cmp->to.val] = 1;
	lp->idx[lp->i.val] = 1;
	lp->step[lp->i.val] = 1;
	for (k=0; k<lp->nbody; k++) {
		b = lp->body[k];
		for (ins=b->ins; ins<&b->ins[b->nins]; ins++)
			if (rtype(ins->to) == RTmp)
				lp->inl[ins->to.val] = 1;
	}
	/* the bound is loop-invariant: the vector header tests it */
	if (rtype(lp->n) == RTmp && (lp->inl[lp->n.val] || fn->tmp[lp->n.val].cls != Kw))
		return rej(__LINE__);

	/* the body */
	found = 0;
	for (k=0; k<lp->nbody; k++) {
		b = lp->body[k];
		for (ins=b->ins; ins<&b->ins[b->nins]; ins++) {
			if (ins->op == Onop || ins->op == Odbgloc)
				continue;
			if (ins->op == Oadd && ins->cls == Kw && req(ins->to, lp->lat)) {
				if (!req(ins->arg[0], lp->i) || !isone(fn, ins->arg[1]))
					return rej(__LINE__);
				found = 1;
				continue;
			}
			if (!classify(fn, lp, ins, &st))
				return rej(__LINE__);
		}
	}
	if (!found || st.nvec == 0 || st.nld + st.nst == 0)
		return rej(__LINE__);
	if (st.hasd && st.hasws)
		return rej(__LINE__);
	lp->vf = st.hasd ? 2 : 4;
	lp->nst = st.nst;

	/* a value of the body is used only in the body; the latch's use is in H's phi */
	for (t=Tmp0; t<nt; t++) {
		if (!lp->dch[t])
			continue;
		for (j=0, u=fn->tmp[t].use; j<fn->tmp[t].nuse; j++, u++)
			if (u->bid >= nb || !lp->inb[u->bid])
				return rej(__LINE__);
	}
	return 1;
}

/* build: the instructions of the block being made, in order (idup copies them) */
static Ins *ib;
static uint nib;

static void
push(int op, int cls, Ref to, Ref a0, Ref a1)
{
	vgrow(&ib, nib+1);
	ib[nib++] = (Ins){.op = op, .cls = cls, .to = to, .arg = {a0, a1}};
}

static Ref
newvec(Fn *fn)
{
	Ref v;

	v = newtmp("vec", Kd, fn);
	fn->tmp[v.val].vec = 1;
	return v;
}

/* an address or index operand: the vector loop's clone of it, or the operand itself */
static Ref
cloneref(Loop *lp, Ref r)
{
	if (rtype(r) == RTmp && r.val < lp->nt && (lp->idx[r.val] || lp->adr[r.val]))
		return lp->map[r.val];
	return r;
}

/* a lane operand: a lane, the broadcast of an invariant, or the broadcast of a constant */
static Ref
laneop(Fn *fn, Loop *lp, Ref r, int cls, int kind)
{
	Ref s, v;
	int bop;

	if (rtype(r) == RTmp && r.val < lp->nt && lp->lane[r.val])
		return lp->map[r.val];
	if (rtype(r) == RTmp && r.val < lp->nt && !req(lp->map[r.val], R))
		return lp->map[r.val];	/* the broadcast of an invariant, already made */
	bop = kind == LW ? Ovecbcastw : kind == LS ? Ovecbcasts : Ovecbcastd;
	if (rtype(r) == RTmp) {
		s = r;
	} else {
		s = newtmp("vec", cls, fn);
		push(Ocopy, cls, s, r, R);
	}
	v = newvec(fn);
	push(bop, Kd, v, s, R);
	if (rtype(r) == RTmp && r.val < lp->nt)
		lp->map[r.val] = v;
	return v;
}

/* the overlap test of the stored bases with the others, over [base, base + n*size) */
static Ref
overlaps(Fn *fn, Loop *lp)
{
	Ref nl, s, x, lens, lenx, ends, endx, c1, c2, o, ov, sum;
	int a, b2;

	nl = newtmp("vec", Kl, fn);
	if (rtype(lp->n) == RCon)	/* a constant bound: its value as a long, not an extsw of a constant */
		push(Ocopy, Kl, nl, getcon((int32_t)fn->con[lp->n.val].bits.i, fn), R);
	else
		push(Oextsw, Kl, nl, lp->n, R);
	ov = R;
	for (a=0; a<lp->nbref; a++) {
		if (!lp->bst[a])
			continue;
		for (b2=0; b2<lp->nbref; b2++) {
			if (b2 == a)
				continue;
			s = lp->bref[a];
			x = lp->bref[b2];
			lens = newtmp("vec", Kl, fn);
			push(Omul, Kl, lens, nl, getcon(lp->bsz[a], fn));
			ends = newtmp("vec", Kl, fn);
			push(Oadd, Kl, ends, s, lens);
			lenx = newtmp("vec", Kl, fn);
			push(Omul, Kl, lenx, nl, getcon(lp->bsz[b2], fn));
			endx = newtmp("vec", Kl, fn);
			push(Oadd, Kl, endx, x, lenx);
			/* [s, ends) and [x, endx) overlap when s < endx and x < ends */
			c1 = newtmp("vec", Kw, fn);
			push(Ocsltl, Kl, c1, s, endx);
			c2 = newtmp("vec", Kw, fn);
			push(Ocsltl, Kl, c2, x, ends);
			o = newtmp("vec", Kw, fn);
			push(Oand, Kw, o, c1, c2);
			if (req(ov, R))
				ov = o;
			else {
				sum = newtmp("vec", Kw, fn);
				push(Oor, Kw, sum, ov, o);
				ov = sum;
			}
		}
	}
	return ov;
}

static void
build(Fn *fn, Loop *lp)
{
	Blk *Hv, *Bv, *Chk, *P, *H, *b;
	Phi *ph, *hp;
	Ref iv, ivn, tt, cc, init, t, a0, a1, ov;
	Ins *ins;
	uint k;
	int a, kind, check;

	H = lp->H;
	P = lp->P;
	ph = lp->ph;
	/* a test is needed when the loop stores and touches some other base too */
	check = lp->nst > 0 && lp->nbref > 1;

	Hv = newblk();
	Bv = newblk();
	Chk = check ? newblk() : 0;
	iv = newtmp("vec", Kw, fn);
	ivn = newtmp("vec", Kw, fn);
	tt = newtmp("vec", Kw, fn);
	cc = newtmp("vec", Kw, fn);
	init = ph->arg[lp->ia];

	/* block order: P, [Chk,] Hv, Bv, then the rest */
	Bv->link = P->link;
	Hv->link = Bv;
	if (check) {
		Chk->link = Hv;
		P->link = Chk;
	} else
		P->link = Hv;

	/* H's phi: the entry from P now comes from Chk (or Hv); the vector loop's exit comes from Hv */
	for (a=0; a<2; a++)
		if (ph->blk[a] == P)
			ph->blk[a] = check ? Chk : Hv;
	if (!check) {
		for (a=0; a<2; a++)
			if (ph->blk[a] == Hv)
				ph->arg[a] = iv;
	} else {
		vgrow(&ph->arg, 3);
		vgrow(&ph->blk, 3);
		ph->arg[2] = iv;
		ph->blk[2] = Hv;
		ph->narg = 3;
	}

	/* Hv: iv = phi(P or Chk: init, Bv: ivn) */
	hp = alloc(sizeof *hp);
	hp->to = iv;
	hp->cls = Kw;
	hp->narg = 2;
	hp->visit = 0;
	hp->link = 0;
	hp->arg = vnew(2, sizeof hp->arg[0], PFn);
	hp->blk = vnew(2, sizeof hp->blk[0], PFn);
	hp->arg[0] = init;
	hp->blk[0] = check ? Chk : P;
	hp->arg[1] = ivn;
	hp->blk[1] = Bv;
	Hv->phi = hp;

	if (P->s1 == H)
		P->s1 = check ? Chk : Hv;
	if (P->s2 == H)
		P->s2 = check ? Chk : Hv;

	if (check) {
		nib = 0;
		ov = overlaps(fn, lp);
		idup(Chk, ib, nib);
		Chk->jmp.type = Jjnz;
		Chk->jmp.arg = ov;
		Chk->s1 = H;
		Chk->s2 = Hv;
	}

	/* Hv: tt = iv + VF; cc = tt <= n; jnz cc, Bv, H */
	nib = 0;
	push(Oadd, Kw, tt, iv, getcon(lp->vf, fn));
	push(Ocslew, Kw, cc, tt, lp->n);
	idup(Hv, ib, nib);
	Hv->jmp.type = Jjnz;
	Hv->jmp.arg = cc;
	Hv->s1 = Bv;
	Hv->s2 = H;

	/* Bv: the body on the lanes */
	memset(lp->map, 0, lp->nt * sizeof lp->map[0]);
	lp->map[lp->i.val] = iv;
	nib = 0;
	for (k=0; k<lp->nbody; k++) {
		b = lp->body[k];
		for (ins=b->ins; ins<&b->ins[b->nins]; ins++) {
			if (ins->op == Onop || ins->op == Odbgloc)
				continue;
			if (ins->op == Oadd && ins->cls == Kw && req(ins->to, lp->lat))
				continue;
			/* index and address arithmetic: cloned on iv */
			if (rtype(ins->to) == RTmp && (lp->idx[ins->to.val] || lp->adr[ins->to.val])) {
				t = newtmp("vec", ins->cls, fn);
				push(ins->op, ins->cls, t, cloneref(lp, ins->arg[0]), cloneref(lp, ins->arg[1]));
				lp->map[ins->to.val] = t;
				continue;
			}
			switch (ins->op) {
			case Oloadsw:
			case Oload:
				t = newvec(fn);
				push(Ovecload, Kd, t, cloneref(lp, ins->arg[0]), R);
				lp->map[ins->to.val] = t;
				continue;
			case Ostorew:
			case Ostores:
			case Ostored:
				push(Ovecstore, Kw, R, lp->map[ins->arg[0].val], cloneref(lp, ins->arg[1]));
				continue;
			default:
				break;
			}
			kind = ins->cls == Kw ? LW : ins->cls == Ks ? LS : LD;
			a0 = laneop(fn, lp, ins->arg[0], ins->cls, kind);
			a1 = laneop(fn, lp, ins->arg[1], ins->cls, kind);
			t = newvec(fn);
			push(vecop(ins->op, kind), Kd, t, a0, a1);
			lp->map[ins->to.val] = t;
		}
	}
	/* ivn = iv + VF; jmp Hv */
	push(Oadd, Kw, ivn, iv, getcon(lp->vf, fn));
	idup(Bv, ib, nib);
	Bv->jmp.type = Jjmp;
	Bv->s1 = Hv;
	Bv->s2 = 0;
}

void
vec(Fn *fn)
{
	Blk **cand, *b;
	uint n, nc, k;
	Loop lp;

	if (strcmp(T.name, "amd64_sysv") != 0 || nofp)
		return;
	ib = vnew(0, sizeof ib[0], PHeap);
	nib = 0;
	/* the candidates first: a loop changed by an earlier one is not in the list */
	cand = vnew(0, sizeof cand[0], PFn);
	nc = 0;
	for (n=0; n<fn->nblk; n++) {
		b = fn->rpo[n];
		if (b->jmp.type == Jjnz) {
			vgrow(&cand, nc+1);
			cand[nc++] = b;
		}
	}
	for (k=0; k<nc; k++) {
		filluse(fn);
		memset(&lp, 0, sizeof lp);
		if (findloop(fn, cand[k], &lp)) {
			if (debug['V'])
				fprintf(stderr, "vec: %s: loop at @%d, %d lanes, %d bases\n", fn->name,
					cand[k]->id, lp.vf, lp.nbref);
			build(fn, &lp);
			fillcfg(fn);
			filluse(fn);
			if (debug['V'])
				printfn(fn, stderr);	/* (ssacheck does not apply here: the ABI has made register copies) */
		}
		free(lp.inl);
		free(lp.dch);
		free(lp.idx);
		free(lp.adr);
		free(lp.lane);
		free(lp.step);
		free(lp.base);
		free(lp.inb);
		free(lp.map);
	}
	vfree(ib);
	ib = 0;
}
