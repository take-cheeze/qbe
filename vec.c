/* Loop vectorizer (nfcxx, -V; amd64_sysv with SSE2).
 *
 * It runs after the SSA passes and before instruction selection, and changes a loop
 * whose body is element-wise over arrays into a vector loop in front of the scalar one:
 *
 *	P:   ...                              the entry of H from outside the loop
 *	     jmp H                            (P -> Chk -> Hv when the loop stores to an array)
 *	Chk: ov = overlap of the stored arrays with the others; jnz ov, H, Hv
 *	Hv:  iv = phi(P: i0, Bv: ivn)         the vector header: runs while i + VF <= n
 *	     va = phi(P: vid, Bv: van)        one vector accumulator per reduction
 *	     tt = add iv, VF
 *	     cc = cslew tt, n
 *	     jnz cc, Bv, H or Rd
 *	Bv:  the body on lanes iv .. iv+VF-1
 *	     ivn = add iv, VF
 *	     jmp Hv
 *	Rd:  the lanes of each accumulator are folded into the scalar one, the value of
 *	     the scalar loop's entry (only with reductions)
 *	     jmp H
 *	H:   i = phi(.. ..), acc = phi(.. Rd: the folded value ..)   the scalar loop, unchanged
 *	     c = csltw i, n
 *	     jnz c, B, E
 *	B:   ... the scalar body ...
 *	L:   i2 = add i, 1
 *	     jmp H
 *
 * Recognized loops (anything else is left alone):
 *   - H: the induction phi i (class w), one instruction c = csltw i, n with n a
 *     temporary of class w defined outside the loop, or a constant; jnz c, B, E.
 *   - Other phis of H are integer reductions (at most NRED): acc = phi(P: init, L: acc2)
 *     with acc2 = op acc, lane, op one of add, and, or, xor, lane a w lane of the loop.
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
 *       the latch i2 = add i, 1, and the reductions' latch values.
 *   - No other instruction, phi, call, or memory access. A value defined in the body is
 *     used only in the body (the latch aside: H's phi uses it).
 *   - All lanes of a loop have one width: 2 for doubles, else 4; w/s with d is refused.
 *     Reductions need the 4 w lanes.
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
	LM,	/* a mask: the lanes of a comparison (w) */
};

enum {
	NBREF = 16,	/* the distinct bases of a loop */
	NRED = 4,	/* the reductions of a loop */
	NACC = 32,	/* the element accesses of a loop */
	NBODY = 32,	/* the blocks of a loop body */
};

typedef struct Loop Loop;
struct Loop {
	Blk *H, *P, *L;
	Blk *body[NBODY];
	int guard[NBODY];	/* per body block: 0 always, 1 on the true path, 2 on the false path */
	Blk *cb, *tb, *fb, *join;	/* the condition's block, its true and false blocks (fb 0 for a triangle), the join */
	int hascond;
	int lastsel;	/* a sel0 was seen: the sel1 after it selects by its mask */
	int nmask;	/* the masks of the body (compares used by a select or the branch) */
	Ref selm;	/* build: the mask of the last sel0 */
	Ref cond;	/* the condition: a temporary of cb's jnz */
	Ref cmask;	/* build: the mask of the condition */
	Ref vones;	/* build: all ones, for the complements of masks */
	uint nbody;
	Phi *ph;	/* the induction phi */
	int ia;		/* index in ph of the value from P; 1 - ia is the latch's */
	Ref i;		/* the induction temporary (ph->to) */
	Ref lat;	/* the latch value i2 */
	Ref n;		/* the bound */
	int ivcls;	/* the class of i and the bound: Kw or Kl */
	int cmpop;	/* the exit test: 0 lt, 1 le, 2 gt, 3 ge (signed) */
	int64_t ivstep;	/* +1 for lt and le, -1 for gt and ge */
	int ptr;	/* a pointer induction: the accesses are through i itself, stepping by unit bytes */
	int64_t unit;	/* the bytes per iteration of i (1 for an index) */
	int vf;		/* lanes per vector */
	int nst;	/* stores */
	uint nt;	/* fn->ntmp when the loop was found */
	char *inl;	/* per temporary: defined in the loop */
	char *dch;	/* per temporary: defined in the body (not the latch) */
	char *idx;	/* per temporary: an index derived from i */
	char *adr;	/* per temporary: an address (base + index) */
	char *strd;	/* per address: a stride-2 access (two vectors, shuffled) */
	int anystrd;
	int idata;	/* the induction i is used as a value (lane); its vector: vi */
	Ref vi, vin, vstep, viota;
	Ref strz, strodd;	/* build: zero, and the odd lanes' mask, for the stride-2 stores */
	char *lane;	/* per temporary: a vector value, its lane kind */
	int64_t *step;	/* per index or address: its step in bytes per i */
	int64_t *ioff;	/* per index or address: its constant byte offset */
	Ref *base;	/* per address: its base */
	char *inb;	/* per block id: in the loop (header or body) */
	Ref *map;	/* build: per temporary, its vector value or clone */
	int nbref;	/* the distinct bases of the accesses */
	int nacc;
	int conv;	/* the int-double conversions of the body */
	int64_t acc_off[NACC];	/* per access: its byte offset from its base at i = 0 */
	int acc_base[NACC];
	int acc_st[NACC];	/* stored to */
	Ref bref[NBREF];
	int bsz[NBREF];	/* per base: the element size */
	int bst[NBREF];	/* per base: stored to */
	/* reductions: the phis of H besides i */
	int nred;
	Phi *redph[NRED];
	Ref redinit[NRED];	/* the entry from P */
	Ref redlat[NRED];	/* the latch value */
	int redop[NRED];	/* Oadd, Oand, Oor or Oxor */
	int redseen[NRED];	/* the latch value was found in the body */
	int *redof;		/* per temporary: k+1 for the accumulator of reduction k, -(k+1) for its latch */
	Ref redvi[NRED];	/* build: the identity vector, the entry of the vector accumulator */
	Ref redva[NRED];	/* build: the vector accumulator (Hv's phi) */
	Ref redvn[NRED];	/* build: its next value (Bv) */
	Ref redhs[NRED];	/* build: the scalar accumulator (Hv's phi), the entry's init */
	Ref redrd[NRED];	/* build: the folded scalar value (Rd) */
	int redcls[NRED];	/* the class of the accumulator: Kw, or Ks and Kd with -F */
};

static int lanecmp(int op);
static int flanecmp(int op);
static int fcmpcls(int op);
static int issel0(Fn *fn, int tn);

struct Stats {
	int nld, nst;
	int hasd, hasws;
	int nvec;
	int conv;
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
isidx(Loop *lp, Ref r)
{
	return rtype(r) == RTmp && lp->idx[r.val];
}

static int
isadr(Loop *lp, Ref r)
{
	return rtype(r) == RTmp && (lp->adr[r.val] || (lp->ptr && lp->idx[r.val]));
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
		case Omul: return -2;	/* no SSE2 instruction: mulw() */
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

	if (!isadr(lp, a))
		return 0;
	if (lp->ptr ? (lp->step[a.val] != size && lp->step[a.val] != -size)
	            : (lp->step[a.val] != size && lp->step[a.val] != 2*size))
		return 0;
	/* a stride-2 access covers the element and its neighbour: its base is then of size 2*size */
	if (!lp->ptr && lp->step[a.val] == 2*size) {
		lp->strd[a.val] = 1;
		lp->anystrd = 1;
		size = 2*size;
	}
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
	if (lp->nacc == NACC)
		return 0;
	lp->acc_off[lp->nacc] = lp->ioff[a.val];
	lp->acc_base[lp->nacc] = j;
	lp->acc_st[lp->nacc] = store;
	lp->nacc++;
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
		lp->ioff[tn] = lp->ioff[a0.val];
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
		lp->ioff[tn] = lp->ioff[a0.val] * c;
		lp->dch[tn] = 1;
		return 1;
	}
	if (ins->op == Oshl && ins->cls == Kl && isidx(lp, a0) && conint(fn, a1, &c)) {
		if (tn < 0 || c < 0 || c > 3)
			return 0;
		lp->idx[tn] = 1;
		lp->step[tn] = lp->step[a0.val] << c;
		lp->ioff[tn] = lp->ioff[a0.val] << c;
		lp->dch[tn] = 1;
		return 1;
	}
	/* an index plus a constant: the same index, shifted by that many bytes */
	if (ins->op == Oadd && ins->cls == Kl && tn >= 0) {
		Ref ix = a0;
		if ((isidx(lp, a0) && conint(fn, a1, &c)) || (isidx(lp, a1) && conint(fn, a0, &c))) {
			if (!isidx(lp, a0))
				ix = a1;
			if (c > (1LL << 31) || c < -(1LL << 31))
				return 0;
			lp->idx[tn] = 1;
			lp->step[tn] = lp->step[ix.val];
			lp->ioff[tn] = lp->ioff[ix.val] + c;
			lp->dch[tn] = 1;
			return 1;
		}
	}
	/* an address: an index plus a loop-invariant base */
	if (ins->op == Oadd && ins->cls == Kl) {
		if (tn < 0)
			return 0;
		if (isidx(lp, a0) && isinv(fn, lp, a1, Kl)) {
			lp->step[tn] = lp->step[a0.val];
			lp->ioff[tn] = lp->ioff[a0.val];
			lp->base[tn] = a1;
		} else if (isidx(lp, a1) && isinv(fn, lp, a0, Kl)) {
			lp->step[tn] = lp->step[a1.val];
			lp->ioff[tn] = lp->ioff[a1.val];
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
		if (req(a0, lp->i))
			lp->idata = 1;
		if (!access(lp, a1, size, kind, 1, st))
			return 0;
		st->nst++;
		return 1;
	}
	/* the conversions of mixed lanes: w to d (cvtdq2pd) and d to w (cvttpd2dq); the lanes are two wide then */
	if ((ins->op == Oswtof && ins->cls == Kd && rtype(a0) == RTmp && lp->lane[a0.val] == LW) ||
	    (ins->op == Odtosi && ins->cls == Kw && rtype(a0) == RTmp && lp->lane[a0.val] == LD)) {
		if (tn < 0)
			return 0;
		lp->lane[tn] = ins->op == Oswtof ? LD : LW;
		lp->dch[tn] = 1;
		st->nvec++;
		st->conv++;
		if (lp->lane[tn] == LD)
			st->hasd = 1;
		else
			st->hasws = 1;
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
	/* the latch value of a reduction: acc2 = op acc, lane */
	if (tn >= 0 && lp->redof[tn] < 0) {
		k = -lp->redof[tn] - 1;
		kind = lp->redcls[k] == Kw ? LW : lp->redcls[k] == Ks ? LS : LD;
		if (ins->cls != lp->redcls[k])
			return 0;
		if (kind == LW ? (ins->op != Oadd && ins->op != Oand && ins->op != Oor && ins->op != Oxor)
		               : ins->op != Oadd)
			return 0;
		if (req(a0, lp->redph[k]->to) && rtype(a1) == RTmp && lp->lane[a1.val] == kind)
			;
		else if (req(a1, lp->redph[k]->to) && rtype(a0) == RTmp && lp->lane[a0.val] == kind)
			;
		else
			return 0;
		if (req(a0, lp->i) || req(a1, lp->i))
			lp->idata = 1;
		lp->redop[k] = ins->op;
		lp->redseen[k] = 1;
		st->nvec++;
		if (kind == LD)
			st->hasd = 1;
		else
			st->hasws = 1;
		return 1;
	}
	/* the select form of an if (ifopt's sel0 and sel1): sel0 names the mask, sel1 selects two lanes by it */
	if (ins->op == Osel0) {
		if (rtype(a0) != RTmp || lp->lane[a0.val] != LM)
			return 0;
		lp->lastsel = 1;
		return 1;
	}
	if (ins->op == Osel1) {
		if (ins->cls != Kw || tn < 0 || !lp->lastsel)
			return 0;
		for (j=0; j<2; j++) {
			if (rtype(ins->arg[j]) != RTmp || lp->lane[ins->arg[j].val] != LW)
				return 0;
		}
		lp->lane[tn] = LW;
		lp->dch[tn] = 1;
		st->nvec++;
		st->hasws = 1;
		return 1;
	}
	/* the mask of the if: the lanes' comparison (only the jnz or a sel0 uses it) */
	if (tn >= 0 && (lanecmp(ins->op) >= 0 || fcmpcls(ins->op) >= 0) && fn->tmp[tn].nuse == 1 &&
	    (lp->hascond ? req(to, lp->cond) : issel0(fn, tn))) {
		int ck = fcmpcls(ins->op) >= 0 ? fcmpcls(ins->op) : Kw;
		kind = ck == Kw ? LW : ck == Ks ? LS : LD;
		if ((ck == Kw ? lanecmp(ins->op) : flanecmp(ins->op)) < 0)
			return 0;
		nlane = 0;
		a[0] = a0;
		a[1] = a1;
		for (j=0; j<2; j++) {
			if (rtype(a[j]) == RTmp && lp->lane[a[j].val] == kind)
				nlane++;
			else if (isinv(fn, lp, a[j], ck) || (rtype(a[j]) == RCon && conint(fn, a[j], &c)))
				;
			else
				return 0;
		}
		if (nlane == 0)
			return 0;
		lp->lane[tn] = LM;
		lp->dch[tn] = 1;
		st->nvec++;
		st->hasws = 1;
		return 1;
	}
	/* arithmetic on lanes */
	k = ins->cls;
	kind = k == Kw ? LW : k == Ks ? LS : k == Kd ? LD : LNone;
	if (kind == LNone || vecop(ins->op, kind) == -1 || tn < 0)
		return 0;
	a[0] = a0;
	a[1] = a1;
	nlane = 0;
	for (j=0; j<2; j++) {
		if (rtype(a[j]) == RTmp && lp->lane[a[j].val] == kind) {
			nlane++;
			if (req(a[j], lp->i))
				lp->idata = 1;
		}
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

/* a lane comparison: its kind, or -1 (signed, w) */
enum { CEq, CNe, CLt, CLe, CGt, CGe };

static int
lanecmp(int op)
{
	switch (op) {
	case Oceqw: return CEq;
	case Ocnew: return CNe;
	case Ocsltw: return CLt;
	case Ocslew: return CLe;
	case Ocsgtw: return CGt;
	case Ocsgew: return CGe;
	}
	return -1;
}

/* the join's phis: each is a select of two lanes (or invariants) of the same width */
static int
classifyjoin(Fn *fn, Loop *lp, struct Stats *st)
{
	Phi *p;
	int a, kd;
	int64_t c;
	Ref r;

	for (p=lp->join->phi; p; p=p->link) {
		if (p->narg != 2)
			return 0;
		kd = p->cls == Kw ? LW : p->cls == Ks ? LS : p->cls == Kd ? LD : LNone;
		if (kd == LNone)
			return 0;
		for (a=0; a<2; a++) {
			r = p->arg[a];
			if (rtype(r) == RTmp && lp->lane[r.val] == kd)
				;
			else if (isinv(fn, lp, r, p->cls) || (rtype(r) == RCon && conint(fn, r, &c)))
				;
			else
				return 0;
		}
		lp->lane[p->to.val] = kd;
		lp->dch[p->to.val] = 1;
		st->nvec++;
		if (kd == LD)
			st->hasd = 1;
		else
			st->hasws = 1;
	}
	return 1;
}

/* a float comparison (ordered): its kind as for lanecmp, or -1 */
static int
flanecmp(int op)
{
	switch (op) {
	case Oceqs: case Oceqd: return CEq;
	case Oclts: case Ocltd: return CLt;
	case Ocles: case Ocled: return CLe;
	case Ocgts: case Ocgtd: return CGt;
	case Ocges: case Ocged: return CGe;
	}
	return -1;
}

/* the operand class of a float compare (Ks or Kd by its name), or -1 */
static int
fcmpcls(int op)
{
	switch (op) {
	case Oceqs: case Oclts: case Ocles: case Ocgts: case Ocges: return Ks;
	case Oceqd: case Ocltd: case Ocled: case Ocgtd: case Ocged: return Kd;
	}
	return -1;
}

/* a comparison whose only use is the sel0 that ifopt makes of a select */
static int
issel0(Fn *fn, int tn)
{
	Use *u;

	if (fn->tmp[tn].nuse != 1)
		return 0;
	u = fn->tmp[tn].use;
	return u->type == UIns && u->u.ins->op == Osel0;
}

/* the signed compare of the exit test, its kind (see Loop.cmpop), or -1 */
static int
cmpkind(int op)
{
	switch (op) {
	case Ocsltw: case Ocsltl: return 0;
	case Ocslew: case Ocslel: return 1;
	case Ocsgtw: case Ocsgtl: return 2;
	case Ocsgew: case Ocsgel: return 3;
	}
	return -1;
}

/* the step of the latch value i2: +1 for i + 1, -1 for i - 1 or i - 1 as sub; 0 for anything else */
static int64_t
latchstep(Fn *fn, Loop *lp, Ins *ins)
{
	int64_t c;

	if (ins->cls != lp->ivcls || !req(ins->arg[0], lp->i) || !conint(fn, ins->arg[1], &c))
		return 0;
	if (ins->op == Oadd)
		return c;
	if (ins->op == Osub)
		return -c;
	return 0;
}

/* find and check the loop headed by H; fills lp, true when H heads a loop to vectorize */
static int
findloop(Fn *fn, Blk *H, Loop *lp)
{
	Blk *B, *L, *P, *b;
	Phi *ph, *p;
	Ins *cmp, *ins;
	struct Stats st;
	uint k, t, nt, nb, j;
	int ia, found, nr, cmpop;
	Use *u;

	memset(&st, 0, sizeof st);
	if (H->jmp.type != Jjnz || H->s1 == H || H->s2 == H || H->npred != 2 || H->nins != 1 || !H->phi)
		return rej(__LINE__);
	cmp = &H->ins[0];
	cmpop = cmpkind(cmp->op);
	if (cmpop < 0 || (cmp->cls != Kw && cmp->cls != Kl) || !req(H->jmp.arg, cmp->to))
		return rej(__LINE__);
	if (fn->tmp[cmp->to.val].nuse != 1)
		return rej(__LINE__);
	if (rtype(cmp->arg[1]) == RCon && fn->con[cmp->arg[1].val].type != CBits)
		return rej(__LINE__);
	if (rtype(cmp->arg[1]) != RTmp && rtype(cmp->arg[1]) != RCon)
		return rej(__LINE__);
	/* the induction phi: the one the comparison tests */
	ph = 0;
	for (p=H->phi; p; p=p->link)
		if (req(p->to, cmp->arg[0]))
			ph = p;
	if (!ph || ph->narg != 2 || ph->cls != cmp->cls)
		return rej(__LINE__);

	/* the body: a chain of blocks with one predecessor each, up to the latch (which jumps back to H); an if
	 * may sit in it, as a diamond or a triangle, that meets in a join block with phis */
	lp->nbody = 0;
	lp->hascond = 0;
	lp->lastsel = 0;
	lp->nmask = 0;
	lp->cb = lp->tb = lp->fb = lp->join = 0;
	B = H->s1;
	for (;;) {
		if (!B || B == H || lp->nbody >= NBODY-4)
			return rej(__LINE__);
		if (B->jmp.type == Jjnz) {
			Blk *Tb = B->s1, *Fb = B->s2, *J;
			if (lp->hascond || B->npred != 1 || B->phi || Tb->npred != 1 || Tb->phi || Tb->jmp.type != Jjmp)
				return rej(__LINE__);
			if (Fb != Tb && Fb->jmp.type == Jjmp && Fb->npred == 1 && !Fb->phi && Fb->s1 == Tb->s1) {
				J = Tb->s1;	/* a diamond */
				lp->fb = Fb;
			} else if (Fb == Tb->s1) {
				J = Fb;		/* a triangle: the false path is the condition's own edge */
				lp->fb = 0;
			} else
				return rej(__LINE__);
			if (J == H || J->npred != 2 || rtype(B->jmp.arg) != RTmp)
				return rej(__LINE__);
			lp->hascond = 1;
			lp->cb = B;
			lp->tb = Tb;
			lp->join = J;
			lp->cond = B->jmp.arg;
			lp->guard[lp->nbody] = 0;
			lp->body[lp->nbody++] = B;
			lp->guard[lp->nbody] = 1;
			lp->body[lp->nbody++] = Tb;
			if (lp->fb) {
				lp->guard[lp->nbody] = 2;
				lp->body[lp->nbody++] = lp->fb;
			}
			B = J;
			continue;
		}
		if (B->jmp.type != Jjmp)
			return rej(__LINE__);
		if (B == lp->join ? B->npred != 2 : (B->npred != 1 || B->phi))
			return rej(__LINE__);
		lp->guard[lp->nbody] = 0;
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
	lp->ivcls = cmp->cls;
	lp->cmpop = cmpop;
	lp->ivstep = cmpop <= 1 ? 1 : -1;
	lp->lat = ph->arg[1-ia];
	lp->n = cmp->arg[1];
	lp->vf = 0;
	lp->nbref = 0;
	lp->nacc = 0;
	nt = fn->ntmp;
	nb = fn->nblk;
	lp->nt = nt;
	lp->inl = emalloc(nt);
	lp->dch = emalloc(nt);
	lp->idx = emalloc(nt);
	lp->adr = emalloc(nt);
	lp->strd = emalloc(nt);
	lp->anystrd = 0;
	lp->idata = 0;
	lp->lane = emalloc(nt);
	lp->step = emalloc(nt * sizeof lp->step[0]);
	lp->ioff = emalloc(nt * sizeof lp->ioff[0]);
	lp->base = emalloc(nt * sizeof lp->base[0]);
	lp->inb = emalloc(nb);
	lp->map = emalloc(nt * sizeof lp->map[0]);
	lp->redof = emalloc(nt * sizeof lp->redof[0]);

	/* the reductions: the other phis of H, each with an entry from P and one from L */
	nr = 0;
	for (p=H->phi; p; p=p->link) {
		if (p == ph)
			continue;
		if (p->narg != 2 || nr == NRED || !(p->cls == Kw || (vecassoc && (p->cls == Ks || p->cls == Kd))))
			return rej(__LINE__);
		lp->redcls[nr] = p->cls;
		if (!((p->blk[0] == P && p->blk[1] == L) || (p->blk[0] == L && p->blk[1] == P)))
			return rej(__LINE__);
		lp->redph[nr] = p;
		lp->redinit[nr] = p->blk[0] == P ? p->arg[0] : p->arg[1];
		lp->redlat[nr] = p->blk[0] == L ? p->arg[0] : p->arg[1];
		if (rtype(lp->redlat[nr]) != RTmp)
			return rej(__LINE__);
		nr++;
	}
	lp->nred = nr;

	for (k=0; k<lp->nbody; k++)
		lp->inb[lp->body[k]->id] = 1;
	lp->inb[H->id] = 1;

	lp->inl[cmp->to.val] = 1;
	lp->idx[lp->i.val] = 1;
	lp->step[lp->i.val] = 1;
	lp->ioff[lp->i.val] = 0;
	if (lp->ivcls == Kw)
		lp->lane[lp->i.val] = LW;	/* usable as a w lane (vector lanes iv + j) */
	/* the mode: an index stepping by one, or a 64-bit pointer stepping by 4 or 8 bytes (its accesses are i itself) */
	{
		int64_t lc = 0;
		for (k=0; k<lp->nbody && !lc; k++)
			for (ins=lp->body[k]->ins; ins<&lp->body[k]->ins[lp->body[k]->nins]; ins++)
				if (req(ins->to, lp->lat)) {
					lc = latchstep(fn, lp, ins);
					break;
				}
		if (lc == 1 || lc == -1) {
			lp->ptr = 0;
			lp->unit = 1;
		} else if (lp->ivcls == Kl && (lc == 4 || lc == -4 || lc == 8 || lc == -8)) {
			lp->ptr = 1;
			lp->unit = lc < 0 ? -lc : lc;
		} else
			return rej(__LINE__);
		if ((lc > 0) != (lp->ivstep > 0))
			return rej(__LINE__);	/* the latch must go the way the exit test runs */
		lp->step[lp->i.val] = lp->ptr ? lc : 1;
		lp->adr[lp->i.val] = lp->ptr;
	}
	for (p=H->phi; p; p=p->link)
		lp->inl[p->to.val] = 1;
	for (k=0; k<lp->nbody; k++) {
		b = lp->body[k];
		for (p=b->phi; p; p=p->link)
			lp->inl[p->to.val] = 1;
		for (ins=b->ins; ins<&b->ins[b->nins]; ins++)
			if (rtype(ins->to) == RTmp)
				lp->inl[ins->to.val] = 1;
	}
	/* a reduction's accumulator, and its latch value */
	for (k=0; k<(uint)nr; k++) {
		lp->redof[lp->redph[k]->to.val] = k+1;
		lp->redof[lp->redlat[k].val] = -(int)(k+1);
		lp->redseen[k] = 0;
	}
	/* the bound is loop-invariant: the vector header tests it */
	if (rtype(lp->n) == RTmp && (lp->inl[lp->n.val] || fn->tmp[lp->n.val].cls != lp->ivcls))
		return rej(__LINE__);

	/* the body */
	found = 0;
	for (k=0; k<lp->nbody; k++) {
		b = lp->body[k];
		if (b == lp->join && !classifyjoin(fn, lp, &st))
			return rej(__LINE__);
		for (ins=b->ins; ins<&b->ins[b->nins]; ins++) {
			if (ins->op == Onop || ins->op == Odbgloc)
				continue;
			if (req(ins->to, lp->lat)) {
				if (latchstep(fn, lp, ins) != lp->ivstep * lp->unit)
					return rej(__LINE__);
				found = 1;
				continue;
			}
			if (!classify(fn, lp, ins, &st)) {
				if (debug['V'])
					fprintf(stderr, "vec: ins %s (cls %d) to %d refused\n", optab[ins->op].name, ins->cls, rtype(ins->to)==RTmp ? (int)ins->to.val : -1);
				return rej(__LINE__);
			}
		}
	}
	if (!found || st.nvec == 0 || st.nld + st.nst == 0)
		return rej(__LINE__);
	for (k=0; k<(uint)nr; k++)
		if (!lp->redseen[k])
			return rej(__LINE__);
	if (lp->hascond) {
		if (lp->lane[lp->cond.val] != LM || lp->anystrd)
			return rej(__LINE__);	/* the condition is a comparison in the body; strided accesses are not conditional */
		for (k=0; k<lp->nbody; k++)
			if (lp->guard[k])
				for (ins=lp->body[k]->ins; ins<&lp->body[k]->ins[lp->body[k]->nins]; ins++)
					if (rtype(ins->to) == RTmp && lp->redof[ins->to.val] < 0)
						return rej(__LINE__);	/* a reduction in a conditional path */
	}
	if (st.hasd && st.hasws && !st.conv)
		return rej(__LINE__);	/* mixed widths only through the conversions */
	lp->vf = st.hasd ? 2 : 4;
	lp->nst = st.nst;
	for (t=Tmp0; t<nt; t++)
		if (lp->lane[t] == LM)
			lp->nmask++;
	/* the cost model: the scalar iteration (per lane) against the vector one, with the vector loop's
	 * overhead and the runtime overlap tests; and no vector loop for a constant bound under two vectors */
	{
		int64_t sc = 2, vc = 4, cst;

		for (k=0; k<lp->nbody; k++) {
			b = lp->body[k];
			for (ins=b->ins; ins<&b->ins[b->nins]; ins++) {
				if (ins->op == Onop || ins->op == Odbgloc || req(ins->to, lp->lat))
					continue;
				sc++;
				if (ins->op == Omul && ins->cls == Kw)
					vc += 7;	/* the SSE2 sequence (mulw) */
				else if (ins->op == Osel1)
					vc += 3;
				else if ((isload(ins->op) || ins->op == Oloadsw) && isadr(lp, ins->arg[0]) && lp->strd[ins->arg[0].val])
					vc += 4;	/* two loads and a shuffle */
				else if (isstore(ins->op) && isadr(lp, ins->arg[1]) && lp->strd[ins->arg[1].val])
					vc += 8;	/* the merge with the odd lanes */
				else
					vc += 1;
			}
		}
		if (lp->nst > 0 && lp->nbref > 1)
			vc += 4 * lp->nbref * (lp->nbref - 1);	/* the runtime overlap tests */
		if (lp->hascond)
			vc += 2;
		if (rtype(lp->n) == RCon && conint(fn, lp->n, &cst) && cst < 2 * lp->vf)
			return rej(__LINE__);	/* too few iterations for a vector loop */
		if (vc > sc * lp->vf) {
			if (debug['V'])
				fprintf(stderr, "vec: %s: not profitable: vector cost %lld, scalar %lld x %d lanes\n",
					fn->name, (long long)vc, (long long)sc, lp->vf);
			return rej(__LINE__);
		}
	}
	if (lp->idata && lp->vf != 4)
		return rej(__LINE__);	/* the induction's lanes are four w lanes */
	if (lp->anystrd && (lp->ivstep < 0 || lp->ptr))
		return rej(__LINE__);	/* the stride-2 windows are taken counting up, and as indices only */
	if (lp->ptr && lp->idata)
		return rej(__LINE__);	/* the induction as a value is an index's */
	/* a store and another access of its base: their distance in elements is 0, or at least VF */
	for (k=0; k<(uint)lp->nacc; k++) {
		int64_t d, sz;
		int q;

		if (!lp->acc_st[k])
			continue;
		sz = lp->bsz[lp->acc_base[k]];
		for (q=0; q<lp->nacc; q++) {
			if (q == (int)k || lp->acc_base[q] != lp->acc_base[k])
				continue;
			d = lp->acc_off[k] - lp->acc_off[q];
			if (d == 0)
				continue;
			if (d % sz != 0)
				return rej(__LINE__);
			if ((d < 0 ? -d : d) / sz < lp->vf)
				return rej(__LINE__);
		}
	}

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

/* a phi of two entries */
static Phi *
mkphi(Ref to, int cls, Blk *b0, Ref a0, Blk *b1, Ref a1)
{
	Phi *p;

	p = alloc(sizeof *p);
	p->to = to;
	p->cls = cls;
	p->narg = 2;
	p->visit = 0;
	p->link = 0;
	p->arg = vnew(2, sizeof p->arg[0], PFn);
	p->blk = vnew(2, sizeof p->blk[0], PFn);
	p->arg[0] = a0;
	p->blk[0] = b0;
	p->arg[1] = a1;
	p->blk[1] = b1;
	return p;
}

/* the low 32 bits of the lanes' products, with SSE2 only: pmuludq on the even lanes and on the odd lanes (shifted
 * down), then the low halves packed and interleaved */
static Ref
mulw(Fn *fn, Ref a, Ref b)
{
	Ref t1, as, bs, t2, x, y, r;

	t1 = newvec(fn);
	push(Ovecmuludq, Kd, t1, a, b);
	as = newvec(fn);
	push(Ovecsrlq32, Kd, as, a, R);
	bs = newvec(fn);
	push(Ovecsrlq32, Kd, bs, b, R);
	t2 = newvec(fn);
	push(Ovecmuludq, Kd, t2, as, bs);
	x = newvec(fn);
	push(Ovecpack02, Kd, x, t1, R);
	y = newvec(fn);
	push(Ovecpack02, Kd, y, t2, R);
	r = newvec(fn);
	push(Ovecunpckl32, Kd, r, x, y);
	return r;
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

	if (req(r, lp->i))
		return lp->vi;	/* the induction as a value: its vector */
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

/* a long of an index or the bound: a constant (of class c), a long temporary, or a w one, extended */
static Ref
asl(Fn *fn, Ref r, int c)
{
	Ref t;
	int64_t v;

	if (conint(fn, r, &v)) {
		t = newtmp("vec", Kl, fn);
		push(Ocopy, Kl, t, getcon(c == Kw ? (int32_t)v : v, fn), R);
		return t;
	}
	if (fn->tmp[r.val].cls == Kl)
		return r;
	t = newtmp("vec", Kl, fn);
	push(Oextsw, Kl, t, r, R);
	return t;
}

/* the bytes [lo, hi) an array base is accessed in by the loop, a superset: from init to the bound */
static void
span(Fn *fn, Loop *lp, Ref base, int size, Ref il, Ref nl, Ref *lo, Ref *hi)
{
	Ref x, y, o1, o2, t;

	if (lp->ivstep > 0) {
		x = il;		/* init, then up to n */
		y = nl;
	} else {
		x = nl;		/* n, then down to init */
		y = il;
	}
	o1 = newtmp("vec", Kl, fn);
	push(Omul, Kl, o1, x, getcon(size, fn));
	*lo = newtmp("vec", Kl, fn);
	push(Oadd, Kl, *lo, base, o1);
	if (lp->ivstep > 0) {
		t = y;
	} else {
		t = newtmp("vec", Kl, fn);
		push(Oadd, Kl, t, y, getcon(1, fn));
	}
	o2 = newtmp("vec", Kl, fn);
	push(Omul, Kl, o2, t, getcon(size, fn));
	*hi = newtmp("vec", Kl, fn);
	push(Oadd, Kl, *hi, base, o2);
}

/* the overlap test of the stored bases with the others: their spans are disjoint */
static Ref
overlaps(Fn *fn, Loop *lp, Ref init)
{
	Ref nl, il, s, x, los, his, lox, hix, c1, c2, o, ov, sum;
	int a, b2;

	nl = asl(fn, lp->n, lp->ivcls);
	il = asl(fn, init, lp->ivcls);
	ov = R;
	for (a=0; a<lp->nbref; a++) {
		if (!lp->bst[a])
			continue;
		for (b2=0; b2<lp->nbref; b2++) {
			if (b2 == a)
				continue;
			s = lp->bref[a];
			x = lp->bref[b2];
			span(fn, lp, s, lp->bsz[a], il, nl, &los, &his);
			span(fn, lp, x, lp->bsz[b2], il, nl, &lox, &hix);
			/* [los, his) and [lox, hix) overlap when los < hix and lox < his */
			c1 = newtmp("vec", Kw, fn);
			push(Ocsltl, Kl, c1, los, hix);
			c2 = newtmp("vec", Kw, fn);
			push(Ocsltl, Kl, c2, lox, his);
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

/* the mask of a lane comparison of two vectors (all ones where it holds) */
static Ref
vcmp(Fn *fn, Loop *lp, int kind, int op, Ref a, Ref b)
{
	Ref t, r;

	if (kind == LS || kind == LD) {
		/* ordered float compares: a > b is b < a, a >= b is b <= a */
		int d = kind == LD;
		int lop = op == CEq ? (d ? Ovecfeqd : Ovecfeqs) : 0;
		Ref x = a, y = b;
		if (op == CGt || op == CGe) {
			x = b;
			y = a;
		}
		if (op == CEq)
			lop = d ? Ovecfeqd : Ovecfeqs;
		else if (op == CLt || op == CGt)
			lop = d ? Ovecfltd : Ovecflts;
		else
			lop = d ? Ovecfled : Ovecfles;
		t = newvec(fn);
		push(lop, Kd, t, x, y);
		return t;
	}
	switch (op) {
	case CEq:
		t = newvec(fn);
		push(Ovecceqw, Kd, t, a, b);
		return t;
	case CNe:
		t = newvec(fn);
		push(Ovecceqw, Kd, t, a, b);
		r = newvec(fn);
		push(Ovecxorw, Kd, r, t, lp->vones);
		return r;
	case CGt:
		t = newvec(fn);
		push(Ovecgtw, Kd, t, a, b);
		return t;
	case CLt:
		t = newvec(fn);
		push(Ovecgtw, Kd, t, b, a);
		return t;
	case CGe:	/* a >= b: not b > a */
		t = newvec(fn);
		push(Ovecgtw, Kd, t, b, a);
		r = newvec(fn);
		push(Ovecxorw, Kd, r, t, lp->vones);
		return r;
	default:	/* CLe: not a > b */
		t = newvec(fn);
		push(Ovecgtw, Kd, t, a, b);
		r = newvec(fn);
		push(Ovecxorw, Kd, r, t, lp->vones);
		return r;
	}
}

/* m ? a : b, lane by lane: (a & m) | (~m & b) */
static Ref
vsel(Fn *fn, Ref m, Ref a, Ref b)
{
	Ref t1, t2, r;

	t1 = newvec(fn);
	push(Ovecandw, Kd, t1, a, m);
	t2 = newvec(fn);
	push(Ovecandn, Kd, t2, m, b);
	r = newvec(fn);
	push(Ovecorw, Kd, r, t1, t2);
	return r;
}

/* the 16-byte buffer of the induction lanes: an alloc16 at the start of the entry block, made once per function */
static Ref vbuf;
static int vbufset;

static Ref
getbuf(Fn *fn)
{
	Blk *s;
	Ins a;

	if (vbufset)
		return vbuf;
	s = fn->start;
	vbuf = newtmp("vec", Kl, fn);
	a = (Ins){.op = Oalloc16, .cls = Kl, .to = vbuf, .arg = {getcon(16, fn), R}};
	vgrow(&s->ins, s->nins+1);
	memmove(s->ins+1, s->ins, s->nins * sizeof s->ins[0]);
	s->ins[0] = a;
	s->nins++;
	vbufset = 1;
	return vbuf;
}

static void
build(Fn *fn, Loop *lp)
{
	Blk *Hv, *Bv, *Chk, *Rd, *P, *H, *b, *X;
	Phi *ph, *p, *hp, *vp, *sp;
	Ref iv, ivn, tt, cc, vb, init, t, t2, e, c, a0, a1, ov, val;
	int64_t delta;
	Ins *ins;
	uint k, j;
	int a, kind, check, nred, rk;

	H = lp->H;
	P = lp->P;
	ph = lp->ph;
	nred = lp->nred;
	/* a test is needed when the loop stores and touches some other base too */
	check = lp->nst > 0 && lp->nbref > 1;

	Hv = newblk();
	Bv = newblk();
	Chk = check ? newblk() : 0;
	Rd = nred ? newblk() : 0;
	iv = newtmp("vec", lp->ivcls, fn);
	ivn = newtmp("vec", lp->ivcls, fn);
	tt = newtmp("vec", lp->ivcls, fn);
	cc = newtmp("vec", Kw, fn);
	/* the lowest index of the vector: iv when counting up, iv - (VF-1) when counting down */
	vb = lp->ivstep > 0 ? iv : newtmp("vec", lp->ivcls, fn);
	init = ph->arg[lp->ia];

	/* the identity vector of each reduction, set up at P: 0, or all ones for and */
	nib = 0;
	if (lp->anystrd) {
		/* the stride-2 stores merge with the old lanes: zero, and the odd lanes' mask (all ones there) */
		Ref zc, oc, zv, ov1;
		zc = newtmp("vec", Kw, fn);
		push(Ocopy, Kw, zc, getcon(0, fn), R);
		zv = newvec(fn);
		push(Ovecbcastw, Kd, zv, zc, R);
		oc = newtmp("vec", Kw, fn);
		push(Ocopy, Kw, oc, getcon(-1, fn), R);
		ov1 = newvec(fn);
		push(Ovecbcastw, Kd, ov1, oc, R);
		lp->strz = zv;
		lp->strodd = newvec(fn);
		/* the odd dwords (or doubles) of zero and ones: [0, -1, 0, -1] or [0, -1] */
		push(lp->vf == 2 ? Ovecunpckhpd : Ovecunpckh32, Kd, lp->strodd, zv, ov1);
	}
	for (k=0; k<(uint)nred; k++) {
		/* the identity: all ones for and; zero for add, or and xor (and +0.0 for a float add) */
		c = newtmp("vec", Kw, fn);
		push(Ocopy, Kw, c, getcon(lp->redop[k] == Oand ? -1 : 0, fn), R);
		lp->redvi[k] = newvec(fn);
		push(Ovecbcastw, Kd, lp->redvi[k], c, R);
		lp->redva[k] = newvec(fn);
		lp->redhs[k] = newtmp("vec", lp->redcls[k], fn);
		lp->redvn[k] = R;
		lp->redrd[k] = R;
	}
	if (lp->hascond || lp->nmask) {
		Ref oc = newtmp("vec", Kw, fn);
		push(Ocopy, Kw, oc, getcon(-1, fn), R);
		lp->vones = newvec(fn);
		push(Ovecbcastw, Kd, lp->vones, oc, R);
	}
	if (lp->idata) {
		/* vi = the lanes iv + j at P (the first window's), with a 16-byte buffer in the entry block */
		Ref buf = getbuf(fn), sb, lj, ad, bc;
		int jj;
		sb = newtmp("vec", Kw, fn);
		push(Ocopy, Kw, sb, init, R);
		if (lp->ivstep < 0) {
			lj = newtmp("vec", Kw, fn);
			push(Osub, Kw, lj, sb, getcon(lp->vf - 1, fn));
			sb = lj;
		}
		for (jj=0; jj<4; jj++) {
			lj = newtmp("vec", Kw, fn);
			push(Oadd, Kw, lj, sb, getcon(jj, fn));
			ad = newtmp("vec", Kl, fn);
			push(Oadd, Kl, ad, buf, getcon(4*jj, fn));
			push(Ostorew, Kw, R, lj, ad);
		}
		lp->viota = newvec(fn);
		push(Ovecload, Kd, lp->viota, buf, R);
		bc = newtmp("vec", Kw, fn);
		push(Ocopy, Kw, bc, getcon(lp->vf, fn), R);
		lp->vstep = newvec(fn);
		push(Ovecbcastw, Kd, lp->vstep, bc, R);
		lp->vi = newvec(fn);
		lp->vin = newvec(fn);
	}
	if (nib) {
		vgrow(&P->ins, P->nins + nib);
		memmove(P->ins + P->nins, ib, nib * sizeof ib[0]);
		P->nins += nib;
	}

	/* block order: P, [Chk,] Hv, Bv, [Rd], then the rest */
	Bv->link = P->link;
	if (nred) {
		Rd->link = Bv->link;
		Bv->link = Rd;
	}
	Hv->link = Bv;
	if (check) {
		Chk->link = Hv;
		P->link = Chk;
	} else
		P->link = Hv;

	/* the body, on the lanes: Bv */
	memset(lp->map, 0, lp->nt * sizeof lp->map[0]);
	lp->map[lp->i.val] = vb;
	for (k=0; k<(uint)nred; k++)
		lp->map[lp->redph[k]->to.val] = lp->redva[k];
	nib = 0;
	if (!req(vb, iv))
		push(Osub, lp->ivcls, vb, iv, getcon((lp->vf - 1) * lp->unit, fn));
	for (k=0; k<lp->nbody; k++) {
		b = lp->body[k];
		if (b == lp->join) {
			/* each join phi selects its true path's lanes where the condition holds */
			for (p=b->phi; p; p=p->link) {
				int kd = p->cls == Kw ? LW : p->cls == Ks ? LS : LD;
				Ref argt = R, argf = R, vt, vf;
				int aa;

				for (aa=0; aa<2; aa++) {
					if (p->blk[aa] == lp->tb)
						argt = p->arg[aa];
					else
						argf = p->arg[aa];
				}
				vt = laneop(fn, lp, argt, p->cls, kd);
				vf = laneop(fn, lp, argf, p->cls, kd);
				lp->map[p->to.val] = vsel(fn, lp->cmask, vt, vf);
			}
		}
		for (ins=b->ins; ins<&b->ins[b->nins]; ins++) {
			if (ins->op == Onop || ins->op == Odbgloc)
				continue;
			if (req(ins->to, lp->lat))
				continue;
			if (rtype(ins->to) == RTmp && lp->lane[ins->to.val] == LM) {
				/* a condition: the lanes' comparison, its mask */
				int ck = fcmpcls(ins->op) >= 0 ? fcmpcls(ins->op) : Kw;
				Ref a0c = laneop(fn, lp, ins->arg[0], ck, ck == Kw ? LW : ck == Ks ? LS : LD);
				Ref a1c = laneop(fn, lp, ins->arg[1], ck, ck == Kw ? LW : ck == Ks ? LS : LD);
				lp->cmask = vcmp(fn, lp, ck == Kw ? LW : ck == Ks ? LS : LD,
					ck == Kw ? lanecmp(ins->op) : flanecmp(ins->op), a0c, a1c);
				lp->map[ins->to.val] = lp->cmask;
				continue;
			}
			if (ins->op == Osel0) {
				lp->selm = lp->map[ins->arg[0].val];
				continue;
			}
			if (ins->op == Osel1) {
				Ref a0s = laneop(fn, lp, ins->arg[0], Kw, LW);
				Ref a1s = laneop(fn, lp, ins->arg[1], Kw, LW);
				lp->map[ins->to.val] = vsel(fn, lp->selm, a0s, a1s);
				continue;
			}
			/* a reduction's latch: the vector accumulator, folded with its lane */
			if (rtype(ins->to) == RTmp && lp->redof[ins->to.val] < 0) {
				rk = -lp->redof[ins->to.val] - 1;
				Ref other = req(ins->arg[0], lp->redph[rk]->to) ? ins->arg[1] : ins->arg[0];
				kind = lp->redcls[rk] == Kw ? LW : lp->redcls[rk] == Ks ? LS : LD;
				a1 = laneop(fn, lp, other, lp->redcls[rk], kind);
				t = newvec(fn);
				push(vecop(ins->op, kind), Kd, t, lp->redva[rk], a1);
				lp->redvn[rk] = t;
				continue;
			}
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
				if (lp->strd[ins->arg[0].val]) {
					/* two vectors, and the even lanes of them: the stride-2 window */
					Ref a0 = cloneref(lp, ins->arg[0]);
					Ref a16 = newtmp("vec", Kl, fn);
					Ref v0 = newvec(fn), v1 = newvec(fn);
					push(Oadd, Kl, a16, a0, getcon(16, fn));
					push(Ovecload, Kd, v0, a0, R);
					push(Ovecload, Kd, v1, a16, R);
					t = newvec(fn);
					push(lp->vf == 2 ? Ovecunpcklpd : Ovecshufe, Kd, t, v0, v1);
					lp->map[ins->to.val] = t;
					continue;
				}
				t = newvec(fn);
				/* two w lanes (the loop has doubles): the low 64 bits */
				push(ins->op == Oloadsw && lp->vf == 2 ? Ovecload8 : Ovecload, Kd, t, cloneref(lp, ins->arg[0]), R);
				lp->map[ins->to.val] = t;
				continue;
			case Ostorew:
			case Ostores:
			case Ostored:
				if (lp->guard[k]) {
					/* a store on one path: the lanes of the other path keep the old values */
					Ref x = req(ins->arg[0], lp->i) ? lp->vi : lp->map[ins->arg[0].val];
					Ref old = newvec(fn), nv;

					int w8 = ins->op == Ostorew && lp->vf == 2;
					push(w8 ? Ovecload8 : Ovecload, Kd, old, cloneref(lp, ins->arg[1]), R);
					nv = lp->guard[k] == 1 ? vsel(fn, lp->cmask, x, old) : vsel(fn, lp->cmask, old, x);
					push(w8 ? Ovecstore8 : Ovecstore, Kw, R, nv, cloneref(lp, ins->arg[1]));
					continue;
				}
				if (lp->strd[ins->arg[1].val]) {
					/* the value's lanes go to the even positions of two vectors; the odd ones keep their values */
					Ref a0 = cloneref(lp, ins->arg[1]);
					Ref a16 = newtmp("vec", Kl, fn);
					Ref x = req(ins->arg[0], lp->i) ? lp->vi : lp->map[ins->arg[0].val];
					Ref xs = newvec(fn), xh = newvec(fn), o0 = newvec(fn), o1 = newvec(fn);
					Ref m0 = newvec(fn), m1 = newvec(fn), out0 = newvec(fn), out1 = newvec(fn);
					int kd = lp->vf == 2;
					push(Oadd, Kl, a16, a0, getcon(16, fn));
					push(kd ? Ovecunpcklpd : Ovecunpckl32, Kd, xs, x, lp->strz);
					push(kd ? Ovecunpckhpd : Ovecunpckh32, Kd, xh, x, lp->strz);
					push(Ovecload, Kd, o0, a0, R);
					push(Ovecload, Kd, o1, a16, R);
					push(Ovecandw, Kd, m0, o0, lp->strodd);
					push(Ovecandw, Kd, m1, o1, lp->strodd);
					push(Ovecorw, Kd, out0, xs, m0);
					push(Ovecorw, Kd, out1, xh, m1);
					push(Ovecstore, Kw, R, out0, a0);
					push(Ovecstore, Kw, R, out1, a16);
					continue;
				}
				push(ins->op == Ostorew && lp->vf == 2 ? Ovecstore8 : Ovecstore, Kw, R,
					req(ins->arg[0], lp->i) ? lp->vi : lp->map[ins->arg[0].val], cloneref(lp, ins->arg[1]));
				continue;
			default:
				break;
			}
			/* the conversions of mixed lanes */
			if (ins->op == Oswtof || ins->op == Odtosi) {
				int from = ins->op == Oswtof ? Kw : Kd;
				a0 = laneop(fn, lp, ins->arg[0], from, from == Kw ? LW : LD);
				t = newvec(fn);
				push(ins->op == Oswtof ? Ovecw2d : Ovecd2w, Kd, t, a0, R);
				lp->map[ins->to.val] = t;
				continue;
			}
			kind = ins->cls == Kw ? LW : ins->cls == Ks ? LS : LD;
			a0 = laneop(fn, lp, ins->arg[0], ins->cls, kind);
			a1 = laneop(fn, lp, ins->arg[1], ins->cls, kind);
			if (vecop(ins->op, kind) == -2 && strcmp(T.name, "arm64") == 0) {
				t = newvec(fn);
				push(Ovecmulw, Kd, t, a0, a1);	/* NEON has the 32-bit lane multiply */
			} else if (vecop(ins->op, kind) == -2)
				t = mulw(fn, a0, a1);
			else {
				t = newvec(fn);
				push(vecop(ins->op, kind), Kd, t, a0, a1);
			}
			lp->map[ins->to.val] = t;
		}
	}
	/* ivn = iv + VF; vin = vi + VF (the lanes); jmp Hv */
	if (lp->idata)
		push(lp->ivstep > 0 ? Ovecaddw : Ovecsubw, Kd, lp->vin, lp->vi, lp->vstep);
	push(lp->ivstep > 0 ? Oadd : Osub, lp->ivcls, ivn, iv, getcon(lp->vf * lp->unit, fn));
	idup(Bv, ib, nib);
	Bv->jmp.type = Jjmp;
	Bv->s1 = Hv;
	Bv->s2 = 0;

	/* Rd: the lanes of each accumulator folded into its scalar one, then to H */
	if (nred) {
		nib = 0;
		for (k=0; k<(uint)nred; k++) {
			int rc = lp->redcls[k];
			t = lp->redhs[k];
			for (j=0; j<(uint)lp->vf; j++) {
				e = newtmp("vec", rc, fn);
				push(rc == Kw ? Ovecext0w + j : rc == Ks ? Ovecext0s + j : Ovecext0d + j, rc, e, lp->redva[k], R);
				t2 = newtmp("vec", rc, fn);
				push(lp->redop[k], rc, t2, t, e);
				t = t2;
			}
			lp->redrd[k] = t;
		}
		idup(Rd, ib, nib);
		Rd->jmp.type = Jjmp;
		Rd->s1 = H;
		Rd->s2 = 0;
	}

	/* the exit of the vector loop: Rd when there are reductions, else H */
	X = nred ? Rd : Hv;

	/* H's phis: the entry from P now comes from Chk (or X); the vector loop's exit from X */
	for (p=H->phi; p; p=p->link) {
		if (p == ph)
			val = iv;
		else {
			rk = 0;
			while (lp->redph[rk] != p)
				rk++;
			val = lp->redrd[rk];
		}
		if (!check) {
			for (a=0; a<2; a++)
				if (p->blk[a] == P) {
					p->blk[a] = X;
					p->arg[a] = val;
				}
		} else {
			for (a=0; a<2; a++)
				if (p->blk[a] == P)
					p->blk[a] = Chk;
			vgrow(&p->arg, 3);
			vgrow(&p->blk, 3);
			p->arg[2] = val;
			p->blk[2] = X;
			p->narg = 3;
		}
	}

	/* Hv: iv = phi(P or Chk: init, Bv: ivn); the accumulators */
	hp = mkphi(iv, lp->ivcls, check ? Chk : P, init, Bv, ivn);
	Hv->phi = hp;
	if (lp->idata) {
		vp = mkphi(lp->vi, Kd, check ? Chk : P, lp->viota, Bv, lp->vin);
		hp->link = vp;
		hp = vp;
	}
	for (k=0; k<(uint)nred; k++) {
		vp = mkphi(lp->redva[k], Kd, check ? Chk : P, lp->redvi[k], Bv, lp->redvn[k]);
		sp = mkphi(lp->redhs[k], lp->redcls[k], check ? Chk : P, lp->redinit[k], Bv, lp->redhs[k]);
		hp->link = vp;
		vp->link = sp;
		hp = sp;
	}

	if (P->s1 == H)
		P->s1 = check ? Chk : Hv;
	if (P->s2 == H)
		P->s2 = check ? Chk : Hv;

	if (check) {
		nib = 0;
		ov = overlaps(fn, lp, init);
		idup(Chk, ib, nib);
		Chk->jmp.type = Jjnz;
		Chk->jmp.arg = ov;
		Chk->s1 = H;
		Chk->s2 = Hv;
	}

	/* Hv: tt = iv + VF; cc = tt <= n; jnz cc, Bv, X */
	nib = 0;
	/* the vector's lanes are in the loop when they all pass the exit test */
	delta = lp->cmpop == 0 ? lp->vf : lp->cmpop == 1 ? lp->vf - 1
		: lp->cmpop == 2 ? -lp->vf : -(lp->vf - 1);
	delta *= lp->unit;
	if (lp->anystrd)
		delta += 1;	/* the last window's second half ends at element 2*(vb+VF)-1 */
	push(Oadd, lp->ivcls, tt, iv, getcon(delta, fn));
	if (lp->cmpop <= 1)
		push(lp->ivcls == Kw ? Ocslew : Ocslel, lp->ivcls, cc, tt, lp->n);
	else
		push(lp->ivcls == Kw ? Ocsgew : Ocsgel, lp->ivcls, cc, tt, lp->n);
	idup(Hv, ib, nib);
	Hv->jmp.type = Jjnz;
	Hv->jmp.arg = cc;
	Hv->s1 = Bv;
	Hv->s2 = nred ? Rd : H;
	(void)rk;
	(void)j;
}

void
vec(Fn *fn)
{
	Blk **cand, *b;
	uint n, nc, k;
	Loop lp;

	if ((strcmp(T.name, "amd64_sysv") != 0 && strcmp(T.name, "arm64") != 0) || nofp)
		return;
	ib = vnew(0, sizeof ib[0], PHeap);
	nib = 0;
	vbufset = 0;
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
		if (debug['W'])
			printfn(fn, stderr);	/* -d W: the IR the pass sees */
		if (findloop(fn, cand[k], &lp)) {
			if (debug['V'])
				fprintf(stderr, "vec: %s: loop at @%d, %d lanes, %d bases, %d reductions\n", fn->name,
					cand[k]->id, lp.vf, lp.nbref, lp.nred);
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
		free(lp.strd);
		free(lp.lane);
		free(lp.step);
		free(lp.ioff);
		free(lp.base);
		free(lp.inb);
		free(lp.map);
		free(lp.redof);
	}
	vfree(ib);
	ib = 0;
}
