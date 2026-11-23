#include "all.h"

/* Loop-invariant loads (nfcxx). gcm pins a load in its block, so a load that reads the same memory on every iteration
   stays in the loop. This pass moves such a load to the loop's preheader. A load moves when
   - its address is a constant, a slot or a temporary defined outside the loop;
   - no instruction of the loop writes memory (stores, block copies, atomics, calls, variadic accesses, allocations), so the
     loaded value is the same on every iteration;
   - its block dominates every exit of the loop (a block that leaves the loop or returns), so the load runs whenever the
     loop runs and the move adds no access the program would not make.
   The loop has a preheader only if the header has one predecessor outside the loop, whose only successor is the header;
   otherwise the loop is left alone. Loops with no exit are left alone too. */

static int
writesmem(Ins *i)
{
	if (isstore(i->op) || isalloc(i->op))
		return 1;
	switch (i->op) {
	case Oblit0:
	case Oblit1:
	case Oxadd:
	case Ocmpxchg0:
	case Ocmpxchg1:
	case Ocall:
	case Ovaarg:
	case Ovastart:
		return 1;
	default:
		return 0;
	}
}

/* the loop with header h: the blocks of its natural loops (all back edges into h), marked in mark[] with stamp */
static uint
loopbody(Blk *h, uint *mark, uint stamp, Blk ***stack_out)
{
	Blk *b, *q, **stack;
	uint n, p, nstk;

	mark[h->id] = stamp;
	nstk = 0;
	stack = vnew(0, sizeof stack[0], PFn);
	n = 1;
	for (p=0; p<h->npred; p++) {
		b = h->pred[p];
		if (!dom(h, b))
			continue;
		if (mark[b->id] != stamp) {
			mark[b->id] = stamp;
			n++;
			vgrow(&stack, nstk+1);
			stack[nstk++] = b;
		}
	}
	while (nstk > 0) {
		b = stack[--nstk];
		for (p=0; p<b->npred; p++) {
			q = b->pred[p];
			if (mark[q->id] != stamp) {
				mark[q->id] = stamp;
				n++;
				vgrow(&stack, nstk+1);
				stack[nstk++] = q;
			}
		}
	}
	*stack_out = stack;
	return n;
}

static int
inloop(uint *mark, uint stamp, Blk *b)
{
	return mark[b->id] == stamp;
}

/* a load at block lb can move out of the loop if its address is loop-invariant and lb dominates every exit */
static int
movable(Fn *fn, uint *mark, uint stamp, Blk *lb, Ins *i, Blk **exits, uint nexits)
{
	Tmp *t;
	uint e;

	if (!isload(i->op))
		return 0;
	if (rtype(i->arg[0]) == RTmp) {
		t = &fn->tmp[i->arg[0].val];
		if (t->bid == -1u)
			return 0;
		if (inloop(mark, stamp, fn->rpo[t->bid]))
			return 0;
	}
	for (e=0; e<nexits; e++)
		if (!dom(lb, exits[e]))
			return 0;
	return 1;
}

void
licm(Fn *fn)
{
	Blk *h, *b, *ph, *lb, **stack, **exits;
	Ins *ni, *mv, *i;
	uint *mark, stamp, n, p, nexits, nmv, k, bi, m, nins;
	Blk **mvb;
	uint *mvi;
	int nph;

	mark = emalloc(fn->nblk * sizeof mark[0]);
	for (n=0; n<fn->nblk; n++)
		mark[n] = 0;
	stamp = 0;
	for (n=0; n<fn->nblk; n++) {
		h = fn->rpo[n];
		/* a header: a block with a predecessor it dominates (a back edge) */
		int back = 0;
		for (p=0; p<h->npred; p++)
			if (dom(h, h->pred[p]))
				back = 1;
		if (!back)
			continue;

		stamp = h->id + 1;
		loopbody(h, mark, stamp, &stack);

		/* the preheader: the one predecessor outside the loop, whose only successor is the header */
		ph = 0;
		nph = 0;
		for (p=0; p<h->npred; p++)
			if (!inloop(mark, stamp, h->pred[p])) {
				ph = h->pred[p];
				nph++;
			}
		if (nph != 1 || ph->jmp.type != Jjmp || ph->s1 != h || ph->s2)
			continue;

		/* the exits: blocks of the loop with a successor outside it, or that return */
		nexits = 0;
		exits = vnew(0, sizeof exits[0], PFn);
		for (bi=0; bi<fn->nblk; bi++) {
			b = fn->rpo[bi];
			if (!inloop(mark, stamp, b))
				continue;
			if (isret(b->jmp.type) || (b->s1 && !inloop(mark, stamp, b->s1)) || (b->s2 && !inloop(mark, stamp, b->s2))) {
				vgrow(&exits, nexits+1);
				exits[nexits++] = b;
			}
		}
		if (nexits == 0)
			continue;

		/* the loop writes no memory */
		int writes = 0;
		for (bi=0; bi<fn->nblk && !writes; bi++) {
			b = fn->rpo[bi];
			if (!inloop(mark, stamp, b))
				continue;
			for (i=b->ins; i<&b->ins[b->nins]; i++)
				if (writesmem(i)) {
					writes = 1;
					break;
				}
		}
		if (writes)
			continue;

		/* the loads to move: found in block order, so their indices are ascending within a block */
		nmv = 0;
		mvb = vnew(0, sizeof mvb[0], PFn);
		mvi = vnew(0, sizeof mvi[0], PFn);
		for (bi=0; bi<fn->nblk; bi++) {
			lb = fn->rpo[bi];
			if (!inloop(mark, stamp, lb))
				continue;
			for (m=0; m<lb->nins; m++) {
				i = &lb->ins[m];
				if (!movable(fn, mark, stamp, lb, i, exits, nexits))
					continue;
				vgrow(&mvb, nmv+1);
				vgrow(&mvi, nmv+1);
				mvb[nmv] = lb;
				mvi[nmv] = m;
				nmv++;
			}
		}
		if (nmv == 0)
			continue;

		/* copy the moved loads out, then rebuild the blocks they leave and the preheader */
		mv = emalloc(nmv * sizeof mv[0]);
		for (k=0; k<nmv; k++) {
			mv[k] = mvb[k]->ins[mvi[k]];
			fn->tmp[mv[k].to.val].bid = ph->id;   /* the result is now defined in the preheader */
		}
		for (k=0; k<nmv; k++) {
			lb = mvb[k];
			if (k > 0 && mvb[k-1] == lb)
				continue;
			nins = 0;
			for (m=0; m<lb->nins; m++) {
				int moved = 0;
				for (p=0; p<nmv; p++)
					if (mvb[p] == lb && mvi[p] == m)
						moved = 1;
				if (!moved)
					nins++;
			}
			ni = vnew(nins, sizeof ni[0], PFn);
			nins = 0;
			for (m=0; m<lb->nins; m++) {
				int moved = 0;
				for (p=0; p<nmv; p++)
					if (mvb[p] == lb && mvi[p] == m)
						moved = 1;
				if (!moved)
					ni[nins++] = lb->ins[m];
			}
			lb->ins = ni;
			lb->nins = nins;
		}
		ni = vnew(ph->nins + nmv, sizeof ni[0], PFn);
		for (m=0; m<ph->nins; m++)
			ni[m] = ph->ins[m];
		for (k=0; k<nmv; k++)
			ni[ph->nins + k] = mv[k];
		ph->ins = ni;
		ph->nins += nmv;

		free(mv);
	}
	free(mark);
}

/* Loop strength reduction (nfcxx). An address B + extsw(i) * c in a loop, with i the induction of the loop
   (i = phi(P: i0, L: i + s), s a constant, c a constant or 1 << k), becomes a pointer induction in the header:
   p = phi(P: B + i0 * c, L: p + c * s), and the address is p. Without it the amd64 code multiplies on every
   iteration. A chain qualifies when every use of the multiply is an add in the loop whose other operand is
   loop-invariant, the header has one entry P (jumping only to the header) and one latch L (jumping only to it). */

#define IVMAX 16

static int
ivconst(Fn *fn, Ref r, int64_t *v)
{
	if (rtype(r) != RCon || fn->con[r.val].type != CBits)
		return 0;
	*v = fn->con[r.val].bits.i;
	return 1;
}

static int
ivphiarg(Phi *p, Blk *b, Ref *r)
{
	uint a;

	for (a=0; a<p->narg; a++)
		if (p->blk[a] == b) {
			*r = p->arg[a];
			return 1;
		}
	return 0;
}

static int
invariantref(Fn *fn, uint *mark, uint stamp, Ref r)
{
	if (rtype(r) != RTmp)
		return 1;
	if (fn->tmp[r.val].bid == -1u)
		return 0;
	return !inloop(mark, stamp, fn->rpo[fn->tmp[r.val].bid]);
}

static void
appendins(Blk *b, Ins i)
{
	Ins *ni;

	ni = vnew(b->nins + 1, sizeof ni[0], PFn);
	memcpy(ni, b->ins, b->nins * sizeof ni[0]);
	ni[b->nins] = i;
	b->ins = ni;
	b->nins++;
}

static void
dropnops(Fn *fn)
{
	Blk *b;
	Ins *ni;
	uint m, n;

	for (b=fn->start; b; b=b->link) {
		n = 0;
		for (m=0; m<b->nins; m++)
			if (b->ins[m].op != Onop)
				n++;
		if (n == b->nins)
			continue;
		ni = vnew(n, sizeof ni[0], PFn);
		n = 0;
		for (m=0; m<b->nins; m++)
			if (b->ins[m].op != Onop)
				ni[n++] = b->ins[m];
		b->ins = ni;
		b->nins = n;
	}
}

/* every use of temporary from (phi args, instruction args, jump argument) becomes to */
static void
renameuses(Fn *fn, Ref from, Ref to)
{
	Blk *b;
	Phi *p;
	Ins *i;
	uint a, m;

	for (b=fn->start; b; b=b->link) {
		for (p=b->phi; p; p=p->link)
			for (a=0; a<p->narg; a++)
				if (req(p->arg[a], from))
					p->arg[a] = to;
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			for (m=0; m<2; m++)
				if (req(i->arg[m], from))
					i->arg[m] = to;
		if (req(b->jmp.arg, from))
			b->jmp.arg = to;
	}
}

/* the temporary r is read by a live instruction, phi or jump */
static int
usedanywhere(Fn *fn, Ref r)
{
	Blk *b;
	Phi *p;
	Ins *i;
	uint a, m;

	for (b=fn->start; b; b=b->link) {
		for (p=b->phi; p; p=p->link)
			for (a=0; a<p->narg; a++)
				if (req(p->arg[a], r))
					return 1;
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			if (i->op != Onop)
				for (m=0; m<2; m++)
					if (req(i->arg[m], r))
						return 1;
		if (req(b->jmp.arg, r))
			return 1;
	}
	return 0;
}

/* the instruction defining r becomes a nop (removed by dropnops) */
static void
killdef(Fn *fn, Ref r)
{
	Blk *b;
	Ins *i;

	for (b=fn->start; b; b=b->link)
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			if (i->op != Onop && req(i->to, r)) {
				i->op = Onop;
				return;
			}
}

/* one chain of the loop with header h (entry P, latch L): 1 if it was rewritten, 0 if none qualifies */
static int
strengthen(Fn *fn, Blk *h, Blk *P, Blk *L, uint *mark, uint stamp)
{
	Phi *ph, *ivp, *np;
	Blk *b;
	Ins *m, *x, *ad, *d;
	Use *use;
	Tmp *t;
	Ref init, lv, iv, mto, xto, adto[IVMAX], adbase[IVMAX], grpb[IVMAX], grpv[IVMAX], pinit, pv, pnext, t1, t2, other;
	int64_t c, s, v, kc, i0;
	uint bi, j, u, n, ng, g;
	int found, isconst;

	filluse(fn);
	ivp = 0;
	for (ph=h->phi; ph && !ivp; ph=ph->link) {
		if (ph->cls != Kw || ph->narg != 2 || rtype(ph->to) != RTmp)
			continue;
		if (!ivphiarg(ph, L, &lv) || rtype(lv) != RTmp || fn->tmp[lv.val].bid == -1u)
			continue;
		if (!inloop(mark, stamp, fn->rpo[fn->tmp[lv.val].bid]) || !fn->tmp[lv.val].def)
			continue;
		d = fn->tmp[lv.val].def;
		if (d->op == Oadd && d->cls == Kw && rtype(d->arg[0]) == RTmp && req(d->arg[0], ph->to) && ivconst(fn, d->arg[1], &s))
			ivp = ph;
	}
	if (!ivp || !ivphiarg(ivp, P, &init))
		return 0;
	iv = ivp->to;

	found = 0;
	for (bi=0; bi<fn->nblk && !found; bi++) {
		b = fn->rpo[bi];
		if (!inloop(mark, stamp, b))
			continue;
		for (j=0; j<b->nins && !found; j++) {
			m = &b->ins[j];
			if ((m->op != Omul && m->op != Oshl) || m->cls != Kl || rtype(m->arg[0]) != RTmp || !ivconst(fn, m->arg[1], &kc))
				continue;
			if (m->op == Oshl) {
				if (kc < 0 || kc > 30)
					continue;
				c = (int64_t)1 << kc;
			} else
				c = kc;
			if (c <= 0 || fn->tmp[m->arg[0].val].bid == -1u || !fn->tmp[m->arg[0].val].def)
				continue;
			if (!inloop(mark, stamp, fn->rpo[fn->tmp[m->arg[0].val].bid]))
				continue;
			x = fn->tmp[m->arg[0].val].def;
			if (x->op != Oextsw || x->cls != Kl || !req(x->arg[0], iv))
				continue;
			/* the uses of the multiply */
			t = &fn->tmp[m->to.val];
			n = t->nuse;
			if (n == 0 || n > IVMAX)
				continue;
			found = 1;
			for (u=0; u<n; u++) {
				use = &t->use[u];
				if (use->type != UIns || !inloop(mark, stamp, fn->rpo[use->bid])) {
					found = 0;
					break;
				}
				ad = use->u.ins;
				if (ad->op != Oadd || ad->cls != Kl || req(ad->arg[0], ad->arg[1]))
					found = 0;
				else if (req(ad->arg[0], m->to) && !req(ad->arg[1], m->to))
					other = ad->arg[1];
				else if (req(ad->arg[1], m->to) && !req(ad->arg[0], m->to))
					other = ad->arg[0];
				else
					found = 0;
				if (!found)
					break;
				if (!invariantref(fn, mark, stamp, other)) {
					found = 0;
					break;
				}
				adto[u] = ad->to;
				adbase[u] = other;
			}
			if (found) {
				mto = m->to;
				xto = x->to;
				break;
			}
		}
	}
	if (!found)
		return 0;
	/* the entry's pointer: the induction's start value must be a constant or a temporary */
	isconst = ivconst(fn, init, &i0);
	if (!isconst && rtype(init) != RTmp)
		return 0;
	if (isconst)
		i0 = (int64_t)(int32_t)i0;

	/* one pointer per distinct base: the entry computes B + i0 * c, the latch adds c * s. With two or more bases a
	   pointer per base costs more adds per iteration than the one index it replaces (saxpy, dot), so only one base
	   is rewritten. */
	for (u=1; u<n; u++)
		if (!req(adbase[u], adbase[0]))
			return 0;
	ng = 0;
	for (u=0; u<n; u++) {
		for (g=0; g<ng; g++)
			if (req(grpb[g], adbase[u]))
				break;
		if (g < ng)
			continue;
		grpb[ng] = adbase[u];
		pv = newtmp("ivsr", Kl, fn);
		grpv[ng] = pv;
		if (isconst) {
			v = i0 * c;
			if (v == 0)
				pinit = grpb[ng];
			else {
				pinit = newtmp("ivsr", Kl, fn);
				appendins(P, (Ins){Oadd, Kl, pinit, {grpb[ng], getcon(v, fn)}});
			}
		} else {
			t1 = newtmp("ivsr", Kl, fn);
			appendins(P, (Ins){Oextsw, Kl, t1, {init, R}});
			if (c != 1) {
				t2 = newtmp("ivsr", Kl, fn);
				appendins(P, (Ins){Omul, Kl, t2, {t1, getcon(c, fn)}});
				t1 = t2;
			}
			pinit = newtmp("ivsr", Kl, fn);
			appendins(P, (Ins){Oadd, Kl, pinit, {grpb[ng], t1}});
		}
		pnext = newtmp("ivsr", Kl, fn);
		appendins(L, (Ins){Oadd, Kl, pnext, {pv, getcon(c * s, fn)}});
		np = vnew(1, sizeof np[0], PFn);
		np->to = pv;
		np->cls = Kl;
		np->narg = 2;
		np->arg = vnew(2, sizeof np->arg[0], PFn);
		np->blk = vnew(2, sizeof np->blk[0], PFn);
		np->arg[0] = pinit;
		np->blk[0] = P;
		np->arg[1] = pnext;
		np->blk[1] = L;
		np->link = h->phi;
		h->phi = np;
		ng++;
	}
	/* each add becomes its group's pointer; the multiply and the extension go when nothing reads them */
	for (u=0; u<n; u++) {
		for (g=0; g<ng; g++)
			if (req(grpb[g], adbase[u]))
				break;
		renameuses(fn, adto[u], grpv[g]);
		killdef(fn, adto[u]);
	}
	killdef(fn, mto);
	if (!usedanywhere(fn, xto))
		killdef(fn, xto);
	dropnops(fn);
	filluse(fn);
	return 1;
}

void
ivsr(Fn *fn)
{
	Blk *h, *P, *L;
	uint *mark, stamp, n, p, k;
	int nlatch, nentry;
	Blk **stack;

	mark = emalloc(fn->nblk * sizeof mark[0]);
	for (n=0; n<fn->nblk; n++)
		mark[n] = 0;
	for (n=0; n<fn->nblk; n++) {
		h = fn->rpo[n];
		if (h->npred != 2)
			continue;
		P = L = 0;
		nlatch = nentry = 0;
		for (p=0; p<h->npred; p++) {
			if (dom(h, h->pred[p])) {
				L = h->pred[p];
				nlatch++;
			} else {
				P = h->pred[p];
				nentry++;
			}
		}
		if (nlatch != 1 || nentry != 1)
			continue;
		if (P->jmp.type != Jjmp || P->s1 != h || P->s2 || L->jmp.type != Jjmp || L->s1 != h || L->s2)
			continue;
		stamp = h->id + 1;
		loopbody(h, mark, stamp, &stack);
		for (k=0; k<IVMAX && strengthen(fn, h, P, L, mark, stamp); k++)
			;
	}
	free(mark);
}
