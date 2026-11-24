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

/* the index a = iv * c0 + q (a w add of an invariant q and iv or a constant multiple of it, in either order):
   1 with c0 and q (and the multiply's temporary in *pt, R when iv itself is added), 0 otherwise */
static int
ivterm(Fn *fn, uint *mark, uint stamp, Ins *a, Ref iv, int64_t *c0, Ref *q, Ref *pt)
{
	Ref pr, qr;
	Ins *p;
	int64_t kc;
	int e;

	if (a->op != Oadd || a->cls != Kw)
		return 0;
	for (e=0; e<2; e++) {
		pr = a->arg[e];
		qr = a->arg[1-e];
		if (rtype(qr) != RTmp || !invariantref(fn, mark, stamp, qr))
			continue;
		if (req(pr, iv)) {
			*c0 = 1;
			*q = qr;
			*pt = R;
			return 1;
		}
		if (rtype(pr) != RTmp || fn->tmp[pr.val].bid == -1u || !fn->tmp[pr.val].def)
			continue;
		if (!inloop(mark, stamp, fn->rpo[fn->tmp[pr.val].bid]))
			continue;
		p = fn->tmp[pr.val].def;
		if ((p->op != Omul && p->op != Oshl) || p->cls != Kw || !req(p->arg[0], iv) || !ivconst(fn, p->arg[1], &kc))
			continue;
		if (p->op == Oshl) {
			if (kc < 0 || kc > 30)
				continue;
			kc = (int64_t)1 << kc;
		}
		if (kc <= 0)
			continue;
		*c0 = kc;
		*q = qr;
		*pt = pr;
		return 1;
	}
	return 0;
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
	int64_t c, s, v, kc, i0, kk, offk;
	Ref atmp;
	int64_t c0, sc;
	Ref qv, pt, tb;
	int mode2;
	Ins *oadd[IVMAX], *ch[4], cl[IVMAX][4], *c2;
	Blk *oblk[IVMAX], *ub;
	Ref cto[4];
	uint nout, ncl[IVMAX], nch, j2, k2, q2, m2;
	Ins *a;
	uint bi, j, u, n, nin, ng, g;
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
		/* a count-down: i = i - c steps by -c */
		else if (d->op == Osub && d->cls == Kw && rtype(d->arg[0]) == RTmp && req(d->arg[0], ph->to) && ivconst(fn, d->arg[1], &s)) {
			s = -s;
			ivp = ph;
		}
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
			if (x->op != Oextsw || x->cls != Kl)
				continue;
			/* the index is i, or i + kk: the address is i's pointer at a displacement of kk * c */
			kk = 0;
			a = 0;
			c0 = 1;
			qv = R;
			pt = R;
			mode2 = 0;
			if (!req(x->arg[0], iv)) {
				if (rtype(x->arg[0]) != RTmp || fn->tmp[x->arg[0].val].bid == -1u || !fn->tmp[x->arg[0].val].def)
					continue;
				if (!inloop(mark, stamp, fn->rpo[fn->tmp[x->arg[0].val].bid]))
					continue;
				a = fn->tmp[x->arg[0].val].def;
				if (a->op == Oadd && a->cls == Kw && req(a->arg[0], iv) && ivconst(fn, a->arg[1], &kk)) {
					if (c * kk != (int64_t)(int32_t)(c * kk))
						continue;
				} else {
					/* the index is iv * c0 + q with q invariant: the base moves by c * q and the step is c * c0 * s */
					if (!ivterm(fn, mark, stamp, a, iv, &c0, &qv, &pt))
						continue;
					if (c * c0 != (int64_t)(int32_t)(c * c0))
						continue;
					mode2 = 1;
				}
			}
			/* the uses of the multiply */
			t = &fn->tmp[m->to.val];
			n = t->nuse;
			if (n == 0 || n > IVMAX)
				continue;
			found = 1;
			nin = 0;
			nout = 0;
			for (u=0; u<n; u++) {
				use = &t->use[u];
				if (use->type != UIns) {
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
				ub = fn->rpo[use->bid];
				if (!inloop(mark, stamp, ub)) {
					/* a use after the loop gets its own copy of the chain in its block (the chain's operands dominate
					   it: the header's phi and the loop's invariants), so the multiply can leave the loop */
					if (mode2 && rtype(qv) == RTmp && fn->tmp[qv.val].bid != -1u && fn->rpo[fn->tmp[qv.val].bid] == ub) {
						found = 0;
						break;
					}
					oadd[nout] = ad;
					oblk[nout] = ub;
					nout++;
					continue;
				}
				adto[nin] = ad->to;
				adbase[nin] = other;
				nin++;
			}
			if (found && nin == 0)
				found = 0;
			if (found) {
				mto = m->to;
				xto = x->to;
				offk = c * kk;
				atmp = a ? a->to : R;
				/* each use after the loop: a copy of the chain (the multiply's defining instructions, in order) before it */
				for (j2=0; j2<nout; j2++) {
					nch = 0;
					if (mode2 && !req(pt, R))
						ch[nch++] = fn->tmp[pt.val].def;
					if (a)
						ch[nch++] = a;
					ch[nch++] = x;
					ch[nch++] = m;
					ncl[j2] = nch;
					for (k2=0; k2<nch; k2++) {
						c2 = &cl[j2][k2];
						*c2 = *ch[k2];
						for (q2=0; q2<k2; q2++)
							for (m2=0; m2<2; m2++)
								if (req(c2->arg[m2], ch[q2]->to))
									c2->arg[m2] = cto[q2];
						c2->to = newtmp("ivsr", c2->cls, fn);
						cto[k2] = c2->to;
					}
					for (m2=0; m2<2; m2++)
						if (req(oadd[j2]->arg[m2], m->to))
							oadd[j2]->arg[m2] = cto[nch-1];
				}
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
	for (u=1; u<nin; u++)
		if (!req(adbase[u], adbase[0]))
			return 0;
	sc = c * c0;
	if (mode2) {
		t1 = newtmp("ivsr", Kl, fn);
		appendins(P, (Ins){Oextsw, Kl, t1, {qv, R}});
		t2 = newtmp("ivsr", Kl, fn);
		appendins(P, (Ins){Omul, Kl, t2, {t1, getcon(c, fn)}});
		tb = newtmp("ivsr", Kl, fn);
		appendins(P, (Ins){Oadd, Kl, tb, {adbase[0], t2}});
		for (u=0; u<nin; u++)
			adbase[u] = tb;
	}
	ng = 0;
	for (u=0; u<nin; u++) {
		for (g=0; g<ng; g++)
			if (req(grpb[g], adbase[u]))
				break;
		if (g < ng)
			continue;
		grpb[ng] = adbase[u];
		pv = newtmp("ivsr", Kl, fn);
		grpv[ng] = pv;
		if (isconst) {
			v = i0 * sc;
			if (v == 0)
				pinit = grpb[ng];
			else {
				pinit = newtmp("ivsr", Kl, fn);
				appendins(P, (Ins){Oadd, Kl, pinit, {grpb[ng], getcon(v, fn)}});
			}
		} else {
			t1 = newtmp("ivsr", Kl, fn);
			appendins(P, (Ins){Oextsw, Kl, t1, {init, R}});
			if (sc != 1) {
				t2 = newtmp("ivsr", Kl, fn);
				appendins(P, (Ins){Omul, Kl, t2, {t1, getcon(sc, fn)}});
				t1 = t2;
			}
			pinit = newtmp("ivsr", Kl, fn);
			appendins(P, (Ins){Oadd, Kl, pinit, {grpb[ng], t1}});
		}
		pnext = newtmp("ivsr", Kl, fn);
		appendins(L, (Ins){Oadd, Kl, pnext, {pv, getcon(sc * s, fn)}});
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
	filluse(fn);
	for (u=0; u<nin; u++) {
		for (g=0; g<ng; g++)
			if (req(grpb[g], adbase[u]))
				break;
		if (offk == 0) {
			renameuses(fn, adto[u], grpv[g]);
			killdef(fn, adto[u]);
		} else {
			/* the add of the displaced address: pointer + displacement (the load takes it in its addressing mode) */
			Ins *ad2 = fn->tmp[adto[u].val].def;
			ad2->op = Oadd;
			ad2->cls = Kl;
			ad2->arg[0] = grpv[g];
			ad2->arg[1] = getcon(offk, fn);
		}
	}
	if (offk != 0 && rtype(atmp) == RTmp && !usedanywhere(fn, atmp))
		killdef(fn, atmp);
	if (!usedanywhere(fn, mto))
		killdef(fn, mto);
	if (!usedanywhere(fn, xto))
		killdef(fn, xto);
	if (mode2) {
		if (!usedanywhere(fn, a->to))
			killdef(fn, a->to);
		if (rtype(pt) == RTmp && !usedanywhere(fn, pt))
			killdef(fn, pt);
	}
	for (j2=0; j2<nout; j2++) {
		Blk *cb = oblk[j2];
		Ins *ni = vnew(cb->nins + ncl[j2], sizeof ni[0], PFn);
		for (k2=0; k2<ncl[j2]; k2++)
			ni[k2] = cl[j2][k2];
		memcpy(ni + ncl[j2], cb->ins, cb->nins * sizeof ni[0]);
		cb->ins = ni;
		cb->nins += ncl[j2];
	}
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

/* loop rotation (nfcxx): the header's test runs again at the latch, so the back edge is the test's branch and no jump
   goes back to the header. A block is in the region of the loop's body (inside: a loop block but not the header) or of
   its exit (outside); a use in a phi counts in the block of its predecessor. */
static int
inreg(uint *mark, uint stamp, Blk *b, Blk *h, int inside)
{
	if (inside)
		return b != h && inloop(mark, stamp, b);
	return !inloop(mark, stamp, b);
}

static int
usedregion(Fn *fn, Ref r, Blk *h, uint *mark, uint stamp, int inside)
{
	Blk *b;
	Phi *p;
	Ins *i;
	uint a, m;

	for (b=fn->start; b; b=b->link) {
		for (p=b->phi; p; p=p->link)
			for (a=0; a<p->narg; a++)
				if (req(p->arg[a], r) && inreg(mark, stamp, p->blk[a], h, inside))
					return 1;
		if (!inreg(mark, stamp, b, h, inside))
			continue;
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

static void
renameregion(Fn *fn, Ref from, Ref to, Blk *h, uint *mark, uint stamp, int inside)
{
	Blk *b;
	Phi *p;
	Ins *i;
	uint a, m;

	for (b=fn->start; b; b=b->link) {
		for (p=b->phi; p; p=p->link)
			for (a=0; a<p->narg; a++)
				if (req(p->arg[a], from) && inreg(mark, stamp, p->blk[a], h, inside))
					p->arg[a] = to;
		if (!inreg(mark, stamp, b, h, inside))
			continue;
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			if (i->op != Onop)
				for (m=0; m<2; m++)
					if (req(i->arg[m], from))
						i->arg[m] = to;
		if (req(b->jmp.arg, from))
			b->jmp.arg = to;
	}
}

static Phi *
newphi2(Fn *fn, Ref to, int cls, Blk *b0, Ref a0, Blk *b1, Ref a1)
{
	Phi *np;

	(void)fn;
	np = vnew(1, sizeof np[0], PFn);
	np->to = to;
	np->cls = cls;
	np->narg = 2;
	np->arg = vnew(2, sizeof np->arg[0], PFn);
	np->blk = vnew(2, sizeof np->blk[0], PFn);
	np->arg[0] = a0;
	np->blk[0] = b0;
	np->arg[1] = a1;
	np->blk[1] = b1;
	return np;
}

/* the loop with header h (entry P, latch L) is rotated if it has the shape: h jumps by a jnz on a comparison that is
   its only instruction, the body is entered only from h, the exit is entered only from h, and L jumps to h */
static int
rotate1(Fn *fn, Blk *h, Blk *L, uint *mark, uint stamp)
{
	Blk *body, *ex;
	Phi *p, *np;
	Ins *i, *cmp;
	Ref al, r, nt;
	int kk, cc, ncmp;
	uint a, n;

	body = h->s1;
	ex = h->s2;
	if (h->jmp.type != Jjnz || L == h || L->jmp.type != Jjmp || L->s1 != h || L->s2)
		return 0;
	if (!inloop(mark, stamp, body) || inloop(mark, stamp, ex) || body->npred != 1 || body->phi)
		return 0;
	if (ex->npred != 1 || ex->phi)
		return 0;
	/* the header is the loop's only exit: no other block leaves the loop (a break) or returns from it */
	{
		Blk *q;

		for (q=fn->start; q; q=q->link) {
			if (q == h || !inloop(mark, stamp, q))
				continue;
			if (isret(q->jmp.type) || (q->s1 && !inloop(mark, stamp, q->s1)) || (q->s2 && !inloop(mark, stamp, q->s2)))
				return 0;
		}
	}
	ncmp = 0;
	cmp = 0;
	for (i=h->ins; i<&h->ins[h->nins]; i++) {
		if (i->op == Onop || i->op == Odbgloc)
			continue;
		if (!iscmp(i->op, &kk, &cc) || !req(i->to, h->jmp.arg) || fn->tmp[i->to.val].nuse != 1)
			return 0;
		cmp = i;
		ncmp++;
	}
	if (ncmp != 1)
		return 0;
	for (p=h->phi; p; p=p->link) {
		n = p->narg;
		for (a=0; a<n; a++)
			if (p->blk[a] == L)
				break;
		if (a == n)
			return 0;
	}

	/* the header's phis: a phi used in the body or in the loop gets a phi at the body; one used after the loop gets
	   a phi at the exit; both take the header's value on entry and the latch's value on the back edge */
	for (p=h->phi; p; p=p->link) {
		for (a=0; a<p->narg; a++)
			if (p->blk[a] == L)
				break;
		al = p->arg[a];
		if (usedregion(fn, p->to, h, mark, stamp, 1)) {
			nt = newtmp("rot", p->cls, fn);
			renameregion(fn, p->to, nt, h, mark, stamp, 1);
			np = newphi2(fn, nt, p->cls, h, p->to, L, al);
			np->link = body->phi;
			body->phi = np;
		}
		if (usedregion(fn, p->to, h, mark, stamp, 0)) {
			nt = newtmp("rot", p->cls, fn);
			renameregion(fn, p->to, nt, h, mark, stamp, 0);
			np = newphi2(fn, nt, p->cls, h, p->to, L, al);
			np->link = ex->phi;
			ex->phi = np;
		}
	}

	/* the latch runs the test for the next iteration: the header's phis become their latch values */
	{
		Ins c = *cmp;
		for (a=0; a<2; a++) {
			r = c.arg[a];
			for (p=h->phi; p; p=p->link)
				if (req(r, p->to)) {
					uint q;
					for (q=0; q<p->narg; q++)
						if (p->blk[q] == L)
							break;
					c.arg[a] = p->arg[q];
					break;
				}
		}
		c.to = newtmp("rot", c.cls, fn);
		appendins(L, c);
		L->jmp.type = Jjnz;
		L->jmp.arg = c.to;
		L->s1 = body;
		L->s2 = ex;
	}
	fillcfg(fn);
	filluse(fn);
	return 1;
}

void
rotate(Fn *fn)
{
	Blk *h, *P, *L, *b, **stack;
	uint *mark, stamp, n, p, nlatch, nentry;
	int changed;

	fillcfg(fn);
	filldom(fn);
	filluse(fn);
	for (n=0; n<fn->nblk; n++) {
		h = fn->rpo[n];
		if (h->npred != 2)
			continue;
		P = L = 0;
		nlatch = nentry = 0;
		for (p=0; p<h->npred; p++) {
			b = h->pred[p];
			if (dom(h, b)) {
				L = b;
				nlatch++;
			} else {
				P = b;
				nentry++;
			}
		}
		if (nlatch != 1 || nentry != 1 || P->jmp.type != Jjmp || P->s1 != h || P->s2)
			continue;
		mark = emalloc(fn->nblk * sizeof mark[0]);
		for (p=0; p<fn->nblk; p++)
			mark[p] = 0;
		stamp = 1;
		loopbody(h, mark, stamp, &stack);
		changed = rotate1(fn, h, L, mark, stamp);
		free(mark);
		if (changed) {
			filldom(fn);
			filluse(fn);
		}
	}
}
