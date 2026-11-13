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
