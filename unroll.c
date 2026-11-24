#include "all.h"

/* Unroll a counted self-loop by four, with a guard.

   The loop is a block B that jumps back to itself on a test t = csltw(in, K), where in = iv + c (c > 0 a constant)
   and iv is a phi of B whose back edge is in. B's test runs first; its success goes to a guard G, which checks
   in < K - 3c: then the three further iterations' tests (in + k*c < K for k = 1..3) all hold, and the unchecked
   copies U1..U3 run them without tests, then go back to B. If the guard fails, the checked copies C1..C3 run
   the iterations one by one with their tests, and C3 goes back to B.

   The copies are built in SSA form (no moves): a value of B used in a copy is the copy's own definition, and a phi
   of B takes, in a copy, the value its back edge carried out of the previous copy. Exit phis get an entry per
   checked copy (the unchecked copies never leave the loop).

   Conservative: B has two predecessors (the preheader and B), no calls, no stack allocation, no argument or
   parameter instructions, and every value defined in B is used only in B, in a phi of ex, or in an instruction of
   ex (only when B is ex's only predecessor). */

static void
unr_append(Blk *b, Ins i)
{
	Ins *ni;

	ni = vnew(b->nins + 1, sizeof ni[0], PFn);
	memcpy(ni, b->ins, b->nins * sizeof ni[0]);
	ni[b->nins] = i;
	b->ins = ni;
	b->nins++;
}

enum { UnrollCopies = 4, UnrollMaxIns = 32 };

static Ref
mapref(Ref *m, Ref r)
{
	if (rtype(r) != RTmp)
		return r;
	return m[r.val];
}

static int
conbits(Fn *fn, Ref r, int64_t *v)
{
	if (rtype(r) != RCon || fn->con[r.val].type != CBits)
		return 0;
	*v = fn->con[r.val].bits.i;
	return 1;
}

/* a use of a value of b: in b; or a phi of ex (the phi's entry from b); or an instruction of ex, which b dominates
   only when b is ex's only predecessor */
static int
okuse(Use *u, Blk *b, Blk *ex)
{
	if (u->bid == b->id)
		return 1;
	if (u->bid != ex->id)
		return 0;
	return u->type == UPhi || ex->npred == 1;
}

/* the test of b is csltw(in, K) with in = iv + c, c > 0, and iv a phi of b with back edge in; the guard constant is
   K - 3c (the test of the copy that runs with in + 3c) */
static int
guardform(Fn *fn, Blk *b, Ins *cmpout, int64_t *k3)
{
	Ins *i, *cmp, *add;
	Phi *p;
	Ref in, iv;
	uint a, n;
	int64_t c, k, lo, hi;

	cmp = 0;
	n = 0;
	for (i=b->ins; i<&b->ins[b->nins]; i++)
		if (i->op != Onop && i->op != Odbgloc && req(i->to, b->jmp.arg)) {
			cmp = i;
			n++;
		}
	if (n != 1 || (cmp->op != Ocsltw && cmp->op != Ocsltl))
		return 0;
	if (rtype(cmp->arg[0]) != RTmp || !conbits(fn, cmp->arg[1], &k))
		return 0;
	in = cmp->arg[0];
	add = 0;
	for (i=b->ins; i<&b->ins[b->nins]; i++)
		if (i->op != Onop && i->op != Odbgloc && req(i->to, in))
			add = i;
	if (!add || add->op != Oadd || rtype(add->arg[0]) != RTmp || !conbits(fn, add->arg[1], &c) || c <= 0)
		return 0;
	iv = add->arg[0];
	for (p=b->phi; p; p=p->link)
		if (req(p->to, iv))
			break;
	if (!p)
		return 0;
	for (a=0; a<p->narg; a++)
		if (p->blk[a] == b)
			break;
	if (a == p->narg || !req(p->arg[a], in))
		return 0;
	/* the guard constant must not wrap in the comparison's class */
	if (cmp->op == Ocsltw) {
		lo = INT32_MIN;
		hi = INT32_MAX;
	} else {
		lo = INT64_MIN / 2;
		hi = INT64_MAX / 2;
	}
	if (c > hi || k < lo + 3*c || k - 3*c < lo || k > hi)
		return 0;
	*cmpout = *cmp;
	*k3 = k - 3*c;
	return 1;
}

static int
unrollok(Fn *fn, Blk *b, Blk *P, Blk *ex, Ins *cmp, int64_t *k3)
{
	Phi *p;
	Ins *i;
	uint n, t;
	Use *u;

	if (b->npred != 2 || (b->pred[0] != P && b->pred[1] != P))
		return 0;
	if (ex == b || ex == P || P == b || b->ntab || b->ehlp || b->ehpad || b->ehmode != EhDefault || b->tail)
		return 0;
	for (p=b->phi; p; p=p->link)
		if (p->narg != 2 || rtype(p->to) != RTmp)
			return 0;
	n = 0;
	for (i=b->ins; i<&b->ins[b->nins]; i++) {
		if (i->op == Onop || i->op == Odbgloc)
			continue;
		if (i->op == Ocall || isalloc(i->op) || isarg(i->op) || ispar(i->op) || isparbh(i->op))
			return 0;
		if (!req(i->to, R) && (rtype(i->to) != RTmp || i->to.val < Tmp0))
			return 0;
		n++;
	}
	if (n == 0 || n > UnrollMaxIns)
		return 0;
	if (!guardform(fn, b, cmp, k3))
		return 0;

	/* every value defined in b is used only in b, in a phi of ex, or in an instruction of ex */
	for (p=b->phi; p; p=p->link) {
		t = p->to.val;
		for (u=fn->tmp[t].use; u<&fn->tmp[t].use[fn->tmp[t].nuse]; u++)
			if (!okuse(u, b, ex))
				return 0;
	}
	for (i=b->ins; i<&b->ins[b->nins]; i++) {
		if (req(i->to, R) || i->op == Odbgloc || i->op == Onop)
			continue;
		t = i->to.val;
		if (fn->tmp[t].ndef != 1)
			return 0;
		for (u=fn->tmp[t].use; u<&fn->tmp[t].use[fn->tmp[t].nuse]; u++)
			if (!okuse(u, b, ex))
				return 0;
	}
	return 1;
}

/* the naming of b's values in a copy: each phi carries the value its back edge had in the previous copy; each
   definition gets a fresh temp */
static void
buildmap(Fn *fn, Blk *b, Ref *prev, Ref *cur, Ref *ident, uint nt)
{
	Phi *p;
	Ins *i;
	uint t, a;

	for (t=0; t<nt; t++)
		cur[t] = ident[t];
	for (p=b->phi; p; p=p->link) {
		for (a=0; a<p->narg; a++)
			if (p->blk[a] == b)
				break;
		cur[p->to.val] = mapref(prev, p->arg[a]);
	}
	for (i=b->ins; i<&b->ins[b->nins]; i++) {
		if (i->op == Onop || i->op == Odbgloc || req(i->to, R))
			continue;
		cur[i->to.val] = newtmp("unr", i->cls, fn);
	}
}

static void
emitcopy(Blk *nb, Blk *b, Ref *m)
{
	Ins *i, ni;

	for (i=b->ins; i<&b->ins[b->nins]; i++) {
		if (i->op == Onop || i->op == Odbgloc)
			continue;
		ni = *i;
		if (!req(i->to, R))
			ni.to = m[i->to.val];
		ni.arg[0] = mapref(m, i->arg[0]);
		ni.arg[1] = mapref(m, i->arg[1]);
		unr_append(nb, ni);
	}
}

/* a value of b used by an instruction of ex (a phi of b, or a def that GCM sank there): ex gets a phi of it, with
   the value from b and from each checked copy */
static void
exitphi(Fn *fn, Blk *b, Blk *ex, Blk **c, Ref **m, Ref def, int cls)
{
	Ins *ei;
	Phi *np;
	Ref nt;
	uint k;
	int used;

	used = req(ex->jmp.arg, def);
	for (ei=ex->ins; ei<&ex->ins[ex->nins]; ei++)
		if (req(ei->arg[0], def) || req(ei->arg[1], def))
			used = 1;
	if (!used)
		return;
	nt = newtmp("unx", cls, fn);
	np = vnew(1, sizeof np[0], PFn);
	np->to = nt;
	np->cls = cls;
	np->narg = UnrollCopies;
	np->arg = vnew(np->narg, sizeof np->arg[0], PFn);
	np->blk = vnew(np->narg, sizeof np->blk[0], PFn);
	np->arg[0] = def;
	np->blk[0] = b;
	for (k=1; k<UnrollCopies; k++) {
		np->arg[k] = m[k][def.val];
		np->blk[k] = c[k];
	}
	np->link = ex->phi;
	ex->phi = np;
	for (ei=ex->ins; ei<&ex->ins[ex->nins]; ei++) {
		if (req(ei->arg[0], def))
			ei->arg[0] = nt;
		if (req(ei->arg[1], def))
			ei->arg[1] = nt;
	}
	if (req(ex->jmp.arg, def))
		ex->jmp.arg = nt;
}

static void
unrollblk(Fn *fn, Blk *b, Blk *ex, Ins *cmp, int64_t k3)
{
	Blk *c[UnrollCopies], *u[UnrollCopies], *G, *nb, *prev;
	Ref *mc[UnrollCopies], *mu[UnrollCopies], *ident, g, orig;
	Phi *p;
	Ins *i;
	uint k, a, t, nt, n;

	nt = fn->ntmp;
	ident = emalloc(nt * sizeof ident[0]);
	for (t=0; t<nt; t++)
		ident[t] = TMP(t);
	mc[0] = mu[0] = ident;
	for (k=1; k<UnrollCopies; k++) {
		mc[k] = emalloc(nt * sizeof mc[k][0]);
		mu[k] = emalloc(nt * sizeof mu[k][0]);
		buildmap(fn, b, mc[k-1], mc[k], ident, nt);
		buildmap(fn, b, mu[k-1], mu[k], ident, nt);
	}

	/* the blocks, linked after b: G, the checked copies C1..C3, the unchecked copies U1..U3 */
	G = newblk();
	nb = b->link;
	prev = G;
	for (k=1; k<UnrollCopies; k++) {
		c[k] = newblk();
		prev->link = c[k];
		prev = c[k];
	}
	for (k=1; k<UnrollCopies; k++) {
		u[k] = newblk();
		prev->link = u[k];
		prev = u[k];
	}
	b->link = G;
	prev->link = nb;

	/* the checked copies: each keeps its test; the exit goes to ex */
	for (k=1; k<UnrollCopies; k++) {
		emitcopy(c[k], b, mc[k]);
		c[k]->jmp.type = Jjnz;
		c[k]->jmp.arg = mapref(mc[k], b->jmp.arg);
		c[k]->s1 = k+1 < UnrollCopies ? c[k+1] : b;
		c[k]->s2 = ex;
	}
	/* the unchecked copies: no test; U3 goes back to b */
	for (k=1; k<UnrollCopies; k++) {
		emitcopy(u[k], b, mu[k]);
		u[k]->jmp.type = Jjmp;
		u[k]->s1 = k+1 < UnrollCopies ? u[k+1] : b;
	}

	/* the guard: the three extra tests hold when in + 3c < K, i.e. in < K - 3c */
	g = newtmp("unrg", cmp->cls, fn);
	unr_append(G, (Ins){cmp->op, cmp->cls, g, {cmp->arg[0], getcon(k3, fn)}});
	G->jmp.type = Jjnz;
	G->jmp.arg = g;
	G->s1 = u[1];
	G->s2 = c[1];
	b->s1 = G;

	/* b's phis: the back edge comes from C3 (checked) or U3 (unchecked), each with its own value */
	for (p=b->phi; p; p=p->link)
		for (a=0; a<p->narg; a++)
			if (p->blk[a] == b) {
				orig = p->arg[a];
				p->blk[a] = c[UnrollCopies-1];
				p->arg[a] = mapref(mc[UnrollCopies-1], orig);
				n = p->narg++;
				vgrow(&p->arg, p->narg);
				vgrow(&p->blk, p->narg);
				p->arg[n] = mapref(mu[UnrollCopies-1], orig);
				p->blk[n] = u[UnrollCopies-1];
				break;
			}

	/* the exit: each checked copy leaves to ex with its own values */
	for (p=ex->phi; p; p=p->link)
		for (a=0; a<p->narg; a++)
			if (p->blk[a] == b) {
				for (k=1; k<UnrollCopies; k++) {
					n = p->narg++;
					vgrow(&p->arg, p->narg);
					vgrow(&p->blk, p->narg);
					p->arg[n] = mapref(mc[k], p->arg[a]);
					p->blk[n] = c[k];
				}
				break;
			}
	/* a value of b used by ex's instructions: a phi of ex */
	for (p=b->phi; p; p=p->link)
		exitphi(fn, b, ex, c, mc, p->to, p->cls);
	for (i=b->ins; i<&b->ins[b->nins]; i++)
		if (!req(i->to, R) && i->op != Odbgloc && i->op != Onop)
			exitphi(fn, b, ex, c, mc, i->to, i->cls);

	for (k=1; k<UnrollCopies; k++) {
		free(mc[k]);
		free(mu[k]);
	}
	free(ident);
}

void
unroll(Fn *fn)
{
	Blk *b, *P, *ex, *next;
	Ins cmp;
	int64_t k3;
	int changed;

	fillcfg(fn);
	filluse(fn);
	changed = 0;
	for (b=fn->start; b; b=next) {
		next = b->link;
		if (b->jmp.type != Jjnz || b->s1 != b)
			continue;
		ex = b->s2;
		P = b->pred[0] == b ? b->pred[1] : b->pred[0];
		if (!ex || !P || P == b)
			continue;
		if (!unrollok(fn, b, P, ex, &cmp, &k3))
			continue;
		unrollblk(fn, b, ex, &cmp, k3);
		changed = 1;
	}
	if (changed) {
		fillcfg(fn);
		filluse(fn);
	}
}
