#include "all.h"

enum {
	SecText,
	SecData,
	SecBss,
};

void
emitlnk(char *n, Lnk *l, int s, FILE *f)
{
	static char *sec[2][3] = {
		[0][SecText] = ".text",
		[0][SecData] = ".data",
		[0][SecBss] = ".bss",
		[1][SecText] = ".abort \"unreachable\"",
		[1][SecData] = ".section .tdata,\"awT\"",
		[1][SecBss] = ".section .tbss,\"awT\"",
	};
	char *pfx, *sfx;

	pfx = n[0] == '"' ? "" : T.assym;
	sfx = "";
	if (T.apple && l->thread) {
		l->sec = "__DATA";
		l->secf = "__thread_data,thread_local_regular";
		sfx = "$tlv$init";
		fputs(
			".section __DATA,__thread_vars,"
			"thread_local_variables\n",
			f
		);
		fprintf(f, "%s%s:\n", pfx, n);
		fprintf(f,
			"\t.quad __tlv_bootstrap\n"
			"\t.quad 0\n"
			"\t.quad %s%s%s\n\n",
			pfx, n, sfx
		);
	}
	if (l->sec) {
		fprintf(f, ".section %s", l->sec);
		if (l->secf)
			fprintf(f, ",%s", l->secf);
	} else
		fputs(sec[l->thread != 0][s], f);
	fputc('\n', f);
	if (l->align)
		fprintf(f, ".balign %d\n", l->align);
	if (l->export)
		fprintf(f, ".globl %s%s\n", pfx, n);
	fprintf(f, "%s%s%s:\n", pfx, n, sfx);
}

void
emitfnlnk(char *n, Lnk *l, FILE *f)
{
	emitlnk(n, l, SecText, f);
}

/* Landing pads (fn->eh_lp, the block labeled eh.lp; live.c). The system unwinder enters the pad at a call of the function
   that a throw unwinds, and the pad resumes the unwind (cproc's cleanup lowering, nfcxx). The call sites and their pad are
   in the function's LSDA (.gcc_except_table), which the personality reads: gcc's __gcc_personality_v0, which with no
   catch types treats a call site with action 0 as a cleanup. The landing pads use DW_EH_PE_* offsets from the function.

   Exception tables (nfcxx, fn->ehtab: the ehtype/ehpad/ehunwind/ehnoexcept directives, all.h Blk.ehmode): several
   landing pads, catch clauses and std::terminate, in the Itanium C++ ABI's LSDA. The personality is
   libstdc++'s __gxx_personality_v0; a landing pad reads the exception and the selector it installs from their registers
   (ehptr, ehsel: %rax and %rdx, x0 and x1; the abi passes make them copies), as gcc's __builtin_eh_pointer does.
   A call of a block with a pad (EhPad) is a call site with that pad and
   the pad's actions (its catch clauses' type ids, then a cleanup when ehclean); a call that unwinds through the
   function (EhDefault) is a call site with no pad; a call that terminates (EhTerm, or EhDefault in an ehnoexcept
   function) has no call site, and the personality calls std::terminate when an exception leaves it. */
static uint ehnext, ehcur, ehcs;   /* numbers of the functions with a pad, of the current one, and of its call sites */
static uint *ehcsv;                /* the current function's call sites: the label number of each one's start */
static uint *ehcsve;               /* and of its end: a site merges the calls after it with the same pad */
static uint ehcscur;               /* the label number of the call being emitted */
static int ehcsbrk;
static int ehcsskip;               /* the call being emitted is of a nothrow callee (ehnothrowcall) */                /* a call with no call site came since the last one: the next does not merge */
static Blk **ehcslp;               /* exception tables: the landing pad of each call site (0: none) */
static uint nehcsv, ehcsvcap;
static int ehtabon;                /* the current function has exception tables (an LSDA) */

static int
ehneedtab(Fn *fn)
{
	return fn->ehtab && (fn->nehpad || fn->ehnoexc || fn->nehspec);
}

void
ehfnbegin(Fn *fn, FILE *f)
{
	ehtabon = 0;
	/* a leaf makes no call: nothing unwinds through it, and the target writes no CFI for it */
	if (!fn->eh_lp && (!ehneedtab(fn) || fn->leaf))
		return;
	if (T.apple)
		die("landing pads need ELF unwind tables (no Apple target)");
	ehcur = ++ehnext;
	nehcsv = 0;
	ehcsbrk = 0;
	ehtabon = !fn->eh_lp;
	fprintf(f,
		".LEHfs%u:\n"
		"\t.cfi_personality 0x9b, DW.ref.%s\n"
		"\t.cfi_lsda 0x1b, .LEHL%u\n",
		ehcur, ehtabon ? "__gxx_personality_v0" : "__gcc_personality_v0", ehcur
	);
}

void
ehlpbegin(Fn *fn, Blk *b, FILE *f)
{
	if (ehtabon) {
		if (b->ehpad)
			fprintf(f, ".LEHp%u_%u:\n", ehcur, b->id);
		return;
	}
	if (b == fn->eh_lp)
		fprintf(f, ".LEHlp%u:\n", ehcur);
}

/* the landing pad of a call of b (exception tables), or -1 when the call has no call site entry */
static Blk *
ehcallpad(Fn *fn, Blk *b, int *none)
{
	*none = 0;
	switch (b->ehmode) {
	case EhPad:
		return b->ehlp;
	case EhTerm:
		*none = 1;
		return 0;
	default:
		*none = fn->ehnoexc;
		return 0;
	}
}

/* a direct call of a callee the function lists with ehnothrow: no exception leaves it, so it needs no call site
   entry, and it may sit inside the range of any (as gcc's calls of nothrow functions do) */
static int
ehnothrowcall(Fn *fn, Ins *i)
{
	Con *c;
	uint k;

	if (!fn->nehnothrow || rtype(i->arg[0]) != RCon)
		return 0;
	c = &fn->con[i->arg[0].val];
	if (c->type != CAddr || c->bits.i != 0)
		return 0;
	for (k=0; k<fn->nehnothrow; k++)
		if (fn->ehnothrow[k] == c->sym.id)
			return 1;
	return 0;
}

/* a call of the function, outside its landing pad: its address range is a call site of the LSDA */
void
ehcsbegin(Fn *fn, Blk *b, Ins *i, FILE *f)
{
	Blk *lp;
	int none;

	ehcsskip = ehnothrowcall(fn, i);
	if (ehcsskip)
		return;
	if (ehtabon) {
		lp = ehcallpad(fn, b, &none);
		if (none) {
			ehcsbrk = 1;
			return;
		}
	} else {
		if (!fn->eh_lp)
			return;
		if (b == fn->eh_lp) {
			ehcsbrk = 1;
			return;
		}
		lp = 0;
	}
	ehcscur = ++ehcs;
	/* as gcc does, a call with the pad of the call site before it, and no call without a site between them, extends
	   that site: the instructions between two calls throw nothing */
	if (nehcsv && !ehcsbrk && ehcslp[nehcsv-1] == lp) {
		ehcsve[nehcsv-1] = ehcscur;
		return;
	}
	ehcsbrk = 0;
	if (nehcsv == ehcsvcap) {
		ehcsvcap = ehcsvcap ? 2 * ehcsvcap : 16;
		ehcsv = realloc(ehcsv, ehcsvcap * sizeof ehcsv[0]);
		ehcsve = realloc(ehcsve, ehcsvcap * sizeof ehcsve[0]);
		ehcslp = realloc(ehcslp, ehcsvcap * sizeof ehcslp[0]);
		if (!ehcsv || !ehcsve || !ehcslp)
			die("out of memory");
	}
	ehcslp[nehcsv] = lp;
	ehcsve[nehcsv] = ehcscur;
	ehcsv[nehcsv++] = ehcscur;
	fprintf(f, ".LEHcsb%u:\n", ehcscur);
}

void
ehcsend(Fn *fn, Blk *b, Ins *i, FILE *f)
{
	int none;

	(void)i;
	if (ehcsskip)
		return;
	if (ehtabon) {
		ehcallpad(fn, b, &none);
		if (none)
			return;
	} else if (!fn->eh_lp || b == fn->eh_lp)
		return;
	fprintf(f, ".LEHcse%u:\n", ehcscur);
}

/* DW.ref.NAME: an indirection the LSDA and the CIE reach PC-relatively (as gcc does), once per file */
static void
ehdwref(char *name, FILE *f)
{
	static char **done;
	static uint ndone;
	uint k;

	for (k=0; k<ndone; k++)
		if (strcmp(done[k], name) == 0)
			return;
	done = realloc(done, (ndone + 1) * sizeof done[0]);
	if (!done)
		die("out of memory");
	/* a copy, not name: a type's symbol is in the function's pool, freed after it */
	done[ndone] = malloc(strlen(name) + 1);
	if (!done[ndone])
		die("out of memory");
	strcpy(done[ndone++], name);
	fprintf(f,
		"\t.hidden DW.ref.%s\n"
		"\t.weak DW.ref.%s\n"
		"\t.section .data.rel.local.DW.ref.%s,\"awG\",@progbits,DW.ref.%s,comdat\n"
		"\t.p2align 3\n"
		"\t.type DW.ref.%s, @object\n"
		"\t.size DW.ref.%s, 8\n"
		"DW.ref.%s:\n"
		"\t.quad %s\n"
		"\t.text\n",
		name, name, name, name, name, name, name, name
	);
}

static void
ehtabend(Fn *fn, FILE *f)
{
	uint k, c, t;
	Blk *lp;
	int types;

	types = fn->nehty > 1 || fn->nehspec;
	fprintf(f,
		"\t.section .gcc_except_table,\"a\",@progbits\n"
		"\t.p2align 2\n"
		".LEHL%u:\n"
		"\t.byte 0xff\n",         /* no LPStart: the pads are offsets from the function */
		ehcur
	);
	if (types)
		fprintf(f,
			"\t.byte 0x9b\n"   /* types: DW_EH_PE_indirect | pcrel | sdata4 (DW.ref) */
			"\t.uleb128 .LEHTT%u-.LEHTTb%u\n"
			".LEHTTb%u:\n",
			ehcur, ehcur, ehcur
		);
	else
		fputs("\t.byte 0xff\n", f);
	fprintf(f,
		"\t.byte 0x01\n"          /* call sites: uleb128 */
		"\t.uleb128 .LEHCTe%u-.LEHCTb%u\n"
		".LEHCTb%u:\n",
		ehcur, ehcur, ehcur
	);
	for (k=0; k<nehcsv; k++) {
		c = ehcsv[k];
		lp = ehcslp[k];
		fprintf(f,
			"\t.uleb128 .LEHcsb%u-.LEHfs%u\n"
			"\t.uleb128 .LEHcse%u-.LEHcsb%u\n",
			c, ehcur, ehcsve[k], c
		);
		if (!lp)
			fputs("\t.uleb128 0\n\t.uleb128 0\n", f);
		else if (!lp->nehact && !fn->nehspec)
			fprintf(f, "\t.uleb128 .LEHp%u_%u-.LEHfs%u\n\t.uleb128 0\n", ehcur, lp->id, ehcur);
		else
			fprintf(f,
				"\t.uleb128 .LEHp%u_%u-.LEHfs%u\n"
				"\t.uleb128 .LEHA%u_%u-.LEHAT%u+1\n",
				ehcur, lp->id, ehcur, ehcur, lp->id, ehcur
			);
	}
	/* the action table: for each pad with catch clauses, one record per clause (its type id, then the offset to the
	   next record, 1 as it follows), and a cleanup record (filter 0) last when the pad also runs cleanups */
	fprintf(f, ".LEHCTe%u:\n.LEHAT%u:\n", ehcur, ehcur);
	for (k=0; k<fn->nehpad; k++) {
		lp = fn->ehpads[k];
		if (!lp->nehact && !fn->nehspec)
			continue;
		fprintf(f, ".LEHA%u_%u:\n", ehcur, lp->id);
		for (t=0; t<lp->nehact; t++) {
			fprintf(f, "\t.sleb128 %d\n", lp->ehact[t]);
			fprintf(f, "\t.sleb128 %d\n", t+1 < lp->nehact || lp->ehclean || fn->nehspec ? 1 : 0);
		}
		/* the dynamic exception specification: filter -1 (its list at offset 0 after .LEHTT) */
		if (fn->nehspec)
			fprintf(f, "\t.sleb128 -1\n\t.sleb128 %d\n", lp->ehclean ? 1 : 0);
		if (lp->ehclean)
			fputs("\t.sleb128 0\n\t.sleb128 0\n", f);
	}
	/* the types table, last entry first: type id N is the N-th entry before .LEHTT */
	if (types) {
		fputs("\t.p2align 2\n", f);
		for (t=fn->nehty-1; t>=1; t--)
			if (fn->ehty[t])
				fprintf(f, "\t.long DW.ref.%s-.\n", fn->ehty[t]);
			else
				fputs("\t.long 0\n", f);
		fprintf(f, ".LEHTT%u:\n", ehcur);
		/* the exception specification list: its type ids, 0 at the end */
		if (fn->nehspec) {
			for (t=0; t<fn->nehspec; t++)
				fprintf(f, "\t.uleb128 %d\n", fn->ehspec[t]);
			fputs("\t.uleb128 0\n", f);
		}
	}
	fputs("\t.text\n", f);
	ehdwref("__gxx_personality_v0", f);
	for (t=1; t<fn->nehty; t++)
		if (fn->ehty[t])
			ehdwref(fn->ehty[t], f);
}

void
ehfnend(Fn *fn, FILE *f)
{
	static int persdef;   /* the personality's DW.ref, defined once per file (as gcc does) */
	uint k, c;

	if (ehtabon) {
		ehtabend(fn, f);
		ehtabon = 0;
		return;
	}
	if (!fn->eh_lp)
		return;
	fprintf(f,
		"\t.section .gcc_except_table,\"a\",@progbits\n"
		"\t.p2align 2\n"
		".LEHL%u:\n"
		"\t.byte 0xff\n"          /* no LPStart: the pads are offsets from the function */
		"\t.byte 0xff\n"          /* no types table: no catch clauses */
		"\t.byte 0x01\n"          /* call sites: uleb128 */
		"\t.uleb128 .LEHCTe%u-.LEHCTb%u\n"
		".LEHCTb%u:\n",
		ehcur, ehcur, ehcur, ehcur
	);
	for (k=0; k<nehcsv; k++) {
		c = ehcsv[k];
		fprintf(f,
			"\t.uleb128 .LEHcsb%u-.LEHfs%u\n"
			"\t.uleb128 .LEHcse%u-.LEHcsb%u\n"
			"\t.uleb128 .LEHlp%u-.LEHfs%u\n"
			"\t.uleb128 0\n",
			c, ehcur, ehcsve[k], c, ehcur, ehcur
		);
	}
	fprintf(f, ".LEHCTe%u:\n\t.text\n", ehcur);
	if (persdef)
		return;
	persdef = 1;
	fputs(
		"\t.hidden DW.ref.__gcc_personality_v0\n"
		"\t.weak DW.ref.__gcc_personality_v0\n"
		"\t.section .data.rel.local.DW.ref.__gcc_personality_v0,\"awG\",@progbits,DW.ref.__gcc_personality_v0,comdat\n"
		"\t.p2align 3\n"
		"\t.type DW.ref.__gcc_personality_v0, @object\n"
		"\t.size DW.ref.__gcc_personality_v0, 8\n"
		"DW.ref.__gcc_personality_v0:\n"
		"\t.quad __gcc_personality_v0\n"
		"\t.text\n",
		f
	);
}

void
emitdat(Dat *d, FILE *f)
{
	static struct {
		char decl[8];
		int64_t mask;
	} di[] = {
		[DB] = {"\t.byte", 0xffL},
		[DH] = {"\t.short", 0xffffL},
		[DW] = {"\t.int", 0xffffffffL},
		[DL] = {"\t.quad", -1L},
	};
	static int64_t zero;
	char *p;

	switch (d->type) {
	case DStart:
		zero = 0;
		break;
	case DEnd:
		if (d->lnk->common) {
			if (zero == -1)
				die("invalid common data definition");
			p = d->name[0] == '"' ? "" : T.assym;
			fprintf(f, ".comm %s%s,%"PRId64,
				p, d->name, zero);
			if (d->lnk->align)
				fprintf(f, ",%d", d->lnk->align);
			fputc('\n', f);
		}
		else if (zero != -1) {
			emitlnk(d->name, d->lnk, SecBss, f);
			fprintf(f, "\t.fill %"PRId64",1,0\n", zero);
		}
		/* ELF symbol type and size, as gcc writes them for an object (nfcxx: the Linux kernel's build compares them) */
		if (!d->lnk->common && !T.apple) {
			p = d->name[0] == '"' ? "" : T.assym;
			fprintf(f, ".type %s%s, @object\n", p, d->name);
			if (zero != -1)
				fprintf(f, ".size %s%s, %"PRId64"\n", p, d->name, zero);
			else
				fprintf(f, ".size %s%s, .-%s%s\n", p, d->name, p, d->name);
		}
		break;
	case DZ:
		if (zero != -1)
			zero += d->u.num;
		else
			fprintf(f, "\t.fill %"PRId64",1,0\n", d->u.num);
		break;
	default:
		if (zero != -1) {
			emitlnk(d->name, d->lnk, SecData, f);
			if (zero > 0)
				fprintf(f, "\t.fill %"PRId64",1,0\n", zero);
			zero = -1;
		}
		if (d->isstr) {
			if (d->type != DB)
				err("strings only supported for 'b' currently");
			fprintf(f, "\t.ascii %s\n", d->u.str);
		}
		else if (d->isref) {
			p = d->u.ref.name[0] == '"' ? "" : T.assym;
			fprintf(f, "%s %s%s%+"PRId64"\n",
				di[d->type].decl, p, d->u.ref.name,
				d->u.ref.off);
		}
		else {
			fprintf(f, "%s %"PRId64"\n",
				di[d->type].decl,
				d->u.num & di[d->type].mask);
		}
		break;
	}
}

typedef struct Asmbits Asmbits;

struct Asmbits {
	bits n;
	int size;
	Asmbits *link;
};

static Asmbits *stash;

int
stashbits(bits n, int size)
{
	Asmbits **pb, *b;
	int i;

	assert(size == 4 || size == 8 || size == 16);
	for (pb=&stash, i=0; (b=*pb); pb=&b->link, i++)
		if (size <= b->size && b->n == n)
			return i;
	b = emalloc(sizeof *b);
	b->n = n;
	b->size = size;
	b->link = 0;
	*pb = b;
	return i;
}

static void
emitfin(FILE *f, char *sec[3])
{
	Asmbits *b;
	int lg, i;
	union { int32_t i; float f; } u;
	union { int64_t i; double f; } uu;

	if (!stash)
		return;
	fprintf(f, "/* floating point constants */\n");
	for (lg=4; lg>=2; lg--)
		for (b=stash, i=0; b; b=b->link, i++) {
			if (b->size == (1<<lg)) {
				fprintf(f,
					".section %s\n"
					".p2align %d\n"
					"%sfp%d:",
					sec[lg-2], lg, T.asloc, i
				);
				if (lg == 4)
					fprintf(f,
						"\n\t.quad %"PRId64
						"\n\t.quad 0\n\n",
						(int64_t)b->n);
				else if (lg == 3) {
					uu.i = b->n;
					fprintf(f,
						"\n\t.quad %"PRId64
						" /* %f */\n\n",
						uu.i, uu.f);
				} else if (lg == 2) {
					u.i = b->n;
					fprintf(f,
						"\n\t.int %"PRId32
						" /* %f */\n\n",
						u.i, (double)u.f);
				}
			}
		}
	while ((b=stash)) {
		stash = b->link;
		free(b);
	}
}

void
elf_emitfin(FILE *f)
{
	static char *sec[3] = { ".rodata", ".rodata", ".rodata" };

	emitfin(f ,sec);
	fprintf(f, ".section .note.GNU-stack,\"\",@progbits\n");
}

void
elf_emitfnfin(char *fn, FILE *f)
{
	fprintf(f, ".type %s, @function\n", fn);
	fprintf(f, ".size %s, .-%s\n", fn, fn);
}

void
macho_emitfin(FILE *f)
{
	static char *sec[3] = {
		"__TEXT,__literal4,4byte_literals",
		"__TEXT,__literal8,8byte_literals",
		"__TEXT,__literal16,16byte_literals",
	};

	emitfin(f, sec);
}

void
pe_emitfin(FILE *f)
{
	static char *sec[3] = { ".rodata", ".rodata", ".rodata" };

	emitfin(f, sec);
}

static uint32_t *file;
static uint nfile;
static uint curfile;

void
emitdbgfile(char *fn, FILE *f)
{
	uint32_t id;
	uint n;

	id = intern(fn);
	for (n=0; n<nfile; n++)
		if (file[n] == id) {
			/* gas requires positive
			 * file numbers */
			curfile = n + 1;
			return;
		}
	if (!file)
		file = vnew(0, sizeof *file, PHeap);
	vgrow(&file, ++nfile);
	file[nfile-1] = id;
	curfile = nfile;
	fprintf(f, ".file %u %s\n", curfile, fn);
}

void
emitdbgloc(uint line, uint col, FILE *f)
{
	if (col != 0)
		fprintf(f, "\t.loc %u %u %u\n", curfile, line, col);
	else
		fprintf(f, "\t.loc %u %u\n", curfile, line);
}
