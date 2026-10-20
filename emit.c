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
   catch types treats a call site with action 0 as a cleanup. The landing pads use DW_EH_PE_* offsets from the function. */
static uint ehnext, ehcur, ehcs;   /* numbers of the functions with a pad, of the current one, and of its call sites */
static uint *ehcsv;                /* the current function's call sites */
static uint nehcsv, ehcsvcap;

void
ehfnbegin(Fn *fn, FILE *f)
{
	if (!fn->eh_lp)
		return;
	if (T.apple)
		die("landing pads need ELF unwind tables (no Apple target)");
	ehcur = ++ehnext;
	nehcsv = 0;
	fprintf(f,
		".LEHfs%u:\n"
		"\t.cfi_personality 0x9b, DW.ref.__gcc_personality_v0\n"
		"\t.cfi_lsda 0x1b, .LEHL%u\n",
		ehcur, ehcur
	);
}

void
ehlpbegin(Fn *fn, Blk *b, FILE *f)
{
	if (b == fn->eh_lp)
		fprintf(f, ".LEHlp%u:\n", ehcur);
}

/* a call of the function, outside its landing pad: its address range is a call site of the LSDA */
void
ehcsbegin(Fn *fn, Blk *b, FILE *f)
{
	if (!fn->eh_lp || b == fn->eh_lp)
		return;
	if (nehcsv == ehcsvcap) {
		ehcsvcap = ehcsvcap ? 2 * ehcsvcap : 16;
		ehcsv = realloc(ehcsv, ehcsvcap * sizeof ehcsv[0]);
		if (!ehcsv)
			die("out of memory");
	}
	ehcsv[nehcsv++] = ++ehcs;
	fprintf(f, ".LEHcsb%u:\n", ehcs);
}

void
ehcsend(Fn *fn, Blk *b, FILE *f)
{
	if (!fn->eh_lp || b == fn->eh_lp)
		return;
	fprintf(f, ".LEHcse%u:\n", ehcsv[nehcsv-1]);
}

void
ehfnend(Fn *fn, FILE *f)
{
	static int persdef;   /* the personality's DW.ref, defined once per file (as gcc does) */
	uint k, c;

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
			c, ehcur, c, c, ehcur, ehcur
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
