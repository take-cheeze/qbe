#include "all.h"

I386Op i386_op[NOp] = {
#define O(op, t, x) [O##op] =
#define X(nm, zf, lf) { nm, zf, lf, },
	#include "../ops.h"
};

static int
i386_memargs(int op)
{
	return i386_op[op].nmem;
}

/* i386, System V ELF (Linux): 32-bit pointers of class w, no
 * floating point and no l class yet (i386_abi0 refuses them),
 * non-PIC code; no conditional moves (-march=i386 has no cmov)
 */
#define I386_COMMON \
	.gpr0 = EAX, \
	.ngpr = NGPR, \
	.fpr0 = ESP + 1, \
	.nfpr = 0, \
	.rglob = BIT(EBP) | BIT(ESP), \
	.nrglob = 2, \
	.rsave = i386_rsave, \
	.nrsave = {NGPS, 0}, \
	.retregs = i386_retregs, \
	.argregs = i386_argregs, \
	.memargs = i386_memargs, \
	.abi0 = i386_abi0, \
	.abi1 = i386_abi, \
	.isel = i386_isel, \
	.emitfn = i386_emitfn, \
	.emitfin = elf_emitfin, \
	.asloc = ".L", \
	.ptr32 = 1, \
	.cansel = 0,

/* the cdecl convention: every argument on the stack */
Target T_i386 = {
	.name = "i386",
	I386_COMMON
};

/* gcc's -mregparm=3, as the Linux kernel builds i386 and
 * real-mode code: up to three words of arguments in eax,
 * edx, ecx (none for a variadic function)
 */
Target T_i386_regparm3 = {
	.name = "i386_regparm3",
	.regparm = 3,
	I386_COMMON
};
