#include "../all.h"

typedef struct I386Op I386Op;

enum I386Reg {
	EAX = RXX+1, /* caller-save */
	ECX,
	EDX,

	EBX, /* callee-save */
	ESI,
	EDI,

	EBP, /* globally live */
	ESP,

	NGPR = ESP - EAX + 1,
	NGPS = EDX - EAX + 1,
	NCLR = EDI - EBX + 1,
};
MAKESURE(i386_reg_not_tmp, ESP < (int)Tmp0);

struct I386Op {
	char nmem;
	char zflag;
	char lflag;
};

/* targ.c */
extern I386Op i386_op[];

/* abi.c */
extern int i386_rsave[];
extern int i386_rclob[];
bits i386_retregs(Ref, int[2]);
bits i386_argregs(Ref, int[2]);
void i386_abi0(Fn *);
void i386_abi(Fn *);

/* isel.c */
void i386_isel(Fn *);

/* emit.c */
void i386_emitfn(Fn *, FILE *);
