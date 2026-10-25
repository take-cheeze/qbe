/* A boot sector for the -m16 tests (tools/test-i386.sh, under
 * qemu-system-i386): in real mode, it loads the next sectors of
 * the floppy after itself, calls test(), writes the 4 bytes of its
 * result to the first serial port and exits qemu through its
 * isa-debug-exit device. */
	.code16gcc
	.section .boot, "ax"
	.globl _start
_start:
	cli
	xorw %ax, %ax
	movw %ax, %ds
	movw %ax, %es
	movw %ax, %ss
	movl $0x7c00, %esp
	/* int 13h: read 16 sectors (cylinder 0, head 0, from
	 * sector 2) to 0x7e00 */
	movw $0x0210, %ax
	movw $0x0002, %cx
	movb $0, %dh
	movw $0x7e00, %bx
	int $0x13
	calll test
	movl %eax, %ebx
	movw $0x3f8, %dx
	movb %bl, %al
	outb %al, %dx
	movb %bh, %al
	outb %al, %dx
	shrl $16, %ebx
	movb %bl, %al
	outb %al, %dx
	movb %bh, %al
	outb %al, %dx
	movb $0, %al
	outb %al, $0xf4
	hlt
	.section .bootsig, "a"
	.byte 0x55, 0xaa
