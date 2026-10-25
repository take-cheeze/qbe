/* the boot sector, then the code it loads */
SECTIONS {
	. = 0x7c00;
	.boot : { *(.boot) }
	. = 0x7dfe;
	.sig : { *(.bootsig) }
	.text : { *(.text*) *(.rodata*) *(.data*) *(.bss*) *(COMMON) }
	/DISCARD/ : { *(.note*) *(.comment) *(.eh_frame) }
}
