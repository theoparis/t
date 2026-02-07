ENTRY(dyld_entry)
SECTIONS
{
    . = 0x200000;
    .text : ALIGN(4096) { 
        *(.text*) 
    }
    .rodata : ALIGN(4096) { 
        *(.rodata*) 
    }
    .data : ALIGN(4096) { 
        *(.data*) 
    }
    .bss : ALIGN(4096) { 
        *(.bss*) 
    }
}