use derive_more::Error;
use derive_more::From;
use derive_more::derive::Display;
use devicetree::decryptor::DecryptWriter;
use devicetree::img3::Img3Error;
use devicetree::img3::Img3Stream;
use rootcause::Report;
use rootcause::prelude::ResultExt as _;
use rootcause::report;
use std::fs::File;
use std::io::Write;

mod cpu;
mod decoder;
// mod explorer;
mod hardware;
mod jit;

use cpu::ArmCpu;
use hardware::Hardware;

extern "C" fn jit_read_helper(cpu_ptr: *mut ArmCpu, addr: u32) -> u32 {
    let cpu = unsafe { &mut *cpu_ptr };
    cpu.read_memory(addr).unwrap_or(0)
}

extern "C" fn jit_write_helper(cpu_ptr: *mut ArmCpu, addr: u32, val: u32) {
    let cpu = unsafe { &mut *cpu_ptr };
    let _ = cpu.write_memory(addr, val);
}

#[derive(Debug, Display, Error, From)]
pub enum EmulatorError {
    #[display("I/O error: {source}")]
    Io { source: std::io::Error },
    #[display("IMG3 error: {source}")]
    Img3 { source: Img3Error },
    #[display("CPU error: {source}")]
    Cpu { source: cpu::CpuError },
}

fn main() -> Result<(), Report> {
    println!("iOS 5 iBEC Emulator");
    println!("==================");

    let ibec_path = "work/Firmware/dfu/iBEC.k48ap.RELEASE.dfu";
    let mut ibec_file = File::open(ibec_path).attach(format!("File not found: {}", ibec_path))?;

    // Parse as IMG3 file
    let mut img3 = Img3Stream::new(&mut ibec_file)?;

    // Use provided IV and key
    let iv = hex::decode("bde7b0d5cf7861479d81eb23f99d2e9e").unwrap();
    let key =
        hex::decode("1ba1f38e6a5b4841c1716c11acae9ee0fb471e50362a3b0dd8d98019f174a2f2").unwrap();

    // Decrypt payload
    let mut decrypted_file = std::fs::File::create("work/Firmware/dfu/iBEC_decrypted.bin").unwrap();
    let mut decrypt_writer = DecryptWriter::new(decrypted_file, &key, &iv);
    img3.stream_data_section(&mut decrypt_writer)?;
    println!("Successfully decrypted payload");

    let decrypted_payload = std::fs::read("work/Firmware/dfu/iBEC_decrypted.bin").unwrap();

    // Hexdump start of payload
    for i in (0..(0x400.min(decrypted_payload.len()))).step_by(16) {
        print!("{:08x}: ", 0x80000000 + i as u32);
        for j in 0..16 {
            if i + j < decrypted_payload.len() {
                print!("{:02x} ", decrypted_payload[i + j]);
            } else {
                print!("   ");
            }
        }
        print!(" |");
        for j in 0..16 {
            if i + j < decrypted_payload.len() {
                let c = decrypted_payload[i + j];
                if c >= 0x20 && c <= 0x7E {
                    print!("{}", c as char);
                } else {
                    print!(".");
                }
            }
        }
        println!("|");
    }

    // Check memory around PC+8+24 = 0x80000000+8+24 = 0x80000020
    let check_addr = 0x80000020u32 as i32;
    println!(
        "Memory at 0x{:08x}: {:02x?}",
        check_addr,
        &decrypted_payload[0x20..0x24]
    );

    // Check memory at 0x108 where LDR will actually read from
    println!(
        "Memory at 0x80000108: {:02x?}",
        &decrypted_payload[0x108..0x10c]
    );

    // Check instruction at 0x8000000c (the branch)
    let branch_offset = 0x0c;
    println!(
        "Branch instruction at 0x8000000c: {:02x?}",
        &decrypted_payload[branch_offset..branch_offset + 4]
    );

    // Check the LDR instruction at 0x80000024
    let ldr_offset = 0x24;
    println!(
        "LDR instruction at 0x80000024: {:02x?}",
        &decrypted_payload[ldr_offset..ldr_offset + 4]
    );

    // Check memory at relocation entry point (0x9c8)
    println!("Relocated code area hexdump (starting at 0x9c8):");
    for i in (0x9c8..(0x9c8 + 0x40.min(decrypted_payload.len() - 0x9c8))).step_by(16) {
        print!("{:08x}: ", 0x80000000 + i as u32);
        for j in 0..16 {
            print!("{:02x} ", decrypted_payload[i + j]);
        }
        println!();
    }

    println!("Literal pool dump at 0x80000300:");
    for i in (0x300..0x340).step_by(16) {
        print!("{:08x}: ", 0x80000000 + i as u32);
        for j in 0..16 {
            print!("{:02x} ", decrypted_payload[i + j]);
        }
        println!();
    }

    // Run CPU emulator
    let mut cpu = ArmCpu::new();
    let hardware = Hardware::new();
    cpu.load_memory(0x80000000, &decrypted_payload);
    cpu.set_hardware(hardware);
    cpu.pc = 0x80000000;

    // Initialize some registers with reasonable values
    cpu.registers[13] = 0x80010000; // Stack pointer
    cpu.registers[14] = 0x80000004; // Link register

    // Simulate NVRAM initialization for iBEC
    cpu.registers[0] = 0x84000000; // NVRAM base
    cpu.registers[1] = 0x85000000; // Boot args
    cpu.registers[2] = 0x86000000; // Kernel args

    println!("Running CPU emulation...");
    let mut step = 0;
    let mut decoded_cache: std::collections::HashMap<u32, decoder::Instruction> =
        std::collections::HashMap::new();

    let mut jit = jit::Jit::new();
    let mut block_sizes: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut last_watch_val = 0;
    let mut compilations = 0;

    while step < 10_000_000_000 {
        let pc = cpu.pc;

        // 1. Try JIT
        if let Some(code_ptr) = jit.get_block(pc) {
            let func: extern "C" fn(
                *mut ArmCpu,
                *mut u32,
                *mut u8,
                *mut u32,
                extern "C" fn(*mut ArmCpu, u32) -> u32,
                extern "C" fn(*mut ArmCpu, u32, u32),
            ) = unsafe { std::mem::transmute(code_ptr) };
            func(
                &mut cpu,
                cpu.registers.as_mut_ptr(),
                cpu.ram.as_mut_ptr(),
                &mut cpu.cpsr,
                jit_read_helper,
                jit_write_helper,
            );
            cpu.pc = cpu.registers[15];
            let size = *block_sizes.get(&pc).unwrap_or(&1);
            step += size;
            if step % 1000000 < size {
                eprintln!(
                    "[Step {}] PC=0x{:08x}, JIT cache: {}",
                    step, cpu.pc, compilations
                );
                eprintln!(
                    "Registers: R0={:08x} R1={:08x} R2={:08x} R3={:08x} SP={:08x} LR={:08x} CPSR={:08x}",
                    cpu.registers[0],
                    cpu.registers[1],
                    cpu.registers[2],
                    cpu.registers[3],
                    cpu.registers[13],
                    cpu.registers[14],
                    cpu.cpsr
                );
            }
            continue;
        }

        // 2. Block discovery and compilation if not in cache
        let mut block_insns = Vec::new();
        let mut curr_pc = pc;
        for _ in 0..100 {
            // Ensure instruction is decoded
            if !decoded_cache.contains_key(&curr_pc) {
                let mut insn_bytes = [0u8; 4];
                for i in 0..4 {
                    insn_bytes[i] = cpu.memory.get(&(curr_pc + i as u32)).copied().unwrap_or(0);
                }
                let is_thumb = (cpu.cpsr >> 5) & 1 != 0;
                if let Ok(insns) = decoder::decode(&insn_bytes, is_thumb) {
                    if !insns.is_empty() {
                        decoded_cache.insert(curr_pc, insns[0].clone());
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let insn = decoded_cache.get(&curr_pc).unwrap().clone();
            block_insns.push((curr_pc, insn.clone()));

            // Basic block terminator check
            let m = insn.mnemonic.as_str();
            if m == "b"
                || m == "bl"
                || m == "bx"
                || m == "blx"
                || m == "cbz"
                || m == "cbnz"
                || m == "it"
                || m == "svc"
                || m == "eret"
                || m == "pop"
                || m == "pop.w"
                || insn
                    .operands
                    .iter()
                    .any(|o| matches!(o, decoder::Operand::Register(15)))
            {
                break;
            }
            curr_pc += insn.size as u32;
        }

        let is_thumb = (cpu.cpsr >> 5) & 1 != 0;
        if let Some(code_ptr) = jit.compile_block(pc, &block_insns, is_thumb) {
            let func: extern "C" fn(
                *mut ArmCpu,
                *mut u32,
                *mut u8,
                *mut u32,
                extern "C" fn(*mut ArmCpu, u32) -> u32,
                extern "C" fn(*mut ArmCpu, u32, u32),
            ) = unsafe { std::mem::transmute(code_ptr) };
            func(
                &mut cpu,
                cpu.registers.as_mut_ptr(),
                cpu.ram.as_mut_ptr(),
                &mut cpu.cpsr,
                jit_read_helper,
                jit_write_helper,
            );
            cpu.pc = cpu.registers[15];
            block_sizes.insert(pc, block_insns.len() as u64);
            step += block_insns.len() as u64;
            compilations += 1;
            continue;
        }

        // 3. Fallback to interpreter
        if let Some(insn) = decoded_cache.get(&pc) {
            let insn = insn.clone();
            let old_pc = cpu.pc;
            cpu.pc_modified = false;
            if let Err(e) = cpu.execute(&insn) {
                println!("    Error at PC 0x{:08x}: {}", old_pc, e);
                break;
            }
            if !cpu.pc_modified {
                cpu.pc += insn.size as u32;
            }
            step += 1;
        } else {
            println!("  Failed to decode at 0x{:08x}", pc);
            break;
        }

        // Minimal UART hook via memory watch
        if let Ok(v0) = cpu.read_memory(0x5FFF7F24) {
            if v0 != last_watch_val && v0 != 0 {
                if v0 >= 0x20 && v0 <= 0x7E {
                    print!("{}", v0 as u8 as char);
                } else if v0 == 0x0a || v0 == 0x0d {
                    if v0 == 0x0a {
                        println!();
                    }
                }
                let _ = std::io::stdout().flush();
                last_watch_val = v0;
            }
        }

        if step % 10_000 == 0 {
            eprintln!(
                "[Step {}] PC=0x{:08x}, JIT cache: {}",
                step, cpu.pc, compilations
            );
        }
    }

    println!("Final CPU state:");
    cpu.dump_registers();
    Ok(())
}
