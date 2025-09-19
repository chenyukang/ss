use elf::ElfBytes;
use elf::abi::PT_LOAD;

// Define a sufficiently large memory space, e.g., 128MB
const MEM_SIZE: usize = 128 * 1024 * 1024;

pub struct CPU {
    x_registers: [u32; 32],
    pc: u32,
    memory: Vec<u8>,
    mem_base: u32,
}

impl CPU {
    pub fn new(code: Vec<u8>) -> Self {
        let mut cpu = CPU {
            x_registers: [0; 32],
            pc: 0,
            memory: code,
            mem_base: 0,
        };
        // Register x0 is always 0, so we don't need to handle it
        cpu.x_registers[0] = 0;
        cpu
    }

    pub fn new_from_elf(elf_data: &[u8]) -> Self {
        let mut memory = vec![0u8; MEM_SIZE];

        let elf = ElfBytes::<elf::endian::AnyEndian>::minimal_parse(elf_data)
            .expect("Failed to parse ELF file");

        // Get the program entry point
        let entry_point = elf.ehdr.e_entry as u32;

        // Iterate through program headers, load PT_LOAD type segments
        for segment in elf.segments().expect("Failed to get segments") {
            if segment.p_type == PT_LOAD {
                let virt_addr = segment.p_vaddr as usize;
                let file_size = segment.p_filesz as usize;
                let mem_size = segment.p_memsz as usize;
                let file_offset = segment.p_offset as usize;

                // Address translation: virtual address -> physical address
                let phys_addr = virt_addr - entry_point as usize;
                // Check memory boundaries
                if phys_addr + mem_size > MEM_SIZE {
                    panic!(
                        "Segment is too large for the allocated memory. vaddr: {:#x}, mem_size: {:#x}",
                        virt_addr, mem_size
                    );
                }

                // Copy data from ELF file to memory
                if file_size > 0 {
                    let segment_data = &elf_data[file_offset..file_offset + file_size];
                    memory[phys_addr..phys_addr + file_size].copy_from_slice(segment_data);
                }
            }
        }

        let mut cpu = CPU {
            x_registers: [0; 32],
            // Set directly to entry_point to match the linker script
            pc: entry_point,
            memory,
            mem_base: entry_point,
        };
        cpu.x_registers[0] = 0;
        cpu
    }

    pub fn run(&mut self) {
        loop {
            let physical_pc = (self.pc - self.mem_base) as usize;
            if physical_pc.saturating_add(1) >= self.memory.len() {
                break;
            }

            // Fetch the first 16 bits to determine instruction size
            let first_half =
                u16::from_le_bytes([self.memory[physical_pc], self.memory[physical_pc + 1]]);

            let new_pc: Option<u32>;
            let pc_increment: u32;

            // Check the lowest 2 bits to determine instruction length
            if first_half & 0x3 != 0x3 {
                // 16-bit compressed instruction
                pc_increment = 2;
                eprintln!(
                    "PC: {:#x} [16-bit] Instruction: {:#06x}",
                    self.pc, first_half
                );
                new_pc = self.execute_compressed_instruction(first_half);
            } else {
                // 32-bit instruction
                pc_increment = 4;
                if physical_pc.saturating_add(3) >= self.memory.len() {
                    break;
                }
                let second_half = u16::from_le_bytes([
                    self.memory[physical_pc + 2],
                    self.memory[physical_pc + 3],
                ]);
                let instruction = (second_half as u32) << 16 | (first_half as u32);

                if instruction == 0 {
                    break;
                }

                new_pc = self.execute_instruction(instruction);
            }

            if let Some(new_pc) = new_pc {
                if new_pc == 0 {
                    break;
                }
                self.pc = new_pc;
            } else {
                self.pc += pc_increment;
            }
        }
    }

    fn execute_compressed_instruction(&mut self, instruction: u16) -> Option<u32> {
        if instruction == 0 {
            panic!("Illegal compressed instruction 0x0 encountered.");
        }
        let opcode = instruction & 0x3; // Opcode is in the lowest 2 bits for C instructions
        let funct3 = (instruction >> 13) & 0x7;

        match (opcode, funct3) {
            // C.SW (Store Word)
            (0b00, 0b110) => {
                // Correct decoding for C.SW
                // | 15-13 | 12-10 | 9-7  | 6-5   | 4-2  | 1-0 |
                // | 110   | uimm  | rs1' | uimm  | rs2' | 00  |
                let rs2_prime = ((instruction >> 2) & 0x7) as usize + 8; // rs2' is in bits 4-2
                let rs1_prime = ((instruction >> 7) & 0x7) as usize + 8; // rs1' is in bits 9-7

                // Reconstruct the immediate byte offset for C.SW
                let imm_bit_6 = ((instruction >> 5) & 0x1) << 6;
                let imm_bits_5_3 = ((instruction >> 10) & 0x7) << 3;
                let imm_bit_2 = ((instruction >> 6) & 0x1) << 2;
                let imm = (imm_bit_6 | imm_bits_5_3 | imm_bit_2) as u32;

                let addr = self.x_registers[rs1_prime].wrapping_add(imm);
                let phys_addr = (addr - self.mem_base) as usize;
                let value = self.x_registers[rs2_prime];

                eprintln!(
                    "C.SW x{}, {}(x{}) -> addr {:#x}",
                    rs2_prime, imm, rs1_prime, addr
                );

                self.memory[phys_addr..phys_addr + 4].copy_from_slice(&value.to_le_bytes());
            }
            // C.LUI (loads > 2) / C.ADDI16SP (rd=2)
            (0b01, 0b011) => {
                let rd = ((instruction >> 7) & 0x1F) as usize;
                if rd == 2 {
                    // C.ADDI16SP
                    let imm = ((((instruction >> 12) & 1) as i32) << 9
                        | (((instruction >> 3) & 0x3) as i32) << 7
                        | (((instruction >> 5) & 1) as i32) << 6
                        | (((instruction >> 2) & 1) as i32) << 5
                        | (((instruction >> 6) & 1) as i32) << 4)
                        as u32;
                    let extended_imm = self.sign_extend_comp(imm, 9);
                    self.x_registers[rd] = self.x_registers[2].wrapping_add(extended_imm);
                    eprintln!("C.ADDI16SP sp, {}", extended_imm as i32);
                } else if rd != 0 {
                    // C.LUI
                    let imm = (((instruction >> 12) & 1) as u32) << 17
                        | (((instruction >> 2) & 0x1F) as u32) << 12;
                    let final_imm = self.sign_extend_comp(imm, 17);
                    self.x_registers[rd] = final_imm;
                    eprintln!("C.LUI x{}, {:#x}", rd, final_imm);
                }
            }
            // C.LI
            (0b01, 0b010) => {
                let rd = ((instruction >> 7) & 0x1F) as usize;
                if rd != 0 {
                    let imm = (((instruction >> 12) & 1) << 5 | ((instruction >> 2) & 0x1F)) as u32;
                    let extended_imm = self.sign_extend_comp(imm, 5);
                    self.x_registers[rd] = extended_imm;
                    eprintln!("C.LI x{}, {}", rd, extended_imm as i32);
                }
            }
            // C.JR / C.JALR / C.MV / C.ADD
            (0b10, 0b100) => {
                let rs1_rd = ((instruction >> 7) & 0x1F) as usize;
                let rs2 = ((instruction >> 2) & 0x1F) as usize;

                if ((instruction >> 12) & 1) == 0 {
                    // funct4=0b1000 -> C.JR or C.MV
                    if rs2 == 0 {
                        // C.JR
                        if rs1_rd == 0 {
                            panic!("C.JR with rs1=x0 is reserved.");
                        }
                        eprintln!("C.JR x{}", rs1_rd);
                        return Some(self.x_registers[rs1_rd]);
                    } else {
                        // C.MV
                        if rs1_rd != 0 {
                            eprintln!("C.MV x{}, x{}", rs1_rd, rs2);
                            self.x_registers[rs1_rd] = self.x_registers[rs2];
                        }
                    }
                } else {
                    // funct4=0b1001 -> C.JALR or C.ADD
                    if rs1_rd == 0 && rs2 == 0 {
                        // HINT, not standard
                    } else if rs1_rd != 0 && rs2 == 0 {
                        // C.JALR
                        eprintln!("C.JALR x{}", rs1_rd);
                        let target_address = self.x_registers[rs1_rd];
                        self.x_registers[1] = self.pc + 2; // Save return address in ra (x1)
                        return Some(target_address);
                    } else if rs1_rd != 0 && rs2 != 0 {
                        // C.ADD
                        eprintln!("C.ADD x{}, x{}", rs1_rd, rs2);
                        self.x_registers[rs1_rd] =
                            self.x_registers[rs1_rd].wrapping_add(self.x_registers[rs2]);
                    }
                }
            }
            // C.J
            (0b01, 0b101) => {
                // Correct bit reconstruction for C.J immediate
                // imm[11|4|9:8|10|6|7|3:1|5] from instruction bits [12|11|10:9|8|7|6|5:3|2]
                let imm = ((((instruction >> 12) & 1) as u32) << 11)  // imm[11]
                    | ((((instruction >> 11) & 1) as u32) << 4)       // imm[4]
                    | ((((instruction >> 9) & 0x3) as u32) << 8)      // imm[9:8]
                    | ((((instruction >> 8) & 1) as u32) << 10)       // imm[10]
                    | ((((instruction >> 7) & 1) as u32) << 6)        // imm[6]
                    | ((((instruction >> 6) & 1) as u32) << 7)        // imm[7]
                    | ((((instruction >> 3) & 0x7) as u32) << 1)      // imm[3:1]
                    | ((((instruction >> 2) & 1) as u32) << 5); // imm[5]
                let offset = self.sign_extend_comp(imm, 11);
                eprintln!("C.J offset {}", offset as i32);
                return Some(self.pc.wrapping_add(offset));
            }
            _ => panic!(
                "Unsupported compressed instruction: {:#06x} (opcode: {:#b}, funct3: {:#b})",
                instruction, opcode, funct3
            ),
        }
        None
    }

    fn execute_instruction(&mut self, instruction: u32) -> Option<u32> {
        // Extract the opcode
        let opcode = instruction & 0x7F;
        eprintln!("Instruction: {:#x}, opcode: {:#x}", instruction, opcode);
        // Dispatch to different execution functions based on opcode
        match opcode {
            0x63 => self.execute_b_type(instruction), // B-type (BEQ, BNE, etc.)
            0x67 => self.execute_jalr(instruction),   // JALR
            0x37 => self.execute_lui(instruction),    // LUI
            0x33 => self.execute_r_type(instruction), // R-type (ADD, SUB, etc.)
            0x13 => self.execute_i_type(instruction), // I-type (ADDI, etc.)
            // ... other instructions
            _ => panic!("Unsupported opcode: {:#x}", opcode),
        }
    }

    fn execute_b_type(&mut self, instruction: u32) -> Option<u32> {
        let (_, funct3, rs1, rs2, imm) = self.decode_b_type(instruction);
        let offset = self.sign_extend(imm, 12);

        let rs1_val = self.x_registers[rs1 as usize];
        let rs2_val = self.x_registers[rs2 as usize];

        let mut taken = false;
        match funct3 {
            0x0 => {
                // BEQ
                eprintln!("BEQ x{}, x{}, offset {}", rs1, rs2, offset as i32);
                if rs1_val == rs2_val {
                    taken = true;
                }
            }
            0x1 => {
                // BNE
                eprintln!("BNE x{}, x{}, offset {}", rs1, rs2, offset as i32);
                if rs1_val != rs2_val {
                    taken = true;
                }
            }
            0x6 => {
                // BLTU
                eprintln!("BLTU x{}, x{}, offset {}", rs1, rs2, offset as i32);
                if rs1_val < rs2_val {
                    taken = true;
                }
            }
            // ... other B-type instructions
            _ => panic!("Unsupported B-type instruction funct3: {:#x}", funct3),
        }

        if taken {
            Some(self.pc.wrapping_add(offset))
        } else {
            None // PC will increment by 4 normally
        }
    }

    fn execute_r_type(&mut self, instruction: u32) -> Option<u32> {
        let (_, rd, funct3, rs1, rs2, funct7) = self.decode_r_type(instruction);

        // Determine the specific operation based on funct3 and funct7
        match (funct3, funct7) {
            (0x0, 0x0) => {
                // ADD
                eprintln!("ADD x{}, x{}, x{}", rd, rs1, rs2);
                self.x_registers[rd as usize] =
                    self.x_registers[rs1 as usize].wrapping_add(self.x_registers[rs2 as usize]);
                eprintln!("reg[{}]={}", rd, self.x_registers[rd as usize]);
            }
            (0x0, 0x20) => {
                // SUB
                self.x_registers[rd as usize] =
                    self.x_registers[rs1 as usize].wrapping_sub(self.x_registers[rs2 as usize]);
            }
            // ... other R-type instructions ...
            _ => panic!("Unsupported R-type instruction: {:#x}", instruction),
        }
        None
    }

    fn execute_i_type(&mut self, instruction: u32) -> Option<u32> {
        let (_, rd, funct3, rs1, imm) = self.decode_i_type(instruction);

        match funct3 {
            0x0 => {
                // ADDI
                let immediate = self.sign_extend(imm, 12);
                eprintln!("ADDI x{}, x{}, {}", rd, rs1, immediate as i32);
                self.x_registers[rd as usize] =
                    self.x_registers[rs1 as usize].wrapping_add(immediate);
                eprintln!("reg[{}]={}", rd, self.x_registers[rd as usize]);
            }
            0x1 => {
                // SLLI (Shift Left Logical Immediate)
                // For RV32I, the shift amount is the lower 5 bits of the immediate.
                let shamt = imm & 0x1F;
                eprintln!("SLLI x{}, x{}, {}", rd, rs1, shamt);
                self.x_registers[rd as usize] = self.x_registers[rs1 as usize] << shamt;
                eprintln!("reg[{}]={}", rd, self.x_registers[rd as usize]);
            }
            // ... other I-type instructions ...
            _ => panic!("Unsupported I-type instruction: {:#x}", instruction),
        }
        None
    }

    fn execute_lui(&mut self, instruction: u32) -> Option<u32> {
        let (_, rd, imm) = self.decode_u_type(instruction);
        // LUI places the immediate in the upper 20 bits of the register,
        // and clears the lower 12 bits. The immediate is already aligned in the instruction.
        self.x_registers[rd as usize] = imm;
        eprintln!("LUI x{}, {:#x}", rd, imm);
        eprintln!("reg[{}]={}", rd, self.x_registers[rd as usize]);
        None
    }

    fn execute_jalr(&mut self, instruction: u32) -> Option<u32> {
        let (_, rd, _, rs1, imm) = self.decode_i_type(instruction);
        let next_pc = self.pc + 4;

        // JALR calculates the target address by adding the sign-extended immediate to rs1
        let target_address = self.x_registers[rs1 as usize].wrapping_add(self.sign_extend(imm, 12));

        // Save the address of the next instruction (pc + 4) to rd
        if rd != 0 {
            self.x_registers[rd as usize] = next_pc;
        }

        eprintln!(
            "JALR x{}, x{}, {:#x} -> new pc: {:#x}",
            rd,
            rs1,
            self.sign_extend(imm, 12) as i32,
            target_address
        );

        // Return the new PC value to jump
        Some(target_address)
    }

    // Helper function: sign extend immediate values
    fn sign_extend(&self, value: u32, imm_len: u32) -> u32 {
        let sign_bit = 1 << (imm_len - 1);
        if (value & sign_bit) != 0 {
            value | (0xFFFFFFFF << imm_len)
        } else {
            value
        }
    }

    fn sign_extend_comp(&self, value: u32, imm_len: u32) -> u32 {
        let sign_bit = 1 << (imm_len - 1);
        if (value & sign_bit) != 0 {
            // The value is negative, so we need to extend the sign bit.
            // For an imm_len bit number, this means filling the bits from imm_len to 31 with 1s.
            value | (0xFFFFFFFF << imm_len)
        } else {
            value
        }
    }

    fn decode_b_type(&self, instruction: u32) -> (u32, u32, u32, u32, u32) {
        let opcode = instruction & 0x7F;
        let funct3 = (instruction >> 12) & 0x07;
        let rs1 = (instruction >> 15) & 0x1F;
        let rs2 = (instruction >> 20) & 0x1F;

        // Reconstruct the 12-bit immediate
        let imm11 = (instruction >> 7) & 0x1;
        let imm4_1 = (instruction >> 8) & 0xF;
        let imm10_5 = (instruction >> 25) & 0x3F;
        let imm12 = (instruction >> 31) & 0x1;

        let imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
        (opcode, funct3, rs1, rs2, imm)
    }

    fn decode_r_type(&self, instruction: u32) -> (u32, u32, u32, u32, u32, u32) {
        let opcode = instruction & 0x7F;
        let rd = (instruction >> 7) & 0x1F;
        let funct3 = (instruction >> 12) & 0x07;
        let rs1 = (instruction >> 15) & 0x1F;
        let rs2 = (instruction >> 20) & 0x1F;
        let funct7 = (instruction >> 25) & 0x7F;
        (opcode, rd, funct3, rs1, rs2, funct7)
    }

    fn decode_i_type(&self, instruction: u32) -> (u32, u32, u32, u32, u32) {
        let opcode = instruction & 0x7F;
        let rd = (instruction >> 7) & 0x1F;
        let funct3 = (instruction >> 12) & 0x07;
        let rs1 = (instruction >> 15) & 0x1F;
        // Immediate value occupies 12 bits, starting from bit 20
        let imm = (instruction >> 20) & 0xFFF;
        (opcode, rd, funct3, rs1, imm)
    }

    fn decode_u_type(&self, instruction: u32) -> (u32, u32, u32) {
        let opcode = instruction & 0x7F;
        let rd = (instruction >> 7) & 0x1F;
        // U-type immediate is in bits 31:12
        let imm = instruction & 0xFFFFF000;
        (opcode, rd, imm)
    }

    #[cfg(test)]
    fn get_val_at_addr(&self, addr: u32) -> u32 {
        let phys_addr = (addr - self.mem_base) as usize;
        let bytes = &self.memory[phys_addr..phys_addr + 4];
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

#[test]
fn run_from_elf() {
    use std::fs::File;
    use std::io::Read;

    // 1. Read the ELF file
    let mut file = File::open("./demo/target/riscv32imac-unknown-none-elf/release/demo")
        .expect("Failed to open the RISC-V executable file");
    let mut elf_data = Vec::new();
    file.read_to_end(&mut elf_data)
        .expect("Failed to read the file");

    // 2. Create CPU instance
    let mut cpu = CPU::new_from_elf(&elf_data);

    // 3. Run the virtual machine
    cpu.run();

    // 4. Output register state
    for i in 0..32 {
        println!("x{}: {:#x}", i, cpu.x_registers[i]);
    }

    // 5. Verify results
    const RESULT_VIRT_ADDR: u32 = 0x1000;
    let val = cpu.get_val_at_addr(0x1000);
    eprintln!("Value at {:#x}: {}", RESULT_VIRT_ADDR, val);
    const EXPECTED_RESULT: u32 = 55; // 1 + 2 + ... + 10
    assert_eq!(val, EXPECTED_RESULT);
}

#[test]
fn run_basic_test() {
    // let a: u32 = 1; let b: u32 = 2; let c = a + b;
    // ADDI x5, x0, 1   (a = 1)
    // ADDI x6, x0, 2   (b = 2)
    // ADD  x7, x5, x6  (c = a + b)
    let program: Vec<u8> = vec![
        0x93, 0x02, 0x10, 0x00, // ADDI x5, x0, 1
        0x13, 0x03, 0x20, 0x00, // ADDI x6, x0, 2
        0xb3, 0x83, 0x62, 0x00, // ADD  x7, x5, x6
    ];

    let mut cpu = CPU::new(program);
    cpu.run();

    // Verify results
    assert_eq!(cpu.x_registers[5], 1);
    assert_eq!(cpu.x_registers[6], 2);
    eprintln!("reg[7]={}", cpu.x_registers[7]);
    assert_eq!(cpu.x_registers[7], 3);
}

fn main() {}
