/// RISC-V 32-bit instruction formats
#[derive(Debug)]
enum Instruction {
    IType {
        rd: usize,
        rs1: usize,
        imm: i32,
        funct3: u32,
    },
    RType {
        rd: usize,
        rs1: usize,
        rs2: usize,
        funct3: u32,
        funct7: u32,
    },
}

/// RISC-V virtual machine vm
pub struct VM {
    x_registers: [u32; 32],
    pc: u32,
    memory: Vec<u8>,
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl VM {
    /// Create a new vm instance with initialized registers
    pub fn new() -> Self {
        Self {
            x_registers: [0; 32],
            pc: 0,
            memory: Vec::new(),
        }
    }

    /// Load program bytecode into memory and reset PC
    pub fn load_program(&mut self, program: Vec<u8>) {
        self.memory = program;
        self.pc = 0;
    }

    /// Fetch 32-bit instruction from memory at current PC
    fn fetch_instruction(&self) -> Option<u32> {
        let pc = self.pc as usize;
        if pc + 4 > self.memory.len() {
            return None;
        }

        // RISC-V uses little-endian byte order
        let instruction = u32::from_le_bytes([
            self.memory[pc],
            self.memory[pc + 1],
            self.memory[pc + 2],
            self.memory[pc + 3],
        ]);

        Some(instruction)
    }

    /// Decode 32-bit instruction into structured format
    fn decode(&self, code: u32) -> Option<Instruction> {
        let opcode = code & 0x7f;

        match opcode {
            0x13 => {
                // I-type instruction (ADDI, etc.)
                let rd = ((code >> 7) & 0x1f) as usize;
                let rs1 = ((code >> 15) & 0x1f) as usize;
                let funct3 = (code >> 12) & 0x7;
                let imm = (code as i32) >> 20; // Sign-extended

                Some(Instruction::IType {
                    rd,
                    rs1,
                    imm,
                    funct3,
                })
            }
            0x33 => {
                // R-type instruction (ADD, SUB, etc.)
                let rd = ((code >> 7) & 0x1f) as usize;
                let rs1 = ((code >> 15) & 0x1f) as usize;
                let rs2 = ((code >> 20) & 0x1f) as usize;
                let funct3 = (code >> 12) & 0x7;
                let funct7 = (code >> 25) & 0x7f;

                Some(Instruction::RType {
                    rd,
                    rs1,
                    rs2,
                    funct3,
                    funct7,
                })
            }
            _ => None, // Unsupported opcode
        }
    }

    /// Execute decoded instruction
    fn execute(&mut self, instruction_type: Instruction) -> Result<String, String> {
        match instruction_type {
            Instruction::IType {
                rd,
                rs1,
                imm,
                funct3,
            } => {
                match funct3 {
                    0x0 => {
                        // ADDI - Add immediate
                        self.write_register(rd, self.x_registers[rs1] + imm as u32);
                        Ok(format!(
                            "ADDI x{}, x{}, {} -> x{} = {}",
                            rd, rs1, imm, rd, self.x_registers[rd]
                        ))
                    }
                    _ => Err(format!("Unsupported I-type funct3: {:#x}", funct3)),
                }
            }
            Instruction::RType {
                rd,
                rs1,
                rs2,
                funct3,
                funct7,
            } => {
                match (funct3, funct7) {
                    (0x0, 0x00) => {
                        // ADD - Add registers
                        let result = self.x_registers[rs1] + self.x_registers[rs2];
                        self.write_register(rd, result);
                        Ok(format!(
                            "ADD x{}, x{}, x{} -> x{} = {}",
                            rd, rs1, rs2, rd, self.x_registers[rd]
                        ))
                    }
                    (0x0, 0x20) => {
                        // SUB - Subtract registers
                        let result = self.x_registers[rs1] - self.x_registers[rs2];
                        self.write_register(rd, result);
                        Ok(format!(
                            "SUB x{}, x{}, x{} -> x{} = {}",
                            rd, rs1, rs2, rd, self.x_registers[rd]
                        ))
                    }
                    _ => Err(format!(
                        "Unsupported R-type instruction: funct3={:#x}, funct7={:#x}",
                        funct3, funct7
                    )),
                }
            }
        }
    }

    /// Write to register, ensuring x0 remains zero
    fn write_register(&mut self, reg: usize, value: u32) {
        if reg != 0 && reg < 32 {
            self.x_registers[reg] = value;
        }
    }

    /// Execute the loaded program
    pub fn run(&mut self) {
        println!("🚀 Starting RISC-V VM execution...\n");

        loop {
            match self.fetch_instruction() {
                Some(instruction) => {
                    println!("PC: {:#x}, Instruction: {:#010x}", self.pc, instruction);

                    match self.decode(instruction) {
                        Some(decoded) => match self.execute(decoded) {
                            Ok(msg) => println!("{}", msg),
                            Err(err) => {
                                println!("❌ Execution error: {}", err);
                                break;
                            }
                        },
                        None => {
                            println!("❌ Unsupported instruction: {:#010x}", instruction);
                            break;
                        }
                    }

                    self.pc += 4;
                    println!();
                }
                None => {
                    println!("📝 End of program reached");
                    break;
                }
            }
        }

        println!("✅ Execution completed!");
        self.print_registers();
    }

    /// Display current register states (only non-zero values)
    pub fn print_registers(&self) {
        println!("\n📊 Register state:");
        let mut printed_any = false;

        for (i, &value) in self.x_registers.iter().enumerate() {
            if value != 0 {
                println!("  x{}: {}", i, value);
                printed_any = true;
            }
        }

        if !printed_any {
            println!("  All registers are zero");
        }
    }
}

/// Test program: equivalent to `let a = 1; let b = 2; let c = a + b;`
fn create_test_program_basic() -> Vec<u8> {
    vec![
        0x93, 0x02, 0x10, 0x00, // ADDI x5, x0, 1   (a = 1)
        0x13, 0x03, 0x20, 0x00, // ADDI x6, x0, 2   (b = 2)
        0xb3, 0x83, 0x62, 0x00, // ADD  x7, x5, x6  (c = a + b)
    ]
}

/// Advanced test program: demonstrates more operations
fn create_test_program_advanced() -> Vec<u8> {
    vec![
        0x93, 0x02, 0xa0, 0x00, // ADDI x5, x0, 10   (a = 10)
        0x13, 0x03, 0x50, 0x00, // ADDI x6, x0, 5    (b = 5)
        0xb3, 0x83, 0x62, 0x00, // ADD  x7, x5, x6   (c = a + b = 15)
        0x33, 0x84, 0x62, 0x40, // SUB  x8, x5, x6   (d = a - b = 5)
    ]
}

fn run_demo(name: &str, description: &str, program: Vec<u8>) {
    println!("🔧 RISC-V Virtual Machine Demo: {}", name);
    println!("===============================");
    println!("Program: {}\n", description);

    let mut vm = VM::new();
    vm.load_program(program);
    vm.run();
    println!();
}

fn main() {
    // Basic demo
    run_demo(
        "Basic Operations",
        "let a = 1; let b = 2; let c = a + b;",
        create_test_program_basic(),
    );

    // Advanced demo
    run_demo(
        "Advanced Operations",
        "let a = 10; let b = 5; let c = a + b; let d = a - b;",
        create_test_program_advanced(),
    );
}
