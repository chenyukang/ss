// 禁用标准库，因为我们没有操作系统
#![no_std]
// 禁用 main 函数的入口点，因为我们的程序没有操作系统来调用 main
#![no_main]

use core::panic::PanicInfo;

#[unsafe(no_mangle)]
pub extern "C" fn _start() {
    let mut sum = 0;
    for i in 1..=10 {
        sum += i;
    }

    // Store the result (which should be 55) in a known memory location.
    let result_ptr = 0x1000 as *mut u32;
    unsafe {
        *result_ptr = sum;
    }
}

// 定义一个恐慌处理函数，这是裸机程序所必需的
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
