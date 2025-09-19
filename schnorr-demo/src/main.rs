use num_bigint::{BigInt, RandBigInt};
use num_traits::{One, Zero};
use rand::Rng;
use rand::thread_rng;
use sha2::{Digest, Sha256};
use std::io::Write; // 用于构建哈希输入

// 这是一个简化的 Schnorr 零知识证明示例，演示了如何使用 Rust 实现基本的交互式证明过程。
fn iteractive_schnorr() {
    // 公开参数：素数p=204859, g=5, x=6 (秘密), h = 5^6 mod 204859 = 15625
    let p: BigInt = BigInt::from(204859u64);
    let g: BigInt = BigInt::from(5u32);
    let x: BigInt = BigInt::from(6u32); // 证明者的秘密
    let h = g.modpow(&x, &p); // h = g^x mod p

    // 进行多轮证明p
    for _ in 0..20 {
        // 证明者：生成承诺 t = g^k mod p
        let mut rng = thread_rng();
        let k = rng.gen_bigint_range(&BigInt::one(), &(&p - BigInt::one()));
        let t = g.modpow(&k, &p);
        println!("证明者发送 t: {}", t);

        // 验证者：生成挑战 c (简化到0..10)
        let c: BigInt = BigInt::from(rng.gen_range(0..10));
        println!("验证者挑战 c: {}", c);

        // 证明者：响应 r = k - c * x mod (p-1)
        let order = &p - BigInt::one(); // 阶
        let r = (&k - &c * &x).modpow(&BigInt::one(), &order); // 确保正数
        println!("证明者响应 r: {}", r);

        // 验证者：检查 g^r * h^c == t mod p
        let left = g.modpow(&r, &p) * h.modpow(&c, &p) % &p;
        if left == t {
            println!("验证通过！");
        } else {
            println!("验证失败！");
        }
    }
}

fn fiat_shamir() {
    // --- 公开参数 ---
    // 在真实世界, p 应该是至少2048位的安全素数
    let p: BigInt = BigInt::from(204859u64);
    let g: BigInt = BigInt::from(2u64);

    // Prover 的秘密 (只有 Prover 知道)
    let secret_x: BigInt = BigInt::from(123456u64);

    // Prover 的公钥 (所有人都知道)
    let public_h = g.modpow(&secret_x, &p);

    println!("--- 公开参数 ---");
    println!("p = {}", p);
    println!("g = {}", g);
    println!("h = g^x mod p = {}", public_h);
    println!("-------------------");

    // --- PROVER: 生成证明 ---
    println!("Prover 正在生成证明...");
    let mut rng = thread_rng();
    let order = &p - BigInt::one();

    // 1. 承诺: 随机选一个 k, 计算 t = g^k mod p
    let k = rng.gen_bigint_range(&BigInt::one(), &order);
    let t = g.modpow(&k, &p);

    // 2. 挑战 (Fiat-Shamir 的魔法在这里!):
    // 把公开信息和承诺 t 一起哈希，模拟一个无法预测的挑战 c
    let mut hasher = Sha256::new();
    hasher.write_all(&g.to_bytes_be().1).unwrap();
    hasher.write_all(&public_h.to_bytes_be().1).unwrap();
    hasher.write_all(&t.to_bytes_be().1).unwrap();
    let hash_bytes = hasher.finalize();
    let c = BigInt::from_bytes_be(num_bigint::Sign::Plus, &hash_bytes) % &order;

    // 3. 响应: 计算 r = k - c*x (mod order)
    let cx = (&c * &secret_x) % &order;
    let mut r = (&k - cx) % &order;
    if r < BigInt::zero() {
        r += &order;
    }

    println!("证明已生成: (r = {}, c = {})", r, c);
    println!("-------------------");

    // --- VERIFIER: 验证证明 ---
    println!("Verifier 正在验证证明...");
    // Verifier 为了验证, 需要自己重新计算 t' = g^r * h^c mod p
    let gr = g.modpow(&r, &p);
    let hc = public_h.modpow(&c, &p);
    let t_prime = (&gr * &hc) % &p;

    // Verifier 再用算出来的 t' 计算 c' = H(g || h || t')
    let mut hasher = Sha256::new();
    hasher.write_all(&g.to_bytes_be().1).unwrap();
    hasher.write_all(&public_h.to_bytes_be().1).unwrap();
    hasher.write_all(&t_prime.to_bytes_be().1).unwrap();
    let hash_bytes = hasher.finalize();
    let c_prime = BigInt::from_bytes_be(num_bigint::Sign::Plus, &hash_bytes) % &order;

    if c == c_prime {
        println!("✅ 验证通过！");
    } else {
        println!("❌ 验证失败！");
    }
}

fn main() {
    iteractive_schnorr();
    fiat_shamir();
}
