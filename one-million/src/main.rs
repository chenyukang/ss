use num_bigint::{BigInt, RandBigInt};
use num_traits::{One, Zero};
use rand::{Rng, thread_rng};

// 财富范围：1..=MAX_WEALTH
const MAX_WEALTH: u64 = 100;
const PRIME_P: u32 = 2_147_483_647; // Mersenne Prime 2^31-1

/// 模拟 Alice (富翁 A)
struct Alice {
    wealth: u64,
    n: BigInt,       // 公钥 N
    e: BigInt,       // 公钥 e
    d: BigInt,       // 私钥 d
    prime_p: BigInt, // 用于最后结果取模的大素数，防止 Bob 推导
}

/// 模拟 Bob (富翁 B)
struct Bob {
    wealth: u64,
    secret_x: BigInt, // Bob 的随机秘密数字
}

impl Alice {
    fn new(wealth: u64) -> Self {
        // Demo 用的“固定”RSA 参数：为了代码可读性与运行稳定性，直接用两素数构造 N。
        let p = BigInt::from(104729u32); // 第 10000 个素数
        let q = BigInt::from(104723u32); // 第 9999 个素数
        let n = &p * &q;
        let phi = (&p - 1) * (&q - 1);
        let e = BigInt::from(65537u32);
        let d = e.modinverse(&phi).expect("无法计算模逆元");

        // 生成一个用于结果验证的素数 P (比 x 小，用于最终校验)
        let prime_p = BigInt::from(PRIME_P);

        Alice {
            wealth,
            n,
            e,
            d,
            prime_p,
        }
    }

    /// 发布公钥
    fn get_public_key(&self) -> (BigInt, BigInt) {
        (self.n.clone(), self.e.clone())
    }

    /// 核心逻辑：处理 Bob 发来的密文 C，返回结果列表（长度 = MAX_WEALTH）
    fn process_ciphertext(&self, c: BigInt) -> (Vec<BigInt>, BigInt) {
        let mut result_list = Vec::new();

        // 遍历所有可能的财富值 (1 到 MAX_WEALTH)
        for i in 1..=MAX_WEALTH {
            let i_big = BigInt::from(i);

            // 1. 尝试性解密: Y = (C + i)^d mod N
            // 如果 i 正好等于 Bob 的财富 b，那么 (C + i) 就是 x^e，解密后就是 x
            let base = &c + &i_big;
            let decrypted_val = base.modpow(&self.d, &self.n);

            // 2. 取模 P (缩小数值范围，方便传输和比较)
            let mut final_val = decrypted_val % &self.prime_p;

            // 简化变体：
            // - 若 i < Alice.wealth：Alice 更富 -> 破坏该项（+1）
            // - 否则：保持正确值
            if i_big < BigInt::from(self.wealth) {
                final_val = (final_val + 1) % &self.prime_p;
            }

            result_list.push(final_val);
        }

        (result_list, self.prime_p.clone())
    }
}

impl Bob {
    fn new(wealth: u64) -> Self {
        let mut rng = thread_rng();
        // 关键点：这里需要保证 x < N 且 x < P。
        // 否则 Alice 解密得到的是 x mod N，而 Bob 用 x mod P 校验会不一致。
        let secret_x = rng.gen_bigint_range(&BigInt::one(), &BigInt::from(PRIME_P));
        Bob { wealth, secret_x }
    }

    /// 第一步：Bob 生成加密请求
    /// C = (x^e - b) mod N
    fn encrypt_request(&self, pub_key: (BigInt, BigInt)) -> BigInt {
        let (n, e) = pub_key;

        // K = x^e mod N
        let k = self.secret_x.modpow(&e, &n);
        let b_big = BigInt::from(self.wealth);

        // C = K - b
        // 注意：在大数减法中要处理负数取模的情况
        let c = k - b_big;

        // 确保发送的是正数 (虽然 num-bigint 处理负数 modpow 也可以，但最好标准化)
        // 实际数学含义是 c mod N
        // 这里简单返回 c 即可，Alice 会加上 i 后再模 N
        c
    }

    fn protocol_says_alice_leq_bob(&self, results: &[BigInt], prime_p: &BigInt) -> bool {
        debug_assert!(self.wealth >= 1 && self.wealth <= MAX_WEALTH);
        let my_check = &self.secret_x % prime_p;
        my_check == results[(self.wealth - 1) as usize]
    }

    /// 最后一步：Bob 查看结果（用于演示输出）
    fn check_result(&self, results: &[BigInt], prime_p: &BigInt) {
        println!("\n[Bob] 查看第 {} 个盒子...", self.wealth);

        let my_check = &self.secret_x % prime_p;
        let alice_value = &results[(self.wealth - 1) as usize];

        println!("  Bob 的 x mod P = {}", my_check);
        println!("  Alice 的值     = {}", alice_value);

        if self.protocol_says_alice_leq_bob(results, prime_p) {
            println!("\n🔴 结果揭晓: 值匹配！");
            println!("   这意味着 Alice 没有修改数据。");
            println!("   逻辑判定: Alice 的财富 <= Bob ({})", self.wealth);
            println!("   🎉 Bob 更富有 (或一样有钱)!");
        } else {
            println!("\n🟢 结果揭晓: 值不匹配！");
            println!("   这意味着 Alice 修改了数据。");
            println!("   逻辑判定: Alice 的财富 > Bob ({})", self.wealth);
            println!("   🎉 Alice 更富有!");
        }
    }
}

// 简单的扩展 trait 用于计算模逆元 (d)
trait ModInverse {
    fn modinverse(&self, n: &BigInt) -> Option<BigInt>;
}

impl ModInverse for BigInt {
    fn modinverse(&self, n: &BigInt) -> Option<BigInt> {
        let (g, x, _) = egcd(self, n);
        if g != BigInt::one() {
            None
        } else {
            Some((x % n + n) % n)
        }
    }
}

// 扩展欧几里得算法求逆元
fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        (a.clone(), BigInt::one(), BigInt::zero())
    } else {
        let (g, x, y) = egcd(b, &(a % b));
        (g, y.clone(), x - (a / b) * y)
    }
}

fn main() {
    println!("--- 姚期智百万富翁问题 ---");
    println!(
        "财富范围: 1..={}。随机生成 Alice/Bob 财富，多轮验证协议判断是否正确。\n",
        MAX_WEALTH
    );

    const TRIALS: usize = 200;
    let mut rng = thread_rng();
    let mut mismatches = 0usize;

    for t in 1..=TRIALS {
        let alice_wealth = rng.gen_range(1..=MAX_WEALTH);
        let bob_wealth = rng.gen_range(1..=MAX_WEALTH);

        let alice = Alice::new(alice_wealth);
        let bob = Bob::new(bob_wealth);

        let pub_key = alice.get_public_key();
        let ciphertext = bob.encrypt_request(pub_key);
        let (results, p) = alice.process_ciphertext(ciphertext);

        if t == 1 {
            println!("[样例] Alice={} vs Bob={}", alice_wealth, bob_wealth);
            bob.check_result(&results, &p);
            println!("\n----------------------------------------\n");
        }

        let protocol_says_alice_leq_bob = bob.protocol_says_alice_leq_bob(&results, &p);
        let truth_alice_leq_bob = alice_wealth <= bob_wealth;

        if protocol_says_alice_leq_bob != truth_alice_leq_bob {
            mismatches += 1;
            println!("[Mismatch #{mismatches}] trial={t}");
            println!("  Alice wealth = {alice_wealth}");
            println!("  Bob wealth   = {bob_wealth}");
            println!("  protocol says Alice <= Bob ? {protocol_says_alice_leq_bob}");
            println!("  truth says    Alice <= Bob ? {truth_alice_leq_bob}");
            println!("  (为定位方便) 重新输出一次 Bob 视角：");
            bob.check_result(&results, &p);
            break;
        }
    }

    if mismatches == 0 {
        println!("✅ {TRIALS} 次随机测试全部通过（协议判断与真实比较一致）");
    } else {
        println!("❌ 发现不一致：{mismatches} 次（已打印首个反例）");
    }
}
