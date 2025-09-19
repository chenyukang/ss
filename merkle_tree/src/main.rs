use hex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// 定义一个 Result 类型别名，方便处理错误
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// 定义哈希特征，支持不同的哈希算法
pub trait Hasher {
    fn hash(&self, data: &[u8]) -> Vec<u8>;
}

// SHA256 哈希器实现
#[derive(Debug)]
pub struct Sha256Hasher;

impl Hasher for Sha256Hasher {
    fn hash(&self, data: &[u8]) -> Vec<u8> {
        Sha256::digest(data).to_vec()
    }
}

// Merkle Tree 结构体，支持泛型哈希器
#[derive(Debug)]
pub struct MerkleTree<H: Hasher> {
    // 存储原始数据
    data: Vec<Vec<u8>>,
    // 存储所有层的哈希值 (从叶子层到根)
    levels: Vec<Vec<Vec<u8>>>,
    // 哈希器
    hasher: H,
    // 数据索引映射
    data_index_map: HashMap<Vec<u8>, usize>,
}

impl<H: Hasher> MerkleTree<H> {
    // 根据数据创建 Merkle Tree
    pub fn new(data: &[&[u8]], hasher: H) -> Result<MerkleTree<H>> {
        if data.is_empty() {
            return Err("Input data cannot be empty.".into());
        }

        // 存储原始数据
        let data_vec: Vec<Vec<u8>> = data.iter().map(|item| item.to_vec()).collect();

        // 创建数据索引映射
        let mut data_index_map = HashMap::new();
        for (index, item) in data_vec.iter().enumerate() {
            data_index_map.insert(item.clone(), index);
        }

        // 计算叶子节点的哈希值
        let leaf_hashes: Vec<Vec<u8>> = data_vec.iter().map(|item| hasher.hash(item)).collect();

        // 构建所有层
        let mut levels = vec![leaf_hashes];

        while levels.last().unwrap().len() > 1 {
            let current_level = levels.last().unwrap();
            let mut next_level: Vec<Vec<u8>> = Vec::new();

            let mut i = 0;
            while i < current_level.len() {
                let left = &current_level[i];
                // 如果是奇数节点，最后一个节点与自身配对
                let right = if i + 1 < current_level.len() {
                    &current_level[i + 1]
                } else {
                    left
                };

                let mut combined = left.clone();
                combined.extend_from_slice(right);
                next_level.push(hasher.hash(&combined));
                i += 2;
            }
            levels.push(next_level);
        }

        Ok(MerkleTree {
            data: data_vec,
            levels,
            hasher,
            data_index_map,
        })
    }

    // 获取 Merkle Root
    pub fn get_root(&self) -> Option<&[u8]> {
        self.levels.last()?.get(0).map(|v| v.as_slice())
    }

    // 根据数据内容生成 Merkle Proof
    pub fn generate_proof(&self, data: &[u8]) -> Option<Vec<Vec<u8>>> {
        // 查找数据在叶子层的索引
        let data_vec = data.to_vec();
        let leaf_index = *self.data_index_map.get(&data_vec)?;

        self.generate_proof_by_index(leaf_index)
    }

    // 根据索引生成 Merkle Proof
    pub fn generate_proof_by_index(&self, mut index: usize) -> Option<Vec<Vec<u8>>> {
        if index >= self.levels[0].len() {
            return None;
        }

        let mut proof = Vec::new();

        // 从叶子层向上遍历到根节点（不包括根节点）
        for level in &self.levels[..self.levels.len() - 1] {
            // 找到兄弟节点的索引
            let sibling_index = if index % 2 == 0 {
                // 当前节点是左节点，兄弟节点在右边
                if index + 1 < level.len() {
                    index + 1
                } else {
                    index // 奇数个节点时，与自己配对
                }
            } else {
                // 当前节点是右节点，兄弟节点在左边
                index - 1
            };

            proof.push(level[sibling_index].clone());

            // 移动到上一层的父节点索引
            index /= 2;
        }

        Some(proof)
    }

    // 生成带有位置信息的证明
    pub fn generate_proof_with_positions(&self, data: &[u8]) -> Option<Vec<(Vec<u8>, bool)>> {
        let data_vec = data.to_vec();
        let leaf_index = *self.data_index_map.get(&data_vec)?;
        self.generate_proof_with_positions_by_index(leaf_index)
    }

    // 根据索引生成带有位置信息的证明（包含是否为左节点的信息）
    pub fn generate_proof_with_positions_by_index(
        &self,
        mut index: usize,
    ) -> Option<Vec<(Vec<u8>, bool)>> {
        if index >= self.levels[0].len() {
            return None;
        }

        let mut proof = Vec::new();

        for level in &self.levels[..self.levels.len() - 1] {
            let is_left_node = index % 2 == 0;
            let sibling_index = if is_left_node {
                if index + 1 < level.len() {
                    index + 1
                } else {
                    index
                }
            } else {
                index - 1
            };

            proof.push((level[sibling_index].clone(), !is_left_node)); // 记录兄弟节点应该放在左边还是右边
            index /= 2;
        }

        Some(proof)
    }

    // 验证数据是否存在于 Merkle Tree 中
    pub fn verify_proof(&self, data: &[u8], proof: &[Vec<u8>]) -> bool {
        // 使用带位置信息的证明来验证
        if let Some(proof_with_positions) = self.generate_proof_with_positions(data) {
            if proof.len() != proof_with_positions.len() {
                return false;
            }

            // 检查证明是否匹配
            for (i, (expected_hash, _)) in proof_with_positions.iter().enumerate() {
                if &proof[i] != expected_hash {
                    return false;
                }
            }

            // 使用位置信息验证
            self.verify_proof_with_positions(data, &proof_with_positions)
        } else {
            false
        }
    }

    // 使用带位置信息的证明验证
    pub fn verify_proof_with_positions(&self, data: &[u8], proof: &[(Vec<u8>, bool)]) -> bool {
        let mut hash = self.hasher.hash(data);

        // 逐层向上计算哈希值
        for (proof_hash, sibling_is_left) in proof.iter() {
            let mut combined = Vec::new();
            // 根据位置信息确定左右顺序
            if *sibling_is_left {
                combined.extend_from_slice(proof_hash);
                combined.extend_from_slice(&hash);
            } else {
                combined.extend_from_slice(&hash);
                combined.extend_from_slice(proof_hash);
            }
            hash = self.hasher.hash(&combined);
        }

        // 验证计算出的哈希值是否等于 Merkle Root
        if let Some(root_hash) = self.get_root() {
            root_hash == hash.as_slice()
        } else {
            false
        }
    }

    // 获取原始数据
    pub fn get_data(&self, index: usize) -> Option<&[u8]> {
        self.data.get(index).map(|v| v.as_slice())
    }

    // 获取叶子节点数量
    pub fn leaf_count(&self) -> usize {
        self.data.len()
    }

    // 打印树结构（用于调试）
    pub fn print_tree(&self) {
        println!("Merkle Tree Structure:");
        for (level_idx, level) in self.levels.iter().enumerate().rev() {
            println!(
                "Level {}: {} nodes",
                self.levels.len() - 1 - level_idx,
                level.len()
            );
            for (node_idx, hash) in level.iter().enumerate() {
                println!("  Node {}: {}", node_idx, hex::encode(&hash[..8])); // 只显示前8字节
            }
        }
    }

    // 插入新的数据项
    pub fn insert(&mut self, data: &[u8]) -> Result<()> {
        let data_vec = data.to_vec();

        // 检查数据是否已存在
        if self.data_index_map.contains_key(&data_vec) {
            return Err("Data already exists in the tree".into());
        }

        // 添加到数据列表
        let new_index = self.data.len();
        self.data.push(data_vec.clone());
        self.data_index_map.insert(data_vec, new_index);

        // 重建树
        self.rebuild_tree()
    }

    // 根据数据内容删除项目
    pub fn remove(&mut self, data: &[u8]) -> Result<bool> {
        let data_vec = data.to_vec();

        // 查找要删除的索引
        if let Some(&index) = self.data_index_map.get(&data_vec) {
            self.remove_by_index(index)?;
            Ok(true)
        } else {
            Ok(false) // 数据不存在
        }
    }

    // 根据索引删除项目
    pub fn remove_by_index(&mut self, index: usize) -> Result<()> {
        if index >= self.data.len() {
            return Err("Index out of bounds".into());
        }

        // 从数据映射中移除
        let removed_data = self.data[index].clone();
        self.data_index_map.remove(&removed_data);

        // 从数据列表中移除
        self.data.remove(index);

        // 重建数据索引映射（因为索引发生了变化）
        self.data_index_map.clear();
        for (new_index, item) in self.data.iter().enumerate() {
            self.data_index_map.insert(item.clone(), new_index);
        }

        // 如果还有数据，重建树；否则清空树
        if self.data.is_empty() {
            self.levels.clear();
        } else {
            self.rebuild_tree()?;
        }

        Ok(())
    }

    // 批量插入数据
    pub fn insert_batch(&mut self, data_batch: &[&[u8]]) -> Result<()> {
        for &data in data_batch {
            let data_vec = data.to_vec();

            // 跳过已存在的数据
            if self.data_index_map.contains_key(&data_vec) {
                continue;
            }

            let new_index = self.data.len();
            self.data.push(data_vec.clone());
            self.data_index_map.insert(data_vec, new_index);
        }

        // 只重建一次树
        if !data_batch.is_empty() {
            self.rebuild_tree()?;
        }

        Ok(())
    }

    // 重建整个 Merkle Tree
    fn rebuild_tree(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Err("Cannot rebuild tree with empty data".into());
        }

        // 计算叶子节点的哈希值
        let leaf_hashes: Vec<Vec<u8>> = self
            .data
            .iter()
            .map(|item| self.hasher.hash(item))
            .collect();

        // 构建所有层
        let mut levels = vec![leaf_hashes];

        while levels.last().unwrap().len() > 1 {
            let current_level = levels.last().unwrap();
            let mut next_level: Vec<Vec<u8>> = Vec::new();

            let mut i = 0;
            while i < current_level.len() {
                let left = &current_level[i];
                // 如果是奇数节点，最后一个节点与自身配对
                let right = if i + 1 < current_level.len() {
                    &current_level[i + 1]
                } else {
                    left
                };

                let mut combined = left.clone();
                combined.extend_from_slice(right);
                next_level.push(self.hasher.hash(&combined));
                i += 2;
            }
            levels.push(next_level);
        }

        self.levels = levels;
        Ok(())
    }

    // 检查数据是否存在
    pub fn contains(&self, data: &[u8]) -> bool {
        let data_vec = data.to_vec();
        self.data_index_map.contains_key(&data_vec)
    }

    // 获取数据的索引
    pub fn get_index(&self, data: &[u8]) -> Option<usize> {
        let data_vec = data.to_vec();
        self.data_index_map.get(&data_vec).copied()
    }

    // 清空树
    pub fn clear(&mut self) {
        self.data.clear();
        self.levels.clear();
        self.data_index_map.clear();
    }

    // 获取树是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// 示例用法
fn main() -> Result<()> {
    println!("Merkle Tree implementation ready!");
    println!("Run `cargo test` to execute all unit tests.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_data() -> (Vec<&'static [u8]>, MerkleTree<Sha256Hasher>) {
        let tx_a: &'static [u8] = b"Transaction A";
        let tx_b: &'static [u8] = b"Transaction B";
        let tx_c: &'static [u8] = b"Transaction C";
        let data = vec![tx_a, tx_b, tx_c];
        let tree = MerkleTree::new(&data, Sha256Hasher).unwrap();
        (data, tree)
    }

    #[test]
    fn test_tree_creation() {
        let (data, tree) = setup_test_data();

        // 验证树创建成功
        assert!(!tree.is_empty());
        assert_eq!(tree.leaf_count(), 3);
        assert!(tree.get_root().is_some());

        // 验证原始数据存储正确
        for (i, &expected_data) in data.iter().enumerate() {
            assert_eq!(tree.get_data(i).unwrap(), expected_data);
        }
    }

    #[test]
    fn test_proof_generation_and_verification() {
        let (data, tree) = setup_test_data();
        let tx_a = data[0];
        let tx_b = data[1];

        // 测试为 tx_a 生成证明
        let proof_a = tree
            .generate_proof(tx_a)
            .expect("Should generate proof for tx_a");
        assert!(!proof_a.is_empty());

        // 验证 tx_a 的证明
        assert!(tree.verify_proof(tx_a, &proof_a));

        // 用 tx_a 的证明验证 tx_b 应该失败
        assert!(!tree.verify_proof(tx_b, &proof_a));

        // 测试按索引生成证明
        let proof_by_index = tree
            .generate_proof_by_index(0)
            .expect("Should generate proof by index");
        assert_eq!(proof_a, proof_by_index);
        assert!(tree.verify_proof(tx_a, &proof_by_index));
    }

    #[test]
    fn test_invalid_data_verification() {
        let (_, tree) = setup_test_data();
        let tx_a = b"Transaction A";
        let invalid_tx = b"Transaction D";

        let proof_a = tree.generate_proof(tx_a).unwrap();

        // 验证不存在的交易应该失败
        assert!(!tree.verify_proof(invalid_tx, &proof_a));

        // 对不存在的数据生成证明应该返回 None
        assert!(tree.generate_proof(invalid_tx).is_none());
    }

    #[test]
    fn test_data_insertion() {
        let (_, mut tree) = setup_test_data();
        let original_count = tree.leaf_count();
        let original_root = tree.get_root().unwrap().to_vec();

        let tx_d = b"Transaction D";

        // 测试插入新数据
        assert!(tree.insert(tx_d).is_ok());
        assert_eq!(tree.leaf_count(), original_count + 1);
        assert!(tree.contains(tx_d));

        // 根应该发生变化
        let new_root = tree.get_root().unwrap();
        assert_ne!(original_root, new_root);

        // 验证新插入的数据
        let proof_d = tree
            .generate_proof(tx_d)
            .expect("Should generate proof for new data");
        assert!(tree.verify_proof(tx_d, &proof_d));

        // 验证原有数据仍然有效
        let tx_a = b"Transaction A";
        let proof_a = tree
            .generate_proof(tx_a)
            .expect("Should generate proof for existing data");
        assert!(tree.verify_proof(tx_a, &proof_a));
    }

    #[test]
    fn test_duplicate_insertion() {
        let (_, mut tree) = setup_test_data();
        let tx_a = b"Transaction A";

        // 尝试插入重复数据应该失败
        let result = tree.insert(tx_a);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_batch_insertion() {
        let (_, mut tree) = setup_test_data();
        let original_count = tree.leaf_count();

        let tx_d: &[u8] = b"Transaction D";
        let tx_e: &[u8] = b"Transaction E";
        let tx_f: &[u8] = b"Transaction F";
        let new_data = vec![tx_d, tx_e, tx_f];

        // 批量插入
        assert!(tree.insert_batch(&new_data).is_ok());
        assert_eq!(tree.leaf_count(), original_count + 3);

        // 验证所有新数据都存在且有效
        for &tx in &new_data {
            assert!(tree.contains(tx));
            let proof = tree.generate_proof(tx).expect("Should generate proof");
            assert!(tree.verify_proof(tx, &proof));
        }
    }

    #[test]
    fn test_data_removal() {
        let (data, mut tree) = setup_test_data();
        let tx_a = data[0];
        let tx_b = data[1];
        let original_count = tree.leaf_count();

        // 删除存在的数据
        let removed = tree.remove(tx_a).unwrap();
        assert!(removed);
        assert_eq!(tree.leaf_count(), original_count - 1);
        assert!(!tree.contains(tx_a));

        // 验证剩余数据仍然有效
        let proof_b = tree
            .generate_proof(tx_b)
            .expect("Should generate proof for remaining data");
        assert!(tree.verify_proof(tx_b, &proof_b));

        // 删除不存在的数据
        let removed_again = tree.remove(tx_a).unwrap();
        assert!(!removed_again);
    }

    #[test]
    fn test_removal_by_index() {
        let (_, mut tree) = setup_test_data();
        let original_count = tree.leaf_count();

        // 记录索引 0 的数据
        let data_at_0 = tree.get_data(0).unwrap().to_vec();

        // 按索引删除
        assert!(tree.remove_by_index(0).is_ok());
        assert_eq!(tree.leaf_count(), original_count - 1);
        assert!(!tree.contains(&data_at_0));

        // 删除无效索引应该失败
        let result = tree.remove_by_index(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_operations() {
        let (_, tree) = setup_test_data();
        let tx_a = b"Transaction A";

        // 测试 contains
        assert!(tree.contains(tx_a));
        assert!(!tree.contains(b"Non-existent"));

        // 测试 get_index
        assert_eq!(tree.get_index(tx_a), Some(0));
        assert_eq!(tree.get_index(b"Non-existent"), None);

        // 测试 is_empty
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_clear_tree() {
        let (_, mut tree) = setup_test_data();

        assert!(!tree.is_empty());
        assert!(tree.leaf_count() > 0);

        tree.clear();

        assert!(tree.is_empty());
        assert_eq!(tree.leaf_count(), 0);
        assert!(tree.get_root().is_none());
    }

    #[test]
    fn test_single_node_tree() {
        let data: Vec<&[u8]> = vec![b"Single Transaction"];
        let tree = MerkleTree::new(&data, Sha256Hasher).unwrap();

        assert_eq!(tree.leaf_count(), 1);
        assert!(tree.get_root().is_some());

        let proof = tree.generate_proof(data[0]).unwrap();
        assert!(proof.is_empty()); // 单节点树的证明为空
        assert!(tree.verify_proof(data[0], &proof));
    }

    #[test]
    fn test_empty_data_error() {
        let empty_data: Vec<&[u8]> = vec![];
        let result = MerkleTree::new(&empty_data, Sha256Hasher);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_proof_generation_edge_cases() {
        let (_, tree) = setup_test_data();

        // 测试超出范围的索引
        assert!(tree.generate_proof_by_index(100).is_none());

        // 测试不存在的数据
        assert!(tree.generate_proof(b"Non-existent").is_none());
    }

    #[test]
    fn test_root_consistency_after_modifications() {
        let (data, mut tree) = setup_test_data();
        let original_root = tree.get_root().unwrap().to_vec();

        // 插入数据后根应该改变
        tree.insert(b"Transaction D").unwrap();
        let root_after_insert = tree.get_root().unwrap().to_vec();
        assert_ne!(original_root, root_after_insert);

        // 删除刚插入的数据，根应该回到原来的值
        tree.remove(b"Transaction D").unwrap();
        let root_after_removal = tree.get_root().unwrap().to_vec();
        assert_eq!(original_root, root_after_removal);

        // 验证原始数据仍然有效
        for &tx in &data {
            let proof = tree.generate_proof(tx).unwrap();
            assert!(tree.verify_proof(tx, &proof));
        }
    }

    #[test]
    fn test_large_tree_operations() {
        // 创建较大的树来测试性能和正确性
        let mut large_data = Vec::new();
        let n = 10000;
        for i in 0..n {
            large_data.push(format!("Transaction {}", i).into_bytes());
        }

        let large_data_refs: Vec<&[u8]> = large_data.iter().map(|v| v.as_slice()).collect();
        let tree = MerkleTree::new(&large_data_refs, Sha256Hasher).unwrap();

        assert_eq!(tree.leaf_count(), n);
        assert!(tree.get_root().is_some());

        // 随机验证几个节点
        let mut rands = vec![];
        for _i in 0..(n / 3) {
            let rand_idx = (rand::random::<u32>() % (n as u32)) as usize;
            rands.push(rand_idx);
        }
        for i in rands.into_iter() {
            let proof = tree.generate_proof(&large_data[i]).unwrap();
            assert!(tree.verify_proof(&large_data[i], &proof));
        }
    }
}
