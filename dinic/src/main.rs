use std::collections::VecDeque;

/// 表示图中的一条边
#[derive(Clone, Debug)]
struct Edge {
    to: usize,       // 边的终点
    capacity: u64,   // 边的剩余容量
    rev_idx: usize,  // 反向边在其终点邻接表中的索引
}

/// Dinic 算法的实现结构体
pub struct Dinic {
    num_nodes: usize,
    graph: Vec<Vec<Edge>>,
    level: Vec<i32>,
    iter: Vec<usize>,
}

impl Dinic {
    /// 创建一个新的 Dinic 实例
    ///
    /// # 参数
    /// * `num_nodes`: 图中节点的总数 (节点ID从 0 到 num_nodes-1)
    pub fn new(num_nodes: usize) -> Self {
        Dinic {
            num_nodes,
            graph: vec![Vec::new(); num_nodes],
            level: vec![0; num_nodes],
            iter: vec![0; num_nodes],
        }
    }

    /// 向图中添加一条从 `from` 到 `to`，容量为 `capacity` 的边
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: u64) {
        // 在添加正向边之前，记录下反向边将要插入的位置
        let to_rev_idx = self.graph[to].len();
        self.graph[from].push(Edge {
            to,
            capacity,
            rev_idx: to_rev_idx,
        });

        // 在添加反向边之前，记录下正向边插入的位置
        let from_rev_idx = self.graph[from].len() - 1;
        self.graph[to].push(Edge {
            to: from,
            capacity: 0, // 反向边的初始容量为 0
            rev_idx: from_rev_idx,
        });
    }

    /// 通过 BFS 构建分层图
    ///
    /// # 返回
    /// 如果汇点 `t` 可达，则返回 `true`，否则 `false`
    fn bfs(&mut self, s: usize, t: usize) -> bool {
        // 初始化 level 数组，-1 表示不可达
        self.level = vec![-1; self.num_nodes];
        let mut q = VecDeque::new();

        self.level[s] = 0;
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            for edge in &self.graph[u] {
                // 如果边有剩余容量，且终点尚未被访问
                if edge.capacity > 0 && self.level[edge.to] < 0 {
                    self.level[edge.to] = self.level[u] + 1;
                    q.push_back(edge.to);
                }
            }
        }

        // 如果 level[t] 不再是 -1，说明汇点可达
        self.level[t] != -1
    }

    /// 通过 DFS 在分层图上寻找增广路径并推送流量
    /// 这是寻找阻塞流的核心步骤
    ///
    /// # 参数
    /// * `u`: 当前节点
    /// * `t`: 汇点
    /// * `f`: 到达当前节点 `u` 的最大可能流量
    ///
    /// # 返回
    /// 从 `u` 出发实际推送的流量
    fn dfs(&mut self, u: usize, t: usize, f: u64) -> u64 {
        // 如果到达汇点，说明找到了一条完整的增广路径
        if u == t {
            return f;
        }

        // 当前弧优化：使用 `self.iter[u]` 来记录当前节点 `u` 尝试过的边
        // 避免在一次 DFS 中重复访问已经无法增广的边
        while self.iter[u] < self.graph[u].len() {
            let edge_idx = self.iter[u];
            let to = self.graph[u][edge_idx].to;
            let capacity = self.graph[u][edge_idx].capacity;
            let rev_idx = self.graph[u][edge_idx].rev_idx;

            // 只沿着分层图中的边进行搜索
            if capacity > 0 && self.level[u] < self.level[to] {
                // 递归地向下游寻找路径，流量上限为 f 和当前边容量的较小值
                let d = self.dfs(to, t, f.min(capacity));

                // 如果下游成功推送了流量 (d > 0)
                if d > 0 {
                    // 更新正向边的剩余容量
                    self.graph[u][edge_idx].capacity -= d;
                    // 更新反向边的剩余容量
                    self.graph[to][rev_idx].capacity += d;
                    return d;
                }
            }
            // 如果当前边无法增广，则尝试下一条边
            self.iter[u] += 1;
        }

        // 如果从 `u` 出发的所有边都无法到达 `t`，返回 0
        0
    }

    /// 计算从 `s` 到 `t` 的最大流
    pub fn max_flow(&mut self, s: usize, t: usize) -> u64 {
        let mut flow = 0;
        loop {
            // 1. 构建分层图，如果汇点不可达，则算法结束
            if !self.bfs(s, t) {
                return flow;
            }

            // 2. 重置 iter 数组，为新一轮的 DFS 做准备 (当前弧优化)
            self.iter = vec![0; self.num_nodes];

            // 3. 在分层图上持续寻找增广路径，直到找不到为止 (找到阻塞流)
            loop {
                let f = self.dfs(s, t, u64::MAX);
                if f == 0 {
                    break;
                }
                flow += f;
            }
        }
    }
}

// 主函数，用于展示如何使用 Dinic 算法
fn main() {
    // 节点数: 6 (0=S, 5=T)
    let num_nodes = 6;
    let s = 0;
    let t = 5;

    let mut dinic = Dinic::new(num_nodes);

    // 添加边和容量
    dinic.add_edge(0, 1, 16);
    dinic.add_edge(0, 2, 13);
    dinic.add_edge(1, 2, 10);
    dinic.add_edge(1, 3, 12);
    dinic.add_edge(2, 1, 4);
    dinic.add_edge(2, 4, 14);
    dinic.add_edge(3, 2, 9);
    dinic.add_edge(3, 5, 20);
    dinic.add_edge(4, 3, 7);
    dinic.add_edge(4, 5, 4);

    let max_flow = dinic.max_flow(s, t);
    println!("从节点 {} 到节点 {} 的最大流是: {}", s, t, max_flow); // 应该输出 23
}


// --- 单元测试 ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_case() {
        let mut dinic = Dinic::new(4);
        let s = 0;
        let t = 3;
        dinic.add_edge(s, 1, 10);
        dinic.add_edge(s, 2, 10);
        dinic.add_edge(1, t, 10);
        dinic.add_edge(2, t, 10);
        assert_eq!(dinic.max_flow(s, t), 20);
    }

    #[test]
    fn test_classic_clrs_example() {
        let mut dinic = Dinic::new(6);
        let s = 0;
        let t = 5;
        dinic.add_edge(0, 1, 16);
        dinic.add_edge(0, 2, 13);
        dinic.add_edge(1, 2, 10);
        dinic.add_edge(1, 3, 12);
        dinic.add_edge(2, 1, 4);
        dinic.add_edge(2, 4, 14);
        dinic.add_edge(3, 2, 9);
        dinic.add_edge(3, 5, 20);
        dinic.add_edge(4, 3, 7);
        dinic.add_edge(4, 5, 4);
        assert_eq!(dinic.max_flow(s, t), 23);
    }

    #[test]
    fn test_undo_flow_case() {
        // 这个测试用例需要反向边才能找到正确答案
        let mut dinic = Dinic::new(4);
        let s = 0;
        let t = 3;
        dinic.add_edge(s, 1, 1);
        dinic.add_edge(s, 2, 1);
        dinic.add_edge(1, 2, 1); // 关键的中间边
        dinic.add_edge(1, t, 1);
        dinic.add_edge(2, t, 1);
        assert_eq!(dinic.max_flow(s, t), 2);
    }

    #[test]
    fn test_disconnected_graph() {
        let mut dinic = Dinic::new(4);
        let s = 0;
        let t = 3;
        dinic.add_edge(s, 1, 10);
        dinic.add_edge(2, t, 10); // S和T不连通
        assert_eq!(dinic.max_flow(s, t), 0);
    }

    #[test]
    fn test_no_capacity() {
        let mut dinic = Dinic::new(4);
        let s = 0;
        let t = 3;
        dinic.add_edge(s, 1, 0);
        dinic.add_edge(1, t, 10);
        assert_eq!(dinic.max_flow(s, t), 0);
    }
}