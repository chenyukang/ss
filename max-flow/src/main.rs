// main.rs

use std::collections::VecDeque;

// --- 1. Dinic 最大流算法的实现 ---

// 边结构：to -> 目标节点, cap -> 容量, rev -> 反向边的索引
#[derive(Clone)]
struct Edge {
    to: usize,
    cap: i64,
    rev: usize,
}

// 图结构：使用邻接表表示
struct Graph {
    adj: Vec<Vec<Edge>>,
}

impl Graph {
    fn new(n: usize) -> Self {
        Graph {
            adj: vec![Vec::new(); n],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: i64) {
        let from_len = self.adj[from].len();
        let to_len = self.adj[to].len();
        self.adj[from].push(Edge { to, cap, rev: to_len });
        self.adj[to].push(Edge { to: from, cap: 0, rev: from_len }); // 反向边初始容量为0
    }
}

// Dinic 算法结构体
struct Dinic {
    graph: Graph,
    level: Vec<i32>,
    iter: Vec<usize>,
}

impl Dinic {
    fn new(graph: Graph) -> Self {
        let n = graph.adj.len();
        Dinic {
            graph,
            level: vec![0; n],
            iter: vec![0; n],
        }
    }

    // 通过 BFS 构建层次图
    fn bfs(&mut self, s: usize) -> bool {
        let n = self.graph.adj.len();
        self.level = vec![-1; n];
        let mut q = VecDeque::new();
        self.level[s] = 0;
        q.push_back(s);

        while let Some(v) = q.pop_front() {
            for edge in &self.graph.adj[v] {
                if edge.cap > 0 && self.level[edge.to] < 0 {
                    self.level[edge.to] = self.level[v] + 1;
                    q.push_back(edge.to);
                }
            }
        }
        self.level[self.graph.adj.len() - 1] != -1
    }

    // 通过 DFS 在层次图上寻找增广路
    fn dfs(&mut self, v: usize, t: usize, f: i64) -> i64 {
        if v == t {
            return f;
        }
        while self.iter[v] < self.graph.adj[v].len() {
            let edge_idx = self.iter[v];
            let edge = self.graph.adj[v][edge_idx].clone();

            if edge.cap > 0 && self.level[v] < self.level[edge.to] {
                let d = self.dfs(edge.to, t, f.min(edge.cap));
                if d > 0 {
                    self.graph.adj[v][edge_idx].cap -= d;
                    let rev_edge_idx = edge.rev;
                    self.graph.adj[edge.to][rev_edge_idx].cap += d;
                    return d;
                }
            }
            self.iter[v] += 1;
        }
        0
    }

    // 计算从 s 到 t 的最大流
    fn max_flow(&mut self, s: usize, t: usize) -> i64 {
        let mut flow = 0;
        while self.bfs(s) {
            let n = self.graph.adj.len();
            self.iter = vec![0; n];
            loop {
                let f = self.dfs(s, t, i64::MAX);
                if f == 0 {
                    break;
                }
                flow += f;
            }
        }
        flow
    }
}


// --- 2. 带有下界约束的最大流主逻辑 ---

// 用于描述原始图的边
struct EdgeWithLowerBound {
    from: usize,
    to: usize,
    lower_bound: i64,
    cap: i64,
}

/// 计算带有下界约束的最大流
/// 返回 Result，如果无解则返回 Err
fn max_flow_with_lower_bounds(
    num_nodes: usize,
    edges: &[EdgeWithLowerBound],
    s: usize,
    t: usize,
) -> Result<i64, &'static str> {
    // --- 阶段一：检查是否存在可行流 ---

    // 1. 计算每个节点的差额 D(v)
    let mut demand = vec![0i64; num_nodes];
    for edge in edges {
        demand[edge.from] -= edge.lower_bound;
        demand[edge.to] += edge.lower_bound;
    }

    // 2. 构建补偿图 G'
    // 新增一个超级源点 S' 和一个超级汇点 T'
    let super_s = num_nodes;
    let super_t = num_nodes + 1;
    let mut g_prime = Graph::new(num_nodes + 2);

    for edge in edges {
        if edge.cap < edge.lower_bound {
            return Err("Infeasible: capacity is less than lower bound");
        }
        // 添加自由流量边
        g_prime.add_edge(edge.from, edge.to, edge.cap - edge.lower_bound);
    }

    let mut total_deficit = 0;
    for i in 0..num_nodes {
        if demand[i] > 0 { // 盈余
            g_prime.add_edge(super_s, i, demand[i]);
        } else if demand[i] < 0 { // 亏空
            g_prime.add_edge(i, super_t, -demand[i]);
            total_deficit += -demand[i];
        }
    }

    // 3. 运行最大流以检查可行性
    let mut dinic_feasibility = Dinic::new(g_prime);
    let feasible_flow = dinic_feasibility.max_flow(super_s, super_t);

    if feasible_flow != total_deficit {
        return Err("Infeasible: Cannot satisfy all lower bound constraints");
    }

    // --- 阶段二：计算最终的最大流 ---
    // 可行流存在。现在，我们在残差图的基础上计算从 s 到 t 的最大流。
    // Dinic 结构体内部的 graph 已经是残差图了。
    let mut dinic_final = dinic_feasibility;
    let additional_flow = dinic_final.max_flow(s, t);

    // 最终的 s-t 最大流就是这个 additional_flow
    // 因为可行流部分只是为了满足下界，可以看作是一个“基准”或“循环流”。
    // 在此之上的从s到t的任何流量都是真正的s-t净流量。
    Ok(additional_flow)
}


// --- 3. 用法演示 ---

fn main() {
    // 示例图：
    // 节点: 0(s), 1, 2, 3(t)
    // 边 (from, to, lower_bound, capacity)
    // 0 -> 1, [10, 20]
    // 0 -> 2, [5, 15]
    // 1 -> 2, [0, 5]  (普通边)
    // 1 -> 3, [8, 10]
    // 2 -> 3, [10, 25]

    let num_nodes = 4;
    let s = 0;
    let t = 3;
    let edges = vec![
        EdgeWithLowerBound { from: 0, to: 1, lower_bound: 10, cap: 20 },
        EdgeWithLowerBound { from: 0, to: 2, lower_bound: 5, cap: 15 },
        EdgeWithLowerBound { from: 1, to: 2, lower_bound: 0, cap: 5 },
        EdgeWithLowerBound { from: 1, to: 3, lower_bound: 8, cap: 10 },
        EdgeWithLowerBound { from: 2, to: 3, lower_bound: 10, cap: 25 },
    ];

    println!("--- Problem: Max-Flow with Lower Bounds ---");
    println!("Graph has {} nodes, source={}, sink={}", num_nodes, s, t);
    for edge in &edges {
        println!("Edge {}->{}: lower_bound={}, capacity={}", edge.from, edge.to, edge.lower_bound, edge.cap);
    }
    println!("--------------------------------------------");


    match max_flow_with_lower_bounds(num_nodes, &edges, s, t) {
        Ok(flow) => {
            println!("✅ Feasible solution found!");
            println!("Maximum flow from {} to {} is: {}", s, t, flow);
            // 在这里，我们可以获得最终的流网络，并进行路径分解...
            // 例如，源点 0 的总流出量必须满足下界之和: 10 + 5 = 15。
            // 最终的最大流是在这个基础上额外增加的流量。
            // 真实的总流量 = 15(下界流) + flow(额外流)
            let total_s_outflow = edges.iter()
                                       .filter(|e| e.from == s)
                                       .map(|e| e.lower_bound)
                                       .sum::<i64>() + flow;
            println!("Total outflow from source {}: {}", s, total_s_outflow);

        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }

    println!("\n--- Infeasible Example ---");
    let infeasible_edges = vec![
        // 1->2 要求至少10，但0->1最多只能给5
        EdgeWithLowerBound { from: 0, to: 1, lower_bound: 0, cap: 5 },
        EdgeWithLowerBound { from: 1, to: 2, lower_bound: 10, cap: 20 },
    ];
     match max_flow_with_lower_bounds(3, &infeasible_edges, 0, 2) {
        Ok(flow) => println!("Maximum flow: {}", flow), // 不会执行到这里
        Err(e) => println!("Error: {}", e),
    }

}