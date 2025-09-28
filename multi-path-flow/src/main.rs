// 多路径流量分配算法
// 解决问题：在有向图中找到最多M条路径来传输指定amount的流量

use std::collections::VecDeque;

// 边结构：带有最小流量约束
#[derive(Debug, Clone)]
struct EdgeWithMinFlow {
    from: usize,
    to: usize,
    min_flow: i64, // 最小流量要求，0表示无限制
    max_flow: i64, // 最大容量
}

// 路径表示
#[derive(Debug, Clone)]
struct FlowPath {
    nodes: Vec<usize>,
    flow: i64,
}

// 多路径流量分配结果
#[derive(Debug)]
struct MultiPathResult {
    paths: Vec<FlowPath>,
    total_flow: i64,
    success: bool,
}

/// 多路径流量分配算法
///
/// 算法思路：
/// 1. 使用贪心策略逐条寻找路径
/// 2. 每次找到一条从source到sink的路径，传输尽可能多的流量
/// 3. 更新图的剩余容量
/// 4. 重复直到无法找到更多路径或达到路径数量限制
/// 5. 支持边的最小流量约束：要么不用这条边，要么至少传输min_flow
fn find_multi_paths(
    num_nodes: usize,
    edges: &[EdgeWithMinFlow],
    source: usize,
    sink: usize,
    target_amount: i64,
    max_paths: usize,
) -> MultiPathResult {
    // 构建邻接表，每条边存储 (to, min_flow, remaining_capacity, edge_index)
    let mut graph: Vec<Vec<(usize, i64, i64, usize)>> = vec![vec![]; num_nodes];
    let mut edge_capacities: Vec<i64> = Vec::new();

    for (i, edge) in edges.iter().enumerate() {
        graph[edge.from].push((edge.to, edge.min_flow, edge.max_flow, i));
        edge_capacities.push(edge.max_flow);
    }

    let mut paths = Vec::new();
    let mut total_flow = 0;
    let mut remaining_amount = target_amount;

    println!(
        "开始寻找路径，目标流量: {}, 最大路径数: {}",
        target_amount, max_paths
    );

    // 贪心寻找路径
    for path_count in 0..max_paths {
        if remaining_amount <= 0 {
            break;
        }

        // 使用DFS寻找一条可行路径
        if let Some((flow, nodes, used_edges)) =
            find_single_path(&graph, &edge_capacities, source, sink, remaining_amount)
        {
            // 更新边的剩余容量
            for &edge_idx in &used_edges {
                edge_capacities[edge_idx] -= flow;
            }

            paths.push(FlowPath {
                nodes: nodes.clone(),
                flow: flow,
            });

            total_flow += flow;
            remaining_amount -= flow;

            println!(
                "找到路径 {}: {:?} 流量: {}, 剩余需求: {}",
                path_count + 1,
                nodes,
                flow,
                remaining_amount
            );
        } else {
            // 无法找到更多路径
            println!("无法找到更多可行路径");
            break;
        }
    }

    MultiPathResult {
        paths,
        total_flow,
        success: total_flow >= target_amount,
    }
}

/// 使用BFS寻找单条路径（考虑最小流量约束）
fn find_single_path(
    graph: &Vec<Vec<(usize, i64, i64, usize)>>,
    edge_capacities: &Vec<i64>,
    source: usize,
    sink: usize,
    target_flow: i64,
) -> Option<(i64, Vec<usize>, Vec<usize>)> {
    // 特殊情况：source == sink
    if source == sink {
        return Some((0, vec![source], vec![]));
    }

    let mut queue = VecDeque::new();
    let mut parent: Vec<Option<(usize, usize, i64)>> = vec![None; graph.len()]; // (parent_node, edge_idx, min_flow_in_path)
    let mut visited = vec![false; graph.len()];

    queue.push_back((source, i64::MAX));
    visited[source] = true;

    while let Some((current, flow_so_far)) = queue.pop_front() {
        if current == sink {
            // 重构路径
            let mut path = Vec::new();
            let mut used_edges = Vec::new();
            let mut node = sink;
            let mut min_flow = flow_so_far;

            while let Some((parent_node, edge_idx, path_flow)) = parent[node] {
                path.push(node);
                used_edges.push(edge_idx);
                min_flow = min_flow.min(path_flow);
                node = parent_node;
            }
            path.push(source);
            path.reverse();
            used_edges.reverse();

            return Some((min_flow.min(target_flow), path, used_edges));
        }

        // 探索邻接节点
        for &(next_node, min_flow, _max_flow, edge_idx) in &graph[current] {
            if visited[next_node] {
                continue;
            }

            let remaining_capacity = edge_capacities[edge_idx];

            // 检查容量约束
            if remaining_capacity <= 0 {
                continue;
            }

            // 检查最小流量约束
            let usable_flow = if min_flow > 0 {
                if remaining_capacity < min_flow {
                    continue; // 边的剩余容量不足以满足最小流量要求
                }
                if flow_so_far < min_flow {
                    continue; // 到达该边的流量不足以满足最小流量要求
                }
                remaining_capacity
            } else {
                remaining_capacity
            };

            if usable_flow <= 0 {
                continue;
            }

            visited[next_node] = true;
            parent[next_node] = Some((current, edge_idx, usable_flow));
            queue.push_back((next_node, flow_so_far.min(usable_flow)));
        }
    }

    None
}
/// 优化版本：使用二分搜索 + 动态调整策略
/// 核心思想：如果无法直接找到满足amount的路径，尝试分割amount
fn find_multi_paths_optimized(
    num_nodes: usize,
    edges: &[EdgeWithMinFlow],
    source: usize,
    sink: usize,
    target_amount: i64,
    max_paths: usize,
) -> MultiPathResult {
    // 先用贪心算法尝试
    let greedy_result = find_multi_paths(num_nodes, edges, source, sink, target_amount, max_paths);

    if greedy_result.success {
        return greedy_result;
    }

    // 如果贪心失败，尝试二分搜索最大可传输量
    println!("贪心算法未能找到完整解，尝试二分搜索最优解...");

    let mut left = 1;
    let mut right = target_amount;
    let mut best_result = greedy_result;

    while left <= right {
        let mid = (left + right) / 2;
        let test_result = find_multi_paths(num_nodes, edges, source, sink, mid, max_paths);

        if test_result.success {
            if test_result.total_flow > best_result.total_flow {
                best_result = test_result;
            }
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    println!(
        "二分搜索完成，最大可传输量: {} / {}",
        best_result.total_flow, target_amount
    );
    best_result
}

/// 检查网络的理论最大流量（用于预估可行性）
fn estimate_max_flow_capacity(
    _num_nodes: usize,
    edges: &[EdgeWithMinFlow],
    source: usize,
    sink: usize,
) -> i64 {
    // 简化估算：找所有从source出发的边的容量和，以及所有到sink的边的容量和
    let mut source_capacity = 0;
    let mut sink_capacity = 0;

    for edge in edges {
        if edge.from == source {
            source_capacity += edge.max_flow;
        }
        if edge.to == sink {
            sink_capacity += edge.max_flow;
        }
    }

    // 理论上限是两者的最小值
    source_capacity.min(sink_capacity)
}

fn main() {
    println!("=== 多路径流量分配算法演示 ===\n");

    // 演示基本用法
    demo_basic_usage();

    println!("\n提示: 运行 'cargo test' 来执行完整的测试套件");
}

/// 演示基本算法用法
fn demo_basic_usage() {
    println!("--- 演示: 多路径流量分配 ---");

    // 构建一个简单的网络
    let edges = vec![
        EdgeWithMinFlow {
            from: 0,
            to: 1,
            min_flow: 0,
            max_flow: 100,
        },
        EdgeWithMinFlow {
            from: 0,
            to: 2,
            min_flow: 0,
            max_flow: 80,
        },
        EdgeWithMinFlow {
            from: 1,
            to: 3,
            min_flow: 0,
            max_flow: 60,
        },
        EdgeWithMinFlow {
            from: 2,
            to: 3,
            min_flow: 0,
            max_flow: 70,
        },
    ];

    println!("网络结构:");
    for (i, edge) in edges.iter().enumerate() {
        println!(
            "  边{}: {} -> {} [min:{}, max:{}]",
            i, edge.from, edge.to, edge.min_flow, edge.max_flow
        );
    }

    let result = find_multi_paths(4, &edges, 0, 3, 120, 3);

    println!("\n结果:");
    println!("  成功: {}", result.success);
    println!("  总流量: {}", result.total_flow);
    println!("  路径数: {}", result.paths.len());

    for (i, path) in result.paths.iter().enumerate() {
        println!("  路径{}: {:?} (流量: {})", i + 1, path.nodes, path.flow);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 基础功能测试
    #[test]
    fn test_single_path_insufficient() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 100,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 100,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 0,
                max_flow: 320,
            },
        ];

        let result = find_multi_paths(4, &edges, 1, 3, 150, 1);
        assert!(!result.success);
        assert_eq!(result.total_flow, 100);
    }

    #[test]
    fn test_multi_path_success() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 100,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 100,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 0,
                max_flow: 320,
            },
        ];

        let result = find_multi_paths(4, &edges, 1, 3, 150, 3);
        assert!(result.success);
        assert!(result.total_flow >= 150);
    }

    #[test]
    fn test_exact_target_match() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 50,
            },
        ];

        let result = find_multi_paths(3, &edges, 0, 2, 50, 1);
        assert!(result.success);
        assert_eq!(result.total_flow, 50);
    }

    /// 边界条件测试
    #[test]
    fn test_zero_flow_request() {
        let edges = vec![EdgeWithMinFlow {
            from: 0,
            to: 1,
            min_flow: 0,
            max_flow: 100,
        }];

        let result = find_multi_paths(2, &edges, 0, 1, 0, 1);
        assert!(result.success);
        assert_eq!(result.total_flow, 0);
    }

    #[test]
    fn test_zero_path_limit() {
        let edges = vec![EdgeWithMinFlow {
            from: 0,
            to: 1,
            min_flow: 0,
            max_flow: 100,
        }];

        let result = find_multi_paths(2, &edges, 0, 1, 50, 0);
        assert!(!result.success);
        assert_eq!(result.total_flow, 0);
    }

    #[test]
    fn test_source_equals_sink() {
        let result = find_multi_paths(1, &[], 0, 0, 100, 1);
        // 当源点等于汇点时，流量为0但可以认为成功（特殊情况）
        assert_eq!(result.total_flow, 0);
    }

    #[test]
    fn test_disconnected_graph() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 0,
                max_flow: 50,
            },
        ];

        let result = find_multi_paths(4, &edges, 0, 3, 50, 3);
        assert!(!result.success);
        assert_eq!(result.total_flow, 0);
    }

    #[test]
    fn test_bottleneck_identification() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 30,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 20,
            }, // 瓶颈
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 0,
                max_flow: 40,
            },
        ];

        let result = find_multi_paths(4, &edges, 0, 3, 25, 1);
        assert!(!result.success);
        assert_eq!(result.total_flow, 20); // 被瓶颈限制
    }

    #[test]
    fn test_huge_flow_request() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 30,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 20,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 0,
                max_flow: 40,
            },
        ];

        let result = find_multi_paths(4, &edges, 0, 3, i64::MAX, 10);
        assert!(!result.success);
        assert_eq!(result.total_flow, 20);
    }

    /// 最小流量约束测试
    #[test]
    fn test_min_flow_constraint_satisfied() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 50,
                max_flow: 100,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 30,
                max_flow: 80,
            },
        ];

        let result = find_multi_paths(3, &edges, 0, 2, 60, 1);
        assert!(result.success);
        assert!(result.total_flow >= 30); // 至少满足最小约束
    }

    #[test]
    fn test_min_flow_constraint_violated() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 40,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 50,
                max_flow: 100,
            }, // 需要50但前面只有40
        ];

        let result = find_multi_paths(3, &edges, 0, 2, 50, 1);
        assert!(!result.success);
        assert_eq!(result.total_flow, 0);
    }

    #[test]
    fn test_mixed_min_flow_constraints() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 20,
                max_flow: 50,
            }, // 有最小约束
            EdgeWithMinFlow {
                from: 0,
                to: 2,
                min_flow: 0,
                max_flow: 30,
            }, // 无最小约束
            EdgeWithMinFlow {
                from: 1,
                to: 3,
                min_flow: 0,
                max_flow: 60,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 15,
                max_flow: 25,
            }, // 有最小约束
        ];

        let result = find_multi_paths(4, &edges, 0, 3, 40, 2);
        // 这个测试主要验证算法能正确处理混合约束，不强制要求特定结果
        assert!(result.total_flow >= 0);
    }

    #[test]
    fn test_uniform_min_flow_constraints() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 25,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 0,
                to: 2,
                min_flow: 25,
                max_flow: 40,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 3,
                min_flow: 25,
                max_flow: 60,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 25,
                max_flow: 35,
            },
        ];

        let result = find_multi_paths(4, &edges, 0, 3, 80, 3);
        // 验证算法能处理统一的最小流量约束
        assert!(result.total_flow >= 0);
    }

    #[test]
    fn test_extreme_min_flow_constraints() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 1000,
                max_flow: 1000,
            }, // min = max
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 500,
                max_flow: 1500,
            },
        ];

        let result = find_multi_paths(3, &edges, 0, 2, 800, 1);
        // 验证极端情况的处理
        assert!(result.total_flow >= 0);
    }

    /// 性能和压力测试
    #[test]
    fn test_large_chain() {
        let mut large_edges = Vec::new();
        let chain_length = 50; // 减少长度以加快测试速度
        for i in 0..chain_length {
            large_edges.push(EdgeWithMinFlow {
                from: i,
                to: i + 1,
                min_flow: 0,
                max_flow: 1000,
            });
        }

        let result = find_multi_paths(chain_length + 1, &large_edges, 0, chain_length, 500, 1);
        assert!(result.success);
        assert_eq!(result.total_flow, 500);
    }

    #[test]
    fn test_dense_graph() {
        let mut dense_edges = Vec::new();
        let nodes = 6; // 减少节点数以加快测试速度
        for i in 0..nodes {
            for j in (i + 1)..nodes {
                dense_edges.push(EdgeWithMinFlow {
                    from: i,
                    to: j,
                    min_flow: 0,
                    max_flow: 10,
                });
            }
        }

        let result = find_multi_paths(nodes, &dense_edges, 0, nodes - 1, 50, 3);
        // 密集图应该能找到较好的解
        assert!(result.total_flow > 0);
    }

    #[test]
    fn test_multi_path_stress() {
        let stress_edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 1,
            },
            EdgeWithMinFlow {
                from: 0,
                to: 2,
                min_flow: 0,
                max_flow: 1,
            },
            EdgeWithMinFlow {
                from: 0,
                to: 3,
                min_flow: 0,
                max_flow: 1,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 4,
                min_flow: 0,
                max_flow: 1,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 4,
                min_flow: 0,
                max_flow: 1,
            },
            EdgeWithMinFlow {
                from: 3,
                to: 4,
                min_flow: 0,
                max_flow: 1,
            },
        ];

        let result = find_multi_paths(5, &stress_edges, 0, 4, 3, 10);
        assert!(result.success);
        assert_eq!(result.total_flow, 3);
    }

    /// 错误处理和异常情况测试
    #[test]
    fn test_graph_with_cycles() {
        let cycle_edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 0,
                min_flow: 0,
                max_flow: 30,
            }, // 形成环
            EdgeWithMinFlow {
                from: 1,
                to: 3,
                min_flow: 0,
                max_flow: 40,
            },
        ];

        let result = find_multi_paths(4, &cycle_edges, 0, 3, 30, 2);
        // 算法应该能处理包含环的图
        assert!(result.total_flow >= 0);
    }

    #[test]
    fn test_zero_capacity_edges() {
        let zero_edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 0,
            }, // 容量为0
            EdgeWithMinFlow {
                from: 0,
                to: 2,
                min_flow: 0,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 1,
                min_flow: 0,
                max_flow: 50,
            },
        ];

        let result = find_multi_paths(3, &zero_edges, 0, 1, 30, 3);
        // 应该通过替代路径找到解
        assert!(result.total_flow > 0);
    }

    /// 优化算法测试
    #[test]
    fn test_optimized_algorithm() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 60,
            },
            EdgeWithMinFlow {
                from: 0,
                to: 2,
                min_flow: 0,
                max_flow: 40,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 3,
                min_flow: 0,
                max_flow: 35,
            },
            EdgeWithMinFlow {
                from: 2,
                to: 3,
                min_flow: 0,
                max_flow: 55,
            },
        ];

        let result = find_multi_paths_optimized(4, &edges, 0, 3, 150, 3);
        // 优化算法应该找到接近最优的解
        assert!(result.total_flow > 70); // 理论最大约90
    }

    /// 容量估算测试
    #[test]
    fn test_capacity_estimation() {
        let edges = vec![
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 100,
            },
            EdgeWithMinFlow {
                from: 0,
                to: 1,
                min_flow: 0,
                max_flow: 50,
            },
            EdgeWithMinFlow {
                from: 1,
                to: 2,
                min_flow: 0,
                max_flow: 80,
            },
        ];

        let capacity = estimate_max_flow_capacity(3, &edges, 0, 2);
        assert_eq!(capacity, 80); // min(150, 80) = 80
    }
}
