use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct InputEdge {
    pub from: usize,
    pub to: usize,
    pub min_flow: i64,
    pub max_flow: i64,
}

fn main() {
    let num_nodes = 10;
    let source = 9;
    let target = 0;
    
    let edges = vec![
        InputEdge {
            from: 0,
            to: 1,
            min_flow: 0,
            max_flow: 10,
        },
        InputEdge {
            from: 1,
            to: 2,
            min_flow: 0,
            max_flow: 10,
        },
    ];
    
    println!("Testing disconnected case: {} nodes, source={}, target={}", num_nodes, source, target);
    println!("Edges: {:?}", edges);
    
    // 检查连通性
    let mut adj = vec![Vec::new(); num_nodes];
    for edge in &edges {
        adj[edge.from].push(edge.to);
    }
    
    // BFS from source
    let mut visited = vec![false; num_nodes];
    let mut queue = VecDeque::new();
    queue.push_back(source);
    visited[source] = true;
    
    while let Some(node) = queue.pop_front() {
        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    
    println!("Can reach target {}: {}", target, visited[target]);
    
    if !visited[target] {
        println!("Graph is disconnected - source {} cannot reach target {}", source, target);
    } else {
        println!("Graph is connected");
    }
}
