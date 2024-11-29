# Graph Homophily Analysis with Torch Geometric

This Python script provides a set of functions for analyzing graph datasets, specifically focusing on evaluating and adjusting homophily (the tendency for nodes with the same label to be connected) in graph-based machine learning models. The code is designed to compute various homophily metrics, including both homogeneity and heterogeneity, while leveraging the `torch_geometric` library to manipulate and analyze graph structures.

Key operations include calculating homophily ratios, entropy-based edge label information, neighborhood similarity measures, and computing compatibility matrices based on class labels.

## Key Functions

### 1. Homophily Metrics:

- **`compute_graph_hete`**: Computes the homophily ratio for a graph (based on whether nodes connected by edges have the same label).
- **`compute_homo_mask` and `compute_homo_mask_new`**: Compute homophily masks based on different criteria, potentially differentiating between homogeneous and heterogeneous graphs.
- **`compute_homo_ratio`**: Computes the homophily ratio for nodes in the graph.
- **`compute_homo_ratio_new`**: Calculates the homophily ratio based on higher-order neighbors (i.e., nodes up to a specified number of hops away).
- **`compute_higher_order_homo_ratio`**: Computes homophily ratios by considering higher-order neighbors (multi-hop neighbors).

### 2. Edge and Node Adjustments:

- **`compute_edge_label_inform`**: Measures the entropy and conditional entropy of edges based on class labels and computes an information score.
- **`adjust_homo_ratio`**: Adjusts homophily ratios by considering the impact of node degrees and edge label information.
- **`compute_node_adjust_homo`**: Adjusts the homophily ratio at the node level, factoring in both the degree and label distribution.

### 3. Neighborhood and Label Analysis:

- **`find_neighbor_hist`**: Computes the histogram of labels for neighbors of a specific node in the graph.
- **`CCNS`**: Computes Cross-Class Neighborhood Similarity (CCNS), as defined in the referenced paper, between nodes of different classes.
- **`category_node_by_label`**: Classifies nodes based on their labels.
- **`cross_class_neighborhood_similarity`**: Computes neighborhood similarity between nodes of different classes.

### 4. Compatibility Matrix:

- **`compat_matrix`**: Computes the class compatibility matrix, which measures how compatible the classes are for nodes connected by edges in the graph.
- **`class_homo`**: Measures the homophily of each class by examining its compatibility with other classes in the graph.

### 5. Statistical Comparison and Visualization:

- **`compare_distribution`**: Compares distributions of values across different datasets and visualizes them using histograms.
- **`probs_correlation`**: Computes the correlation between probability distributions of different classes.
- **`class_homo`**: Calculates class homophily and measures the degree of homophily across different classes in the graph.
