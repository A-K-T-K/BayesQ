import sys
import time
import csv
import psutil
import logging
import itertools
import gc
import numpy as np
import networkx as nx

# ==========================================
# CONFIGURATION
# ==========================================
MIN_NODES = 3
MAX_NODES = 20
DENSITY = 0.4              # Probability of edge creation (0.4 = moderately dense DAG)
GRAPHS_PER_SIZE = 5        # Number of unique random topologies to test per size
REPEATS_PER_GRAPH = 5      # Number of timing runs per topology (to average out noise)
OUTPUT_FILENAME = "benchmark_results.csv"
# ==========================================

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Benchmark")

try:
    from qiskit import QuantumCircuit, QuantumRegister
except ImportError:
    print("Please install requirements: pip install qiskit networkx psutil")
    sys.exit(1)

class RobustBenchmark:
    def __init__(self):
        self.process = psutil.Process()

    def generate_random_dag(self, n_nodes, density=0.4):
        """Generates a random DAG ensuring topological ordering."""
        G = nx.DiGraph()
        nodes = [f"N{i}" for i in range(n_nodes)]
        G.add_nodes_from(nodes)
        
        # Create edges only from lower index to higher index to guarantee DAG
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < density:
                    G.add_edge(nodes[i], nodes[j])
        return G

    def generate_dummy_cpts(self, G):
        """Generates random CPTs for the given graph structure."""
        node_data = {}
        for node in G.nodes():
            num_parents = len(list(G.predecessors(node)))
            # Generate random probabilities for 2^k rows
            # We generate 2*2^k random numbers
            cpt_raw = np.random.rand(2**num_parents * 2)
            cpt = cpt_raw.tolist()
            
            # Normalize pairs [P(0), P(1)] to sum to 1.0
            for i in range(0, len(cpt), 2):
                total = cpt[i] + cpt[i+1]
                cpt[i] = cpt[i] / total
                cpt[i+1] = cpt[i+1] / total
                
            node_data[node] = {'states': ['0', '1'], 'cpt': cpt}
        return node_data

    def build_circuit_headless(self, G, node_data):
        """
        Core circuit synthesis logic. 
        This mirrors the QBNApp.build_qbayesian_circuit method exactly
        but strips away all GUI/Matplotlib overhead.
        """
        sorted_nodes = list(nx.topological_sort(G))
        node_name_to_idx = {name: i for i, name in enumerate(sorted_nodes)}
        
        qr_list = [QuantumRegister(1, name=n) for n in sorted_nodes]
        qc = QuantumCircuit(*qr_list)
        flat_qubits = [qr[0] for qr in qr_list]
        
        def qb(name): return flat_qubits[node_name_to_idx[name]]

        for i, node in enumerate(sorted_nodes):
            parents = sorted(list(G.predecessors(node)))
            cpt = node_data[node]['cpt']
            
            if not parents:
                # Root Node
                theta = 2 * np.arcsin(np.sqrt(cpt[1]))
                qc.ry(theta, flat_qubits[i])
            else:
                # Child Node
                p_states = [node_data[p]['states'] for p in parents]
                combos = list(itertools.product(*p_states))
                parent_qubits = [qb(p) for p in parents]
                
                # Iterate in reverse (gray code-like traversal)
                for combo_idx in range(len(combos) - 1, -1, -1):
                    combo = combos[combo_idx]
                    flipped = []
                    
                    # 1. Apply X gates to set controls
                    for p, s in zip(parents, combo):
                        if s == node_data[p]['states'][0]:
                            qc.x(qb(p))
                            flipped.append(p)
                            
                    # 2. Apply Rotation
                    start_idx = combo_idx * 2
                    theta = 2 * np.arcsin(np.sqrt(cpt[start_idx + 1]))
                    
                    if len(parents) == 1:
                        qc.cry(theta, parent_qubits[0], flat_qubits[i])
                    else:
                        qc.mcry(theta, parent_qubits, flat_qubits[i])
                        
                    # 3. Uncompute X gates
                    for p in reversed(flipped):
                        qc.x(qb(p))
        return qc

    def run(self):
        print(f"Running Robust Benchmark (Density={DENSITY}, Graphs/Size={GRAPHS_PER_SIZE}, Repeats={REPEATS_PER_GRAPH})...")
        print("-" * 65)
        print(f"{'Nodes':<6} | {'Avg Time (s)':<12} | {'Std Dev (s)':<12} | {'Avg Mem (MB)':<12}")
        print("-" * 65)
        
        results = []

        for n in range(MIN_NODES, MAX_NODES + 1):
            times = []
            mems = []
            
            # 1. Test across different graph topologies
            for _ in range(GRAPHS_PER_SIZE):
                G = self.generate_random_dag(n, density=DENSITY)
                data = self.generate_dummy_cpts(G)
                
                # 2. Repeat measurement for stability on THIS graph
                for _ in range(REPEATS_PER_GRAPH):
                    # Force GC before timing to prevent cleanup spikes affecting data
                    gc.collect()
                    
                    # Measure Memory Baseline
                    mem_before = self.process.memory_info().rss
                    
                    # Measure Time
                    start = time.perf_counter()
                    qc = self.build_circuit_headless(G, data)
                    end = time.perf_counter()
                    
                    # Measure Memory After
                    mem_after = self.process.memory_info().rss
                    
                    times.append(end - start)
                    mems.append(mem_after / 1024 / 1024) # Convert to MB

            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_mem = np.mean(mems)
            
            print(f"{n:<6} | {avg_time:<12.5f} | {std_time:<12.5f} | {avg_mem:<12.2f}")
            
            results.append({
                "nodes": n,
                "qubits": n,
                "avg_time": avg_time,
                "std_dev": std_time,
                "avg_mem": avg_mem
            })
            
        return results

    def save(self, results):
        try:
            with open(OUTPUT_FILENAME, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["nodes", "qubits", "avg_time", "std_dev", "avg_mem"])
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"\nSuccess! Results saved to {OUTPUT_FILENAME}")
        except IOError as e:
            logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    bm = RobustBenchmark()
    data = bm.run()
    bm.save(data)