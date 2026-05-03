import os
import re
import networkx as nx
import matplotlib.pyplot as plt

# Mapping of core architectural types to the file that defines their concept
TYPE_TO_FILE = {
    "llama_model": "llama-model.h",
    "llama_adapter_lora": "llama-adapter.h",
    "llama_adapter_cvec": "llama-adapter.h",
    "llama_context": "llama-context.h",
    "llama_memory_i": "llama-memory.h",
    "llama_memory_context_i": "llama-memory.h",
    "llama_ubatch": "llama-batch.h",
    "llama_cparams": "llama-cparams.h",
    "llama_hparams": "llama-hparams.h",
    "llama_vocab": "llama-vocab.h",
    "llama_layer": "llama-model.h",
    "llm_graph_context": "llama-graph.h",
    "llm_graph_params": "llama-graph.h",
    "llama_kv_cache_context": "llama-kv-cache.h"
}

FILES_TO_SCAN = [
    "src/llama-adapter.h", "src/llama-batch.h", "src/llama-context.h", "src/llama-cparams.h",
    "src/llama-ext.h", "src/llama-graph.h", "src/llama-hparams.h", "src/llama-memory.h",
    "src/llama-model-loader.h", "src/llama-model-saver.h", "src/llama-model.h",
    "src/llama-vocab.h", "src/llama.cpp"
]

def find_logical_dependencies():
    G = nx.DiGraph()
    
    # Add nodes
    for file_path in FILES_TO_SCAN:
        basename = os.path.basename(file_path)
        G.add_node(basename)
        
    edges_found = set()
    
    for file_path in FILES_TO_SCAN:
        if not os.path.exists(file_path):
            continue
            
        basename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Remove comments to avoid false positives in documentation
            content = re.sub(r'//.*', '', content)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            
            for type_name, target_file in TYPE_TO_FILE.items():
                if target_file == basename:
                    continue # Skip self-dependencies
                    
                # Look for the type name as a distinct word
                pattern = r'\b' + re.escape(type_name) + r'\b'
                if re.search(pattern, content):
                    G.add_edge(basename, target_file)
                    edges_found.add((basename, target_file, type_name))

    print("=== DÉPENDANCES LOGIQUES TROUVÉES ===")
    for src, dst, tname in sorted(edges_found):
        print(f"{src} utilise '{tname}' -> dépend logiquement de {dst}")
        
    print("\n=== RECHERCHE DE BOUCLES LOGIQUES ===")
    cycles = list(nx.simple_cycles(G))
    if not cycles:
        print("Aucune boucle logique trouvée (Étrange...)")
    else:
        print(f"{len(cycles)} boucles logiques trouvées !")
        for i, c in enumerate(cycles):
            print(f"Boucle {i+1}: {' <-> '.join(c)} <-> {c[0]}")
            
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Use spring layout but with high repulsion to separate nodes
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Identify nodes in cycles for coloring
    nodes_in_cycles = set()
    for c in cycles:
        nodes_in_cycles.update(c)
        
    node_colors = ['#ff6666' if node in nodes_in_cycles else '#99ccff' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9, edgecolors='black')
    
    # Draw edges, make cycle edges bold and red
    edge_colors = []
    edge_widths = []
    
    # Create a set of cycle edges for quick lookup
    cycle_edges = set()
    for c in cycles:
        for i in range(len(c)):
            cycle_edges.add((c[i], c[(i+1)%len(c)]))
            
    for u, v in G.edges():
        if (u, v) in cycle_edges:
            edge_colors.append('red')
            edge_widths.append(2.5)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.0)
            
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=20, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Graphe des Dépendances LOGIQUES (Couplage Fort)\nEn rouge: Les fichiers impliqués dans des boucles", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('logical_dependency_graph.png', dpi=300, bbox_inches='tight')
    print("\n[+] Graphe visuel des dépendances LOGIQUES sauvegardé : logical_dependency_graph.png")

if __name__ == '__main__':
    find_logical_dependencies()
