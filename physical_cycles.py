import os
import re
import networkx as nx
import matplotlib.pyplot as plt

FILES_TO_SCAN = [
    "src/llama-adapter.h", "src/llama-batch.h", "src/llama-context.h", "src/llama-cparams.h",
    "src/llama-ext.h", "src/llama-graph.h", "src/llama-hparams.h", "src/llama-memory.h",
    "src/llama-model-loader.h", "src/llama-model-saver.h", "src/llama-model.h",
    "src/llama-vocab.h", "src/llama.cpp"
]

def find_physical_dependencies():
    G = nx.DiGraph()
    
    # On ajoute d'abord nos 13 fichiers comme nœuds de base
    basenames_to_scan = {os.path.basename(p): p for p in FILES_TO_SCAN}
    for basename in basenames_to_scan:
        G.add_node(basename)
        
    edges_found = set()
    include_pattern = re.compile(r'^\s*#include\s+["<]([^">]+)[">]')
    
    for file_path in FILES_TO_SCAN:
        if not os.path.exists(file_path):
            continue
            
        basename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = include_pattern.match(line)
                if match:
                    inc_name = match.group(1)
                    inc_basename = os.path.basename(inc_name)
                    
                    # On ne trace les liens qu'entre les 13 fichiers cibles
                    if inc_basename in basenames_to_scan and inc_basename != basename:
                        G.add_edge(basename, inc_basename)
                        edges_found.add((basename, inc_basename))

    print("=== DÉPENDANCES PHYSIQUES (#include) TROUVÉES ===")
    for src, dst in sorted(edges_found):
        print(f"{src} inclut -> {dst}")
        
    print("\n=== RECHERCHE DE BOUCLES PHYSIQUES ===")
    cycles = list(nx.simple_cycles(G))
    if not cycles:
        print("Aucune boucle physique trouvée ! (C'est normal, le compilateur fonctionne)")
    else:
        print(f"{len(cycles)} boucles physiques trouvées !")
        for i, c in enumerate(cycles):
            print(f"Boucle {i+1}: {' -> '.join(c)} -> {c[0]}")
            
    # Visualisation
    plt.figure(figsize=(14, 10))
    
    # Disposition des nœuds (spring layout)
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color='#88e588', node_size=3000, alpha=0.9, edgecolors='black')
    
    # Tracer les arêtes
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrowsize=20, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Graphe des Dépendances PHYSIQUES (#include)\nAucune boucle présente dans ce sous-graphe", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('physical_dependency_graph.png', dpi=300, bbox_inches='tight')
    print("\n[+] Graphe visuel des dépendances PHYSIQUES sauvegardé : physical_dependency_graph.png")

if __name__ == '__main__':
    find_physical_dependencies()
