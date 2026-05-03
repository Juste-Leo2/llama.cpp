# Analyse des Dépendances Architectures : Le Couplage Fort

## 1. La différence fondamentale

Dans un projet C/C++, il existe deux types de dépendances entre vos fichiers.

### A. La dépendance "Physique" (Les `#include`)
C'est ce que notre premier script a cherché. Le fichier A contient `#include "B.h"`. 
Si A inclut B, et B inclut A, le compilateur **plante** (car il tourne en boucle pour inclure le texte). 
> [!SUCCESS]
> **Bonne nouvelle :** Notre script a prouvé qu'il n'y a **zéro boucle physique** parmi les 13 fichiers de `llama.cpp`. Ce problème n'existe pas, le compilateur s'en sort très bien.

### B. La dépendance "Logique" (L'Architecture Spaghettis)
C'est ce que **Sentrux** détecte, et ce que notre nouveau script vient de cartographier.
Une dépendance logique se produit lorsqu'une classe/structure a **conscience** de l'existence d'une autre, même si on contourne le `#include` avec une *forward declaration* (ex: `struct llama_context;`).

Si le `Modele` connaît l'`Adaptateur`, et que l'`Adaptateur` connaît le `Modele`, ils sont **couplés fort**. 
Même si le compilateur ne plante pas, pour un architecte logiciel (et pour Sentrux), c'est une dépendance circulaire. Vous ne pouvez pas réutiliser, tester ou modifier l'un sans embarquer l'autre.

---

## 2. Ce qu'il se passe dans `llama.cpp`

Notre nouveau script vient de révéler pas moins de **20 boucles logiques** entre les 13 fichiers cœurs. 
Voici les nœuds critiques (le "plat de spaghettis") :

> [!WARNING]
> **Le Cycle Modèle/Adaptateur**
> - `llama-model.h` contient un pointeur vers ses adaptateurs : `std::unordered_set<llama_adapter_lora *> loras;`
> - `llama-adapter.h` contient un pointeur vers son modèle : `llama_model * model = nullptr;`
> **Résultat :** Ils sont inséparables.

> [!WARNING]
> **Le Cycle Mémoire/Contexte/Batch**
> - `llama-memory.h` a besoin du contexte pour s'initialiser : `init_update(llama_context * lctx)`
> - `llama-context.h` a besoin de la mémoire pour fonctionner.
> - Et la mémoire a aussi besoin de `llama_ubatch`, tandis que le batch référence la mémoire.

---

## 3. Pourquoi est-ce "grave" et comment le résoudre ?

Ce n'est pas "mortel" pour le fonctionnement actuel du programme. Le code s'exécute très bien.
Cependant, pour l'**évolution du projet**, c'est toxique. Ajouter une nouvelle fonctionnalité au Modèle risque de nécessiter de modifier le Contexte, l'Adaptateur, la Mémoire et le Graphe. C'est pour bloquer cette dégénérescence de l'architecture que la règle Sentrux `max_cycles = 0` existe.

### L'intervention nécessaire ("Chirurgie Architecturale")

On ne peut pas juste effacer une ligne. Il faut changer la conception pour **casser la boucle** :

1. **Supprimer les pointeurs de retour inutiles :** 
   Par exemple, pourquoi `llama_adapter_lora` doit-il stocker un `llama_model *` en permanence ? Ne pourrions-nous pas simplement lui passer le `llama_model` en argument uniquement au moment où la fonction `apply()` est appelée ?
2. **Utiliser des identifiants au lieu de pointeurs :**
   Plutôt que le Modèle stocke une liste de pointeurs vers les Adaptateurs, il pourrait juste stocker des `ID` d'adaptateurs (entiers), brisant ainsi la dépendance vers le type lui-même.
3. **Extraire des interfaces pures :**
   Si la mémoire a besoin du contexte, peut-être qu'elle n'a besoin que d'une toute petite partie des données du contexte. On pourrait créer une structure `llama_memory_params` indépendante.

Une fois que nous sommes d'accord sur le constat et la vision, nous pourrons établir un plan technique précis (`implementation_plan.md`) pour casser le premier cycle majeur (Modèle / Adaptateur) afin d'isoler ces composants.


# draft: refactor(llama): break logical cycles between model, context and adapter