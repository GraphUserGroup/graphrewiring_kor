# Spectral-Background

`**Introduction**`

본 섹션에서는 Graph rewiring 사용되는 기본 개념인 Spectral Background에 대해 알아보겠습니다. 본 튜토리얼의 핵심을 다룬다기보다, 1.왜 이런 문제가 제기되었는지 2.문제를 해결하기 위해 튜토리얼 저자들은 어떤 방식으로 접근했는지 3.솔루션 아이디어를 구현하기 위해 사용되는 재료 총 3가지에 대해 살펴보는게 본 섹션의 주 목적입니다.

## Overview

- GNN 고질적인 문제 ‘over-smoothing’ , ‘over-squashing’에 대해 알아봅니다.
- Graph Rewiring 에 대해 알아보며, 과연 이 방법론이 정보 전달 비대칭성을 개선하는데 도움이 되는가에 대해서 알아봅니다.
- Diffwire 의 핵심 요소인 CT-LAYER, GAP-LAYER의 재료들이 무엇인지 알아봅니다.

---

`Motivation`

over-squashing 의 현상을 rewiring 으로 하는게 본 튜토리얼 아이디어의 핵심입니다. graph topology 에 비례하여 message-passing이 발생하고, 그에따라 정보전달이 발생합니다. 이때 정보전달을 어떤식으로 하는지에 따라 GNN의 구조가 달라집니다. 본 섹션에서는 GNN 대표적인 모델 3가지를 시작으로 레이어 수에 따라 얼마나 성능이 달라지는가에 대한 이야기를 해보겠습니다.

# GNN principle

→ 연결된 노드들로부터 Message Passing 을 받아 Aggregation 한 정보의 Smoothness를 기반으로 weight 가 갱신됨. 즉, graph feature 와 graph topology 두 가지에 따라 weight 가 변하며, 해당 weight를 기반으로 추론을 진행함.

![Thomas, Josephine M., et al. "Graph neural networks designed for different graph types: A survey." *arXiv preprint arXiv:2204.03080* (2022).](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf54f5ce-c5df-4b45-a831-35ac4d405d0d/Untitled.png)

Thomas, Josephine M., et al. "Graph neural networks designed for different graph types: A survey." *arXiv preprint arXiv:2204.03080* (2022).

크게 convolutional , attentional , message-passing으로 나뉘며, 본 튜토리얼에선 Message-passing 의 공식을 활용한 MPNN 레이어를 대상으로 진행함.

**단순하게 접근해보면, 정보가 많을수록 추론성능이 높아진다 라는 생각을 할 수 있기에 GNN layer를 여러개 쌓자. 라는 생각을 할 수 있습니다. 과연 그럴까요?**

# GNN layer increasing are good or bad?

그래프 토폴로지(구조)를 측정한 뒤, 이를 고려해서 GNN 층을 쌓는게 합리적인 프로세스라 할 수 있습니다. 바로 그래프 데이터의 고유성격인 ‘연결’이 반영된 데이터라는 특성 때문인데요. 타 데이터(tabular , image , text) 들과는 다르게 데이터 다수가 power-law 분포를 띕니다. 다시 말해서, 빈익빈 부익부 데이터들이 많습니다. 특정 노드에 많은 엣지가 연결되어 있다면, 그에 비례하여 많은 정보량이 전송되어 특정 노드만의 고유 정보를 잃게 될 수 있습니다. 반대로, 특정 노드에 적은 엣지가 연결되어 있다면 상대적으로 적은 정보만 전송받기에 특정 노드만의 고유 정보가 지배적일 수 있습니다. 그렇다면 추론시에 다른 노드들의 정보가 많이 반영되었거나, 본인 노드만의 정보가 많이 반영되었거나 둘 중 하나의 경우가 발생하게 됩니다. 결국 ‘적절한’ 정보 전달(message passing)이 핵심이며, 이를 개선하고자 다양한 연구들이 시도되고 있습니다. 

개인적으로는 GNN 모델링 전, 다음 3가지 분포를 기본적으로 측정해보신 후 의사결정을 하시는걸 추천합니다.

1. graph diameter(그래프 구조 크기) , 2. degree distribution(연결 분포) , 3. assortativity coefficient(레이블 분포)

위 내용에 대해 더 궁금하시다면, 아래 논문을 참조하시면 많은 도움이 될 거라 생각합니다.

Deac, Andreea, Marc Lackenby, and Petar Veličković. "Expander graph propagation." *Learning on Graphs Conference*. PMLR, 2022. ****[[paper link](https://proceedings.mlr.press/v198/deac22a/deac22a.pdf)] 

![Alon, U. and Yahav, E. “On the bottleneck of graph neural networks and its practical implications”. In ICLR 2021](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/164f8188-876e-4576-b7b8-3eb192102e6e/Untitled.png)

Alon, U. and Yahav, E. “On the bottleneck of graph neural networks and its practical implications”. In ICLR 2021

where *k* = n layers , *r* = radius , *d* = graph diameter. 그래프 본질적인 특성을 파악해보고,그에 타당한 레이어를 쌓아야 효율적인 학습을 할 수 있습니다. 여기에서는 *r*,*d* 그래프 지름을 그 지표로 활용합니다. **(over-smoothing)** 이를 고려치 않고 막무가내로 레이어를 증가(k)시킨다면 오히려 smoothness 가 과도하게 증가하여 노드들을 구별하기 어려워집니다. 즉, message passing 으로 전달된 정보들이 다양성이 적어지기에 generalization 에서 불리한 상황 발생하는 거죠. **(over-squashing)** 또한 , 그래프 내 특정 노드(bottleneck)에 과도한 정보가 전달되어 그래프 내 정보교류가 고루 분포되지 않음.

> To capture long-range dependencies between nodes, GNNs must perform at least as many message-passing steps (i.e., have as many layers) as the distance between node pairs to not incur in under-reaching. However, building deep GNNs presents an inherent challenge. As depth increases, the receptive field of nodes grows exponentially, thus requiring more information to be encoded in the same fixed-size vectors.
> 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b2871568-6fad-4945-af56-fbf06dde9f52/Untitled.png)

> Over-squashing arises when the derivative in above equation result becomes too small, indicating that the representation of node v is mostly insensitive to the information initially present at node u. While increasing the layer Lipschitz constants or the dimension H can mitigate the issue , this may come at the expense of model generalization . Therefore, different methods have been proposed to alter the graph topology in a more favorable way to message-passing.
> 

- Lipschitz constants : it measures the rate of change of a function as we move between adjacent vertices in the graph.

---

`**Background**`

# Rewiring methods

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9ed110c6-ab16-45be-af91-4f9be47568bc/Untitled.png)

**Heat** **kernel**[diffusion based] 

A 는 Transition matrix 를 의미합니다.  parameter 로 heat equation(t) 를 활용합니다. 

$$
M_{heat} = e^{-t\mathbf{A}}.
$$

**Threshold**

$$
\theta^{Heat}_{m}=e^{-t}\frac{1}{m!} \ ,t>0.
$$

시간이 변함에 따라 information flow 가 어떻게 변하는지 측정하기 위해 heat 요소를 threshold 로 활용합니다. 

**PageRank** [diffusion based] 

A 는 Transition matrix 를 의미합니다. parameter 로 \alpha 를 활용합니다. 

$$
M_{PageRank} = \alpha(1 − (1 − \alpha)\mathbf{A}^+.
$$

**Threshold**

$$
\theta^{Pagerank}_{m}=\alpha(1-\alpha)^m,0 < \alpha < 1.
$$

\alpha를 사전에 지정해줍니다. 특정 노드가 연결된 타 노드로 이동할것인지에 대한 여부를 확률적으로 계산할 때 \alpha 값에 영향을 받습니다. 

**SDRF** [curvature based] iteratively samples an edge (u, v) proportionally to how negatively curved it is, *then adds the new edge (u′, v′) able to provide the largest increase of Ricuv*. (The algorithm optionally removes the most positively curved edges to avoid growing the graph excessively.)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce2def8e-ee95-474b-9d55-f682c78ffe56/Untitled.png)

**GRLEF** [curvature based] proposes to improve a graph spectral gap by working exclusively locally via the triangle counts ♯△uv, which are cheaper to compute as they require just neighborhood information. The algorithm iteratively samples an edge (u, v) proportionally to the inverse of triangle count, that is from an area of the graph that is locally far away from being fully-connected. Then *it chooses the pair of edges (u, u′),(v, v′) to flip into (u, v′),(v, u′) which provides the smallest net change in triangle count. This behavior can be interpreted as mitigating a very low local curvature* (as suggested by the small term ♯△uv in Ricuv) at the expense of a reduction in curvature of more positively curved neighboring edges. Banerjee et al. [7] supported the approach of their rewiring algorithm by empirically finding a correspondence between triangle count decrease and spectral gap increase.

![GRLEF algorithm](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a73abb8-301a-4e28-af1e-c10b9626a687/Untitled.png)

GRLEF algorithm

**EGP** [curvature based & messpage passing] are derived from the Cayley graphs of finite groups SL(2, Zn), which are 4-regular and thus guarantee sparsity. Interestingly, these graphs have all negatively curved edges with Ric_uv = −1/2. In our experiments we will thus use the message-passing matrix M_EGP = A_Cay A, where A_Cay is the adjacency matrix of said Cayley graphs.

![cayley graph](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2dca2c7-ad66-44e0-b94b-30a9c43b8e8d/Untitled.png)

cayley graph

![EGP algorithm](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ae18d782-4fb0-4d2e-a180-62ff5df55ca4/Untitled.png)

EGP algorithm

**DiffWire**(Hybrid , curvature and diffusion) , minimize the structure transformation but keep the information by sparsification(using the CT , GAP layer). 

> Our aim is to leverage **diffusion and curvature theories** to propose a new approach for graph rewiring that preserves the graph’s structure.
> 

DiffWire ‘diffusion’ and ‘curvature’ theories 의 기본이 되는 개념 spectral theory 에 대해 알아야합니다. 위에 나열된 방법론들에 대한 자세한 설명은 2️⃣ Transductive rewiring 에서 이야기 하겠습니다.

---

# Bottleneck measurement

Sparifiaction 를 위해서 그래프 구조 중 어느 노드에서 bottleneck 이 발생하는지 파악해야합니다. 크게 local bottleneck, global bottleneck measurement 로 나뉩니다. 

* Sparsification refers to the process of transforming a given graph into a sparser graph while preserving certain structural properties or characteristics. A sparser graph has fewer edges than the original graph, which can be beneficial for various computational and analytical purposes.

**Local bottleneck**

Intuitively, for a tree the receptive field grows exponentially in the branching factor, while at the other opposite a complete graph has a constant receptive field. To provide a metric to quantify this local behavior, they have proposed the balanced Forman curvature, defined as

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b6297473-e863-4119-a24c-dd59d4108298/Untitled.png)

SDRF [52] iteratively samples an edge (u, v) proportionally to how negatively curved it is, then adds the new edge (u′, v′) able to provide the largest increase
of Ricuv. (The algorithm optionally removes the most positively curved edges to avoid growing the graph excessively.)

**Global bottleneck**

A more global metric is the **Cheeger constant** ‘h_G’, which quantifies the minimum fraction of edges that need to be removed in order to make the graph disconnected. A small ‘h_G’ thus indicates that few edges act as a bridge between two otherwise disconnected communities. However, computing the Cheeger constant is an NP-hard problem, so the lower bound given by the spectral gap λ1 (i.e. the smallest positive Laplacian eigenvalue) is used as a proxy measure in practice: hG ≥ 1/2 λ1 . 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e0a4be52-e5d9-409a-a88f-099f812e406f/Untitled.png)

> → having a small Cheeger constant is equivalent to the graph having a ‘bottleneck’. in the sense that there is a collection of edges BA that, when removed, disconnects the vertices into two sets (A and its complement, V pGqzA), with the property that the sizes of A and its complement are significantly larger than the size of BA. -Expander grpah propagation.
> 

**Effective resistance.**

> Effective resistance [14] provides an additional way to measure bottlenecks in graph topology. **The resistance Res_uv between two nodes is proportional to the commute time Com_uv, which is the number of expected steps for a random walk to go back and forth between nodes u, v. An high resistance between two nodes is an indication of the difficulty for messages to pass from node u to node v.** Black et al. [9] proved a sensitivity bound similar to (2) relating high effective resistance Resvu between pairs of nodes to a reduced sensitivity of the representations h(L) v to input features xu. Furthermore, effective resistance is inversely related to the square of the Cheeger constant by the inequality max(u,v)∈E Res_uv ≤ 1h2G[4]. ***Arnaiz-Rodríguez et al. [4] have proposed a layer for learning effective resistance to re-weight the original graph adjacency (hence ‘DiffWire’) in the perspective of sampling a spectrally similar but sparser graph which preserves the graph structural information [47].*** The additional intuitive effect is to enlarge the relative capacity of high resistance edges, which correspond to bridges over more densely connected communities. In our experiments we implement the DiffWire approach by computing the effective resistance in exact form by Resuv = (1u − 1v)⊤L+(1u − 1v) with 1u the indicator vector of node u. The resulting message-passing matrix therefore is MDiffWire = Res ⊙ A, where ‘⊙’ denotes the elementwise product.
> 

---

# Bottleneck solution , curvature vs. diffusive.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/efc6c9e6-7140-4e40-ace7-9ae093dd7b1a/Untitled.png)

- diffusion works better in homophilic graphs. Needs parameters \alpha (or t) and \epsilon
- SDRF works better in hetrophilic graphs Needs parameters \tau and C^+

→ 위 두 방법 모두 ‘parameter’가 필요함. 다르게 말하면 parameter 에 따라 성능 의존성이 강하므로 unseen task 인 generalization performance 에서의 성능을 보장하기에는 어렵다. 라고 볼 수 있음. 하지만 본 방법론 Diffwire는 Parameter-free methodology 가능함.

---

## How?

‘The Lovász Bound’

the Lovász bound is a mathematical concept that provides an upper bound on the chromatic number of a graph. It is obtained by solving a linear program.

엄밀한 수학적 유도식과 설명을 원하시는 분은 [https://www.cs.cmu.edu/~15859n/RelatedWork/random-walks-on-graphs.pdf] 를 추천드립니다.

![left ; CT-LAYER , right : GAP-LAYER.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4ab6cb8e-8614-409b-8e98-7c9211cfecc0/Untitled.png)

left ; CT-LAYER , right : GAP-LAYER.

CT-LAYER : as **a metric-based** , neural approach.

→ a layer that learns **the commute times** and **uses them as a relevance function for edge re-weighting.**

![**Graph Rewiring: From theory to Applications in Fairness , page 50/n**](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7ffca247-8b76-430d-aea8-48b495dd2ff5/Untitled.png)

**Graph Rewiring: From theory to Applications in Fairness , page 50/n**

GAP-LAYER : as **a direct-neural approach to optimize the structure** of the graph to the task at hand.

→ a layer to **optimize the spectral gap**, depending on the nature of the network and the task at hand. 

compute these gradient either using Laplacians(L, with Fiedler \lambda_2) or normalized Laplacians (L, with Fiedler \lambda_2^’).

- Fiedler Vector
    - Courant-Fisher Theorem을 통해 , self-adjoint matrix 를 추출합니다. 이 때 목적은 ‘**orthogonal’** 를 추출하기 위함입니다. 또한, Rayliegh Quotient 를 활용하여 주어진 벡터가 행렬 A의 작용 하에서 어떻게 "늘어지거나" 수축하는지에 대해 파악하여 eigenvalue 를 측정합니다.
        
        ![**Graph Rewiring: From theory to Applications in Fairness , page 20/n**](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/60a040cc-b0bc-4b2f-8581-977ca23a36f7/Untitled.png)
        
        **Graph Rewiring: From theory to Applications in Fairness , page 20/n**
        
- Average Cut Problem
    - 앞서 언급한 fiedler vector 는 아래 average cut problem 에서 활용됩니다. 최적의 partition 을 위해 각 노드를 기준으로 cutting 을 하는데, 모든 노드 조합을 고려해야하기에 NP-Hard 문제에 속합니다. 이를 fielder vector 를 통해 해결합니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/78bf0369-165e-462a-8638-509a63f35a91/Untitled.png)
    

** Fiedler vector vs. eigen vector 차이점.

eigenvector 의 하위 요소가 Fiedler vector 이며, optimal partition 을 위한 재료로 활용됩니다. 

** Green's Function:

In mathematics and physics, Green's function is a mathematical tool used to solve certain types of differential equations. It provides a way to find the response of a system to an impulse or a localized source.

---

`**Code with explanation**`

# Graph Laplacian

그래프 라플라시안 행렬(Graph Laplacian matrix)은 그래프 이론에서 중요한 개념 중 하나입니다. 그래프 라플라시안 행렬은 주어진 그래프의 구조적인 특성을 나타내는 정방 행렬(asymmetric matrix)입니다. 그래프 라플라시안 행렬은 그래프의 정점들과 간선들의 연결성을 표현합니다. 일반적으로, 그래프 라플라시안 행렬은 그래프의 인접 행렬(A)과 차수 행렬(D)의 차로 정의됩니다. 여기서 인접 행렬은 그래프의 정점 간의 연결 관계를 나타내며, 차수 행렬은 각 정점의 차수(연결된 간선의 수)를 대각 행렬로 표현합니다. 일반적으로 사용되는 그래프 라플라시안 행렬은 다음과 같이 정의됩니다:

$$
L = D - A
$$

여기서 L은 그래프 라플라시안 행렬을 나타내며, D는 차수 행렬, A는 인접 행렬입니다. 예시 그림을 통해 알아보겠습니다.

![redraw.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/93999196-78ee-4f41-89d7-09adee4a88db/Untitled.png)

redraw.

## code snippet

```python
A = nx.adjacency_matrix(G)
L = nx.laplacian_matrix(G)

e, evecs = np.linalg.eig(L.todense())
idx = e.argsort()
e = e[idx]
evecs = evecs[:,idx]
```

# Dirichlet energy graph

앞서 도출한 그래프 라플라시안과 본 섹션에서 다루어볼 디리클레 에너지는 무슨 관련이 있을까요?

그래프 라플라시안과 디리클레 에너지는 그래프 이론과 스펙트럴 그래프 이론 분야에서 밀접한 관련이 있는 개념입니다. 그래프 라플라시안은 그래프의 중요한 구조적 특성을 나타내는 행렬 표현이며, 디리클레 에너지는 그래프 상의 매끄러움이나 규칙성을 측정하는 척도입니다. 그래프의 라플라시안 행렬은 n개의 정점을 갖는 무방향 그래프에 대해 L = D - A로 정의됩니다. 여기서 D는 대각성분이 정점의 차수인 대각 차수 행렬이고, A는 그래프의 인접 행렬입니다. **라플라시안 행렬은 그래프의 연결성과 위상에 대한 정보를 제공합니다.** 한편, **디리클레 에너지는 그래프 상의 정점에 정의된 함수가 상수에서 얼마나 벗어나는지를 측정하는 척도**입니다.

$$
E(f) = \frac{1}{2} \sum_{i,j}\in E , (f(i)-f(j))^2
$$

여기서 (i, j)는 그래프의 간선을 나타내고, w_ij는 그 간선에 연결된 가중치이며, f(i)와 f(j)는 각각 정점 i와 j에서 함수의 값입니다. 이를 통해 디리클레 에너지는 **그래프 상에서 함수의 매끄러움이나 규칙성을 측정**합니다. 정리해보자면, 라플라시안 행렬은 그래프의 구조적 특성을 포착하는 반면, 디리클레 에너지는 그래프에 정의된 함수의 부드러움 또는 규칙성을 측정합니다. **라플라시안 행렬의 고유 벡터는 이러한 고유 벡터에 대한 함수의 제곱 투영의 관점에서 디리클레 에너지를 표현하기 위한 기초를 제공**합니다.

**참고로, 본 논문에서는 rewiring 전/후 fiedler vector 간 dirichlet energy 를 측정하여, 최적의 sparification 을 진행합니다.**

## code snippet

```python
#1
idxAll = set(range(G.number_of_nodes()))
idxB = set([0, 19])
idxD = set([x for x in idxAll if x not in idxB])
# print(idxAll)

idxB = np.array(list(idxB))
idxD = np.array(list(idxD))
# print(idxB)
# print(idxD)

Ldense = L.todense()
print(Ldense)
LdenseD = Ldense[idxD,:]
LdenseD = LdenseD[:,idxD]

print(f"LdenseD matrix{LdenseD}")
print(idxD.shape[0])
print(idxB.shape[0])
print(np.zeros((2,2)))

Bdense = np.zeros((idxB.shape[0],idxD.shape[0]))
print(Ldense.shape)
Bdense = Ldense[idxB,:]
Bdense = Bdense[:,idxD]
print(Bdense)

Inv = np.matmul(np.linalg.inv(LdenseD),-np.transpose(Bdense))
print("inv",Inv)
uB = np.array([1,0])
uD = np.matmul(Inv,np.transpose(uB))
print(uD)
print(uD.shape)

#2
def calculate_dirichlet_energy(G, u):
    """
    Calculate the Dirichlet energy of a function u defined on a graph G.
    
    Parameters:
        G (NetworkX Graph): The graph on which the function is defined.
        u (dict): A dictionary mapping node IDs to function values.
        
    Returns:
        float: Dirichlet energy of the function.
    """
    energy = 0.0
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        degree = G.degree(node)
        u_node = u[node]
        
        for neighbor in neighbors:
            u_neighbor = u[neighbor]
            energy += (u_neighbor - u_node)**2
        
        energy += degree * u_node**2
    
    return energy
```

# Heat Kernel and graph diffusion

히트 커널(heat kernel)과 그래프 확산 모델(graph diffusion model)은 모두 그래프 상에서의 확산 현상과 관련이 있습니다. 히트 커널과 그래프 확산 모델은 **정보나 열이 그래프의 정점을 통해 어떻게 전파**되는지를 설명합니다. 히트 커널은 **연속적인 공간에서 열이나 정보의 시간에 따른 변화를 특성화**하는데 사용됩니다. 그래프에 적용되는 경우, 히트 커널은 정점 간 연결성에 기반하여 열이나 정보가 어떻게 확산되는지를 측정합니다. 반면, 그래프 확산 모델은 특히 그래프 상에서의 확산 과정을 다룹니다.  **정보나 열이 그래프의 간선을 통해 확산되는 것을 나타냅니다. 확산이 일반적으로 이산시간 마르코프 체인을 사용하여 모델링되며, 정점 간의 전이 확률은 그래프 간선에 연결된 가중치나 유사도 측정치에 따라 결정**됩니다. 히트 커널과 그래프 확산 모델의 상관관계는 **연속적인 확산과 이산적인 확산 과정 간의 관계**에 있습니다. 히트 커널은 그래프에 적용되어 이산화될 수 있으며, 이는 그래프 확산 모델과 밀접한 관련이 있습니다. 이산적인 그래프 확산 과정은 히트 커널이 기술하는 연속적인 열 확산을 근사화한 것으로 볼 수 있습니다.

## code snippet

```python
def heat_kernel(G,e,evecs,beta):
  n = G.number_of_nodes()
  H = np.zeros((n,n))
  Phi = evecs
  Lambda = np.diag(np.exp(-beta*e))
  H = np.matmul(Phi,Lambda)
  H = np.matmul(H,np.transpose(Phi))
  return H
```

# Commute Times

Commute Times는 그래프의 두 개의 정점 집합 간에 랜덤 워커(Random walker)가 한 집합에서 다른 집합으로 이동한 후 되돌아오기까지 평균적으로 걸리는 시간으로 정의됩니다. 수학적으로는 **그래프 라플라시안의 두 번째로 작은 고유값과 세 번째로 작은 고유값의 차이의 역수**로 계산됩니다.

스펙트럴 커뮤트 타임은 두 개의 정점 집합 간의 전이의 난이도를 측정합니다. 만약 **스펙트럴 커뮤트 타임이 크다면, 그것은 두 개의 집합이 잘 분리되고 구별됨**을 나타냅니다. 반대로, **스펙트럴 커뮤트 타임이 작다면, 그것은 두 개의 집합이 상호 연결되고 구별하기 어렵다는 것을 시사**합니다.

스펙트럴 커뮤트 타임은 그래프 분할, 커뮤니티 탐지, 클러스터링 알고리즘 등 다양한 응용 분야에서 활용됩니다. 스펙트럴 커뮤트 타임을 분석함으로써 그래프 내에서 조화로운 커뮤니티를 식별하거나 그래프를 여러 클러스터로 분할하는 최적의 절단을 찾을 수 있습니다.

<aside>
💡 랜덤 워커(Random walker)란 그래프 상에서 이동하는 가상의 개체를 의미합니다. 랜덤 워커는 특정 정점에서 시작하여 그래프의 다른 정점들 사이를 무작위로 이동하면서 그래프를 탐색합니다. 이동 경로는 현재 위치의 이웃 정점들 사이에서 무작위로 선택됩니다. 랜덤 워커는 그래프의 구조와 연결성을 탐색하고 특정 정점 간의 이동 패턴이나 정보 전파의 특성을 조사하는 데 사용됩니다. 스펙트럴 커뮤트 타임의 경우, 랜덤 워커는 하나의 정점 집합에서 다른 정점 집합으로 이동하며 시간에 따른 평균 이동 시간을 계산하는 데 활용됩니다. 이를 통해 그래프 내의 구조적인 속성과 클러스터 간의 관계를 파악할 수 있습니다.

</aside>

the CTE will preserve the commute times distance in a Euclidean space. Note that this latent space of the nodes can not only be described spectrally but also in a parameter free-manner, which is not the case for other spectral embeddings, such as heat kernel or diffusion  maps as they rely on a time parameter t. More precisely, the embedding **matrix Z whose columns contain the nodes’ commute times embeddings** is spectrally given by:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf5decd5-b74d-4c1c-b12b-7ba666a21efe/Untitled.png)

where Λ is the diagonal matrix of the unnormalized **Laplacian L eigenvalues** and **F is the matrix of their associated eigenvectors**. Similarly, Λ0 contains the eigenvalues of the normalized Laplacian L and G the eigenvectors. We have F = GD−1/2 or fi = giD−1/2 , where D is the degree matrix.

## code snippet

```python
def graph_vol(G):
  A = nx.adjacency_matrix(G)
  D = A.sum(axis=1)
  D = D.squeeze()
  d = np.zeros(G.number_of_nodes())
  for i in range(G.number_of_nodes()):
    d[i] = D[0,i]
  vol = d.sum()
  return vol

def commute_times_embedding(G,e, evecs):
  n = G.number_of_nodes()
  vol = graph_vol(G)
  A = nx.adjacency_matrix(G)
  D = A.sum(axis=1)
  Phi = evecs
  Lambda = np.diag(e)
  Lambda = fractional_matrix_power(Lambda, -0.5)
  Lambda[0,0] = 0
  CTE = np.sqrt(vol)*np.matmul(Lambda,np.transpose(Phi))
  return CTE

CTE = commute_times_embedding(G,e,evecs)
CT = pdist(np.transpose(CTE), 'euclidean')
CT = squareform(CT)

# Commute Times as an optimzation problem 
NL = nx.normalized_laplacian_matrix(G)
NLdense = NL.todense()
# Finding eigen values and eigen vectors
eNL, evecsNL = np.linalg.eig(NLdense)
# Sort them (both e's and evecs's) ascending
idx =eNL.argsort()
eNL = eNL[idx]
evecsNL = evecsNL[:,idx]

# Let Z be the CT Embedding
# Initialize (first row with Zeros)
n = nx.number_of_nodes(G)
Z = np.zeros((1,3))
# Deterministic initialization
Z = np.concatenate((Z,np.ones((n-1,3))),axis = 0)
# Random initialization
Z = np.random.rand(n,3)
plt.imshow(Z, alpha=0.8, cmap="seismic")

def optimizing_commute_times_cost_function(Z, NLdense, lambdaReg):
  return np.trace(np.matmul(np.matmul(np.transpose(Z),NLdense),Z)) + \
 lambdaReg*np.linalg.norm(np.matmul(np.transpose(Z),Z)-np.eye(Z.shape[1]))
mu = 0.01
lambdaReg =0.1
cost_list=[]
embedding_list = []
embedding_list.append(Z)
maxiter = 1000
for i in range(maxiter):
  grad = 2*np.matmul(NLdense,Z) + lambdaReg*4*np.matmul(Z,np.matmul(np.transpose(Z),Z)-np.eye(Z.shape[1]))  #+ lambdaReg*4*np.matmul(np.transpose(Z),np.matmul(Z,np.transpose(Z))-np.eye(Z.shape[0]))
  Z = Z - mu*grad
  cost_list.append(optimizing_commute_times_cost_function(Z,NLdense,lambdaReg))
  embedding_list.append(Z)

CTZ = pdist(Z, 'euclidean')
CTZ = squareform(CTZ)

```

---

`Conclusion` **** 

- new idea for positional encoding
- 

---

## Reference

1. https://blog.twitter.com/engineering/en_us/topics/insights/2022/over-squashing--bottlenecks--and-graph-ricci-curvature
2. https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15859-f11/www/notes/lecture11.pdf
3. Tortorella, Domenico, and Alessio Micheli. "Is Rewiring Actually Helpful in Graph Neural Networks?." *arXiv preprint arXiv:2305.19717* (2023)
4. Deac, Andreea, Marc Lackenby, and Petar Veličković. "Expander graph propagation." *Learning on Graphs Conference*. PMLR, 2022.
5. Arnaiz-Rodríguez, Adrián, et al. "DiffWire: Inductive Graph Rewiring via the Lov\'asz Bound." *arXiv preprint arXiv:2206.07369* (2022).
6. Banerjee, Pradeep Kr, et al. "Oversquashing in GNNs through the lens of information contraction and graph expansion." *2022 58th Annual Allerton Conference on Communication, Control, and Computing (Allerton)*. IEEE, 2022.
7. Topping, Jake, et al. "Understanding over-squashing and bottlenecks on graphs via curvature." *arXiv preprint arXiv:2111.14522* (2021).
