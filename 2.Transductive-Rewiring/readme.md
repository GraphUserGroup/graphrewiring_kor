# Transductive-Rewiring

## ✏️ Introduction

---

- **Transductive Graph Rewiring**
    
    : graph의 feature를 활용해 graph rewiring을 진행하는 Transductive Graph Rewiring을 소개합니다.
    : 전처리 단계를 통해 각 그래프의 새로운 convolution matrix를 계산한 뒤 rewiring에 활용합니다.
    : 본 튜토리얼에서는 Transductive Graph Rewiring의 2가지 방법을 소개합니다.
    
    1. Parameterized diffusion 
    2. Curvature-based approaches

- **Diffusion Rewiring**
    
    : GNN에서 핵심이 되는 Graph convolution은 일반적으로 1hop 이웃의 메시지를 활용합니다.
    : MPNN에서 더 먼 이웃의 메시지를 받아오려면 심층 레이어가 필요합니다.
    : 심층 레이어를 활용할 경우 embedding이 비슷해지는 over-smoothing 문제가 발생합니다.
    
    : 1hop 이웃을 활용하는 MPNN의 한계를 극복할 수 있는 Spectral GNNs도 있습니다.
    : Spectral GNNs는 1hop 이웃에만 의존하지 않고 더 복잡한 그래프 속성들을 활용합니다. 
    : 하지만 이 방법은 MPNN보다 성능이 떨어지고, 이전에 보지 못한 그래프에 일반화할 수 없다는 한계가 있습니다.
    : 또한 멀리 떨어진 노드에서 오는 메시지가 고정된 크기의 벡터로 압축되면서 발생하는 over-squashing 문제가 존재합니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8783045f-b904-46ec-a64b-f19659f7365b/Untitled.png)
    
    : 튜토리얼에서는 이 문제를 해결할 수 있는 방법으로 Diffusive Rewiring 기법을 소개합니다.
    : Graph Diffusion Convolution(GDC)는 MPNN과 Spectral GNNs의 장점을 결합한 방식입니다.
    
    : 멀리 떨어진 이웃의 메시지를 받아오기 위해 diffusion 프로세스를 활용, sparsification을 거쳐 기존 그래프를 rewiring합니다. 
    : 대표적인 graph diffusion 프로세스에는 PPR, heat kernel이 있습니다.
    : 본 튜토리얼에서는 그 중에 PPR(Personalized-PageRank) 알고리즘에 대해 살펴봅니다.
    
    $$
    S = \alpha(I_n + (\alpha-1)A)^{-1}
    $$
    
- **Curvatured-based Rewiring**
    
    : 앞서 살펴본 Diffusion Rewiring은 그래프의 기본 구조를 유지하면서 노이즈를 제거하는 방식이었다면 curvature-based rewiring은 훨씬 더 `외과적`인 접근입니다.
    : 그래프에서 over-squashing 현상을 유발하는 구조적인 특성을 bottleneck이라고 합니다.
    : Curvature-based rewiring은 이 bottleneck을 유발하는 지역을 찾아내 문제를 해결합니다.
    
    : bottleneck을 제대로 이해하기 위해선 그래프를 기하학적으로 살펴볼 필요가 있습니다.
    : 그래프의 기하학적 구조에 따라 message receptive field의 증가 형태가 달라질 수 있기 때문입니다.
    
    : 그리드 형태의 그래프라면 노드의 receptive field는 산술급수적으로 늘어납니다.
    : 반면 트리 구조의 그래프는 노드의 receptive field가 기하급수적으로 늘어납니다.
    : receptive field가 기하급수적으로 늘어나는 트리 구조의 그래프는 over-squashing이 발생할 가능성이 커집니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dc8f2069-f230-49f9-8091-4e016120b466/Untitled.png)
    
    : Curvature Rewiring은 리치 곡률을 활용해 over-squashing이 발생할 가능성이 높은 엣지를 찾아냅니다.
    : 해당 엣지 근처에 국부적으로 엣지를 추가하는 식으로 over-squashing을 해소합니다.
    

## ✏️ Personalized-PageRank

---

- **Personalized-PageRank(PPR) 알고리즘**
    
    $$
    S = \alpha(I_n + (\alpha-1)A)^{-1}
    $$
    
    : Personalized-PageRank 알고리즘에서는 일반적으로 alpha로 표시되는 감쇠 계수(damping factor)가 필요합니다. 이 감쇠 계수는 노드가 outgoing 링크를 따라갈 확률과 임의의 노드로 이동할 확률을 결정합니다.
    
    1. Add self-loops
    : 인접 행렬 A에 대각 행렬을 추가하여 self-loop를 추가합니다. 수정된 인접 행렬은 A_loop라고 부르겠습니다.
    2. Create Symmetric Transition Matrix
    : 행과 열을 정규화하여 A_loop에서 대칭 전이 행렬 T_sym을 만들어 냅니다.
    : 이 단계를 통해 그래프의 랜덤 워크가 연결된 노드에 균등하게 분포되도록 합니다.
    3. Compute PPR-based diffusion matrix
    : PPR을 기반으로 확산 행렬을 계산합니다
    : 이 확산 행렬은 그래프의 링크를 따라 각 노드에 도달할 확률을 나타냅니다.
    4. Sparsify using a threshold
    : 특정 임계값(eps) 미만의 값을 0으로 설정해서 계산된 확산 행렬 S를 희소화합니다.
    : 이 단계를 통해 낮은 확률을 제거하고 중요한 연결에만 집중하도록 합니다.
    : 희소화된 행렬은 열에 대해 정규화하여 각 노드별 확률의 합이 1이 되도록 합니다. 

### 🧑‍💻 GDC with Personalized-PageRank

```python
def gdcPageRank(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S, T_sym
```

- **python을 통해 gdcPageRank 함수 정의**
    
    : 이 함수에는 세 개의 매개변수가 사용됩니다.
    : 1. CSR(Compressed Sparse Row) 형식으로 된 그래프의 인접 행렬 A
    : 2. 감쇠 계수를 나타내는 alpha(부동 소수점 값)
    : 3. sparsification에 사용되는 임계값 eps(부동 소수점 값)
    
    <aside>
    💡 참고. CSR 형식
    
    sparse matrix는 대부분의 행렬의 값이 0이기 때문에 계산 등의 과정에서 매우 비효율적입니다. CSR(Compressed Sparse Row)는 sparse matrix를 효율적으로 관리하기 위한 방법 중 하나로  `Data`, `Row`, `Col` 3개의 벡터를 이용하여 행렬을 표시합니다.
    
    </aside>
    

- **Self loops**
    
    : 그래프의 인접 행렬 A에 크기가 N인 단위 행렬을 추가해서 새로운 행렬 A_loop를 생성하겠습니다.
    : 이 연산을 통해 그래프의 각 노드에 자체 루프를 추가합니다.
    

- **Symmetric transition matrix**
    
    : 여기서는 대칭 전이 행렬(symmetric transition matrix) T_sym을 계산합니다. 
    : 먼저 A_loop의 열을 합친 뒤 그 결과를 D_loop_vec에 저장합니다.
    : 그런 다음엔 D_loop_vec의 각 요소에 역제곱근을 취하고 그 결과를 D_loop_vec_invsqrt에 저장합니다.
    : 나온 결과 값에 sp.diags 함수를 통해 대각 행렬(D_loop_invsqrt)를 만듭니다.
    : 마지막으로 D_loop_invsqrt, A_loop, D_loop_invsqrt를 함께 곱하여 대칭 전이 행렬 T_sym을 계산합니다.
    

- **PPR-based diffusion**
    
    : 이 코드 라인에선 PPR 기반 확산 행렬 S를 계산합니다
    : 확산 행렬 S를 구하는 수식은 다음과 같습니다.
    
    $$
    S = \alpha(I_n + (\alpha-1)A)^{-1}
    $$
    

- **Sparsify using threshold epsilon**
    
    : 임계값 eps보다 크거나 같은 S의 요소는 유지, 나머지는 0으로 설정합니다.
    : 계산된 결과 값은 S_tilde에 저장됩니다.
    : 그리고 S_tilde에서 column-normalized된 전이 행렬 T_S를 계산합니다. 
    : 최종적으로 T_S와 원래 대칭 전이 행렬 T_sym을 반환합니다.
    

### 🧑‍💻 Create an Example Graph

```python
# Use here a larger SBM with smaller gap 
sizes = [100, 100]

# SBM with small gap
probsG = [[0.8, 0.01], [0.01, 0.8]]
G = nx.stochastic_block_model(sizes, probsG, seed=0)
A = nx.adjacency_matrix(G)

A_S = sp.csr_matrix(np.asarray(A.todense()))
T_S, T_sym = gdcPageRank(A, alpha =0.1, eps=10e-09) 
# alpha = 0.1, eps = 10e-6, 10e-8 works well for this graph. 
# However eps = 10e-4 increases self-diffusion and 10e-2 reaches no-out-diffusion
```

- **Stochastic Block Model**
    
    : SBM(Stochastic Block Model)은 특정 노드끼리 community structure를 가지는 그래프를 만드는 generative model입니다. SBM에서 그래프는 블록(community)의 크기(sizes)와 서로 다른 블록 간의 엣지 확률(probsG)을 지정하여 정의할 수 있습니다. 
    
    1. sizes
        
        : 블록의 크기 k는 길이가 k인 list로 표시할 수 있습니다.
        : `sizes[i]`는 블록 i의 크기를 나타냅니다.
        : 위의 예시에서 `sizes = [100, 100]`이라는 건 각각 100개의 노드를 포함하는 블록 두 개가 담긴 그래프를 의미합니다.
        
    2. probsG
        
        : 블록 간의 엣지 확률은 대칭행렬 k X K로 표현할 수 있습니다.
        : `probsG[i][j]`는 블록 i의 노드와 블록 j의 노드 사이에 엣지가 존재할 확률을 나타냅니다.
        : 위의 예시에서 `probsG = [[0.8, 0.01], [0.01, 0.8]]`는 블록 내의 엣지 확률이 0.80으로 높고, 블록 간의 엣지 확률은 0.01로 낮습니다.
        

### 🧑‍💻 Before and After applying gdcPageRank

```python
plt.imshow(A.todense(), alpha=0.8, cmap="seismic")
plt.colorbar()

plt.imshow(T_sym.todense(), alpha=0.8, cmap="seismic")
plt.colorbar()
```

![original A graph](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9119a330-e362-4386-a8fb-7318d0d2ff99/Untitled.png)

original A graph

![gdcPageRank(A) = T_sym graph](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8b584f82-914d-43f6-99ef-285850f79a93/Untitled.png)

gdcPageRank(A) = T_sym graph

- **Visualization of the Adjacency matrix**
    
    : todense() 함수를 통해 행렬 A를 시각화할 수 있는 고밀도 행렬로 변환합니다.
    : 결과로 나온 이미지는 행렬 A를 사각형 그리드를 나타냅니다.
    : 빨간색에 가까울수록 엣지가 존재함을 나타내고, 파란색에 가까울수록 엣지가 없음을 나타냅니다. 
    

- **Visualization of T_S**
    
    : gdcPageRank를 통해 나온 결과 값 중 T_S의 경우엔 임계값 eps보다 작은 값은 0으로 drop됐고, column-normalized 되었습니다.
    : 즉, T_S는 stochastic matrix라고 할 수 있습니다.
    : 그래프 T_S에 대해 엣지 존재 여부를 시각화하면 아래와 같은 plot을 얻을 수 있습니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1aa944fc-222f-4460-ad62-970902b674ed/Untitled.png)
    
    ```python
    # Get intra and inter edges magnitudes of T_S for latter thresholding 
    # Partition
    S = set([x for x in range(sizes[0])])
    notS = set([x for x in range(sizes[0], 2*sizes[0])])
    
    #Intra-class magnitudes
    T_S_intra_S = get_intra_edges_magnitude(T_S,S)
    print(len(T_S_intra_S))
    T_S_intra_notS = get_intra_edges_magnitude(T_S,notS)
    print(len(T_S_intra_notS))
    T_S_intra = list()
    for i in range(len(T_S_intra_S)):
      T_S_intra.append(T_S_intra_S[i])
    for i in range(len(T_S_intra_notS)):
      T_S_intra.append(T_S_intra_notS[i])
    
    # Inter-class magnitudes 
    T_S_inter = get_inter_edges_magnitude(T_S,S,notS)
    print(len(T_S_inter))
    ```
    
    : SBM의 첫번째 블록에 속한 노드를 분리해 S라는 집합을 만듭니다.
    : notS 집합에는 두번째 블록에 속한 노드가 포함되어 있습니다.
    
    : 먼저 class 내부의 엣지 크기를 측정해봅니다.
    : T_S_intra_S에는 집합 S의 엣지 크기를 저장하는 리스트입니다.
    : `get_intra_edges_magnitude` 함수를 이용합니다.
    : 마찬가지로 T_S_intra_notS에는 집합 notS의 엣지 크기를 저장합니다.
    : 두 개를 단일 리스트 T_S_intra로 결합합니다.
    
    : 이번엔 class 사이의 엣지 크기를 재보겠습니다.
    : T_S_inter에는 집합 S와 집합 notS 사이의 엣지 수를 저장합니다.
    
    ⇒ 엣지 크기는 intra_S, intra_notS = 4,950 / S_inter는 10,000입니다.
    
    ```python
    # Edge magnitude Distributions (intra vs inter)
    # Using seaborn
    import pandas as pd
    import seaborn as sns
    
    MagGIntra = T_S_intra
    MagGInter = T_S_inter
    
    df_intraG = pd.DataFrame(list(zip(MagGIntra,['Intra']*len(MagGIntra))), columns=['Magnitude', 'Edge Type'])
    df_interG = pd.DataFrame(list(zip(MagGInter,['Inter']*len(MagGInter))), columns=['Magnitude', 'Edge Type'])
    df_mag = pd.concat([df_intraG, df_interG]).reset_index()
    
    # Visualization
    style = {'bins':30, 'kde':True, 'element':"step"}
    plt.figure(figsize=(8,6))
    sns.histplot(data=df_mag, 
                 x='Magnitude', 
                 hue=df_mag[['Edge Type']].apply(tuple, axis=1), 
                 alpha=0.5, 
                 palette=["r", "g"], 
                 **style)
    plt.show()
    ```
    
    : seaborn 라이브러리를 사용해서 intra와 inter의 엣지 크기 분포를 비교해보겠습니다.
    : 우선 T_S_intra는 MagGIntra에, T_S_inter는 MagGInter에 할당합니다.
    : 할당된 리스트를 Magnitude와 Edge Type 두 개의 열이 있는 데이터프레임으로 만듭니다.
    : 생성한 데이터프레임 df_mag를 가지고 히스토그램 시각화를 진행합니다.
    : Edge Type에 따라 색상을 다르게 표시해주면 아래와 같은 plot을 얻을 수 있습니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/75066606-6d66-4970-a08f-c0f861cfe57b/Untitled.png)
    

## ✏️ **Curvature-based Rewiring**

---

- **Ricci Curvature**
    
    : 이번에는 그래프의 bottleneck 때문에 발생하는 Over-squashing 문제를 해결해보는 방법을 살펴기 위해 그래프의 기하학적 구조를 알 수 있는 리치 곡률을 이용합니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/619314c9-921f-4b1e-9c75-515a6e0c488e/Untitled.png)
    
    : 그래프에서 리치 곡률은 두 노두 사이에서 나오는 엣지에 의해 형성된 로컬 구조를 통해 확인할 수 있습니다.
    : 곡률이 0보다 큰 spherical geometry에서는 삼각형(Clique) 구조를, 곡률이 0인 euclidean geometry에서는 평행 구조를, 곡률이 0보다 작은 hyperbolic geometry에서는 트리(Tree) 구조를 나타냅니다. 
    
     
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e6b311ef-2561-412a-b7fe-853a675b95b2/Untitled.png)
    
     
    

- **Balanced Forman Curvature**
    
    $$
    Ric(i, j) := \frac {2} {d_i} + \frac {2} {d_j} - 2 + 2 \frac {|♯_{∆}(i, j)|} {max(d_i, d_j)}  + \frac {|♯_{∆}(i, j)|} {min(d_i, d_j)} +  \frac {(γ_{max})^{-1}} {max(d_i, d_j)} (|♯_{□}^{i}| + |♯_{□}^{j}| )  
    $$
    
    : 리치 곡률을 그래프에 적용해 balanced forman curvature를 계산해봅니다. 
    : balanced forman curvature에서는 Sperical geometry의 특성을 확인할 수 있는 삼각형의 갯수, Euclidean geometry의 특성을 확인할 수 있는 사각형의 갯수를 이용해 그 곡률을 계산합니다.
    : 계산된 곡률이 마이너스인 엣지들은 over-squashing을 일으키는 bottleneck 현상을 일으킵니다.
    
    1. $♯_{∆}(i, j)$ : 노드  $i, j$가 이루는 삼각형의 갯수
    2. $♯_{□}^{i}(i, j)$ :  노드  $i, j$가 이루는 사각형(내부 대각선이 존재하지 않는 사각형)에서 노드 $i$의 이웃 노드 갯수
    3. $γ_{max}$ : 공통 노드를 공유하는 사각형의 최대 갯수
    
    - **0-1 엣지 balanced forman curvature 계산**
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7e67b60b-f3f5-4e65-b34f-ea9ffddf49d3/Untitled.png)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf1efe3d-7463-452d-9815-44d04f292e99/Untitled.png)
        

### 🧑‍💻 balanced forman curvature

```python
@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:])"
)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0:
            C[i, j] = 0
            return

        if d_in[i] > d_out[j]:
            d_max = d_in[i]
            d_min = d_out[j]
        else:
            d_max = d_out[j]
            d_min = d_in[i]

        if d_max * d_min == 0:
            C[i, j] = 0
            return

        sharp_ij = 0
        lambda_ij = 0
        for k in range(N):
            TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

            TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

        C[i, j] = (
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
        )
        if lambda_ij > 0:
            C[i, j] += sharp_ij / (d_max * lambda_ij)
```

- **@cuda.jit**
    
    : 먼저 함수 정의 앞에 함수 데코레이터 @cuda.jit가 표시됩니다. 
    : 이 데코레이터는 함수가 CUDA를 사용하고 있음을 나타냅니다. 
    : cuda.jit 데코레이터는 함수 인수의 데이터 유형과 반환 유형을 지정합니다. 
    : 이 경우 함수는 여러 개의 float32 배열과 정수를 받고 아무것도 반환하지 않습니다.
    
    : CUDA 그리드에서 인덱스 i와 j를 추출해 전체 노드 수 N보다 작은지 확인합니다.
    : 이 검사를 통해 조건문 내의 코드가 유효한 그리드 인덱스에 대해서만 실행하도록 합니다.
    

- **Argument**
    
    A: 그래프의 인접 행렬
    A2: 그래프의 제곱 인접 행렬
    d_in: 그래프에서 각 노드의 in-degree
    d_out: 그래프에서 각 노드의 out-degree
    N: 그래프의 노드 수
    C: 재구성할 curvature 행렬
    

- _**balanced forman curvature**
    
    : 노드 i와 j의 in-degree, out-degree를 기반으로 d_max, d_min을 결정합니다.
    : d_max와 d_min의 곱이 0이면  C의 값은 0으로 설정합니다. 0으로 인한 오류를 방지하고, 곡률을 0으로 고정합니다.
    : 계산된 값을 이용해 행렬 C의 인덱스 (i, j)에 대한 곡률을 계산합니다.
    : lambda_ij가 0보다 크면 sharp_ij, d_max, lambda_ij를 기준으로 곡률 값에 항을 추가합니다.
    

- **balanced forman curvature**
    
    : 이 함수는 _balanced_forman_curvature 함수를 감싸는 wrapper 역할을 합니다.
    : A(인접 행렬), C(곡률 행렬) 두 개의 입력 인자를 받습니다.
    : _balanced_forman_curvature 함수에 필요한 인자를 계산한 뒤, 해당 인자들을 통해 곡률을 반환합니다.
    

### 🧑‍💻 balanced forman post delta

```python
@cuda.jit(
    "void(float32[:,:], float32[:,:], float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32)"
)
def _balanced_forman_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    I, J = cuda.grid(2)

    if (I < dim_i) and (J < dim_j):
        i = i_neighbors[I]
        j = j_neighbors[J]

        if (i == j) or (A[i, j] != 0):
            D[I, J] = -1000
            return

        # Difference in degree terms
        if j == x:
            d_in_x += 1
        elif i == y:
            d_out_y += 1

        if d_in_x * d_out_y == 0:
            D[I, J] = 0
            return

        if d_in_x > d_out_y:
            d_max = d_in_x
            d_min = d_out_y
        else:
            d_max = d_out_y
            d_min = d_in_x

        # Difference in triangles term
        A2_x_y = A2[x, y]
        if (x == i) and (A[j, y] != 0):
            A2_x_y += A[j, y]
        elif (y == j) and (A[x, i] != 0):
            A2_x_y += A[x, i]

        # Difference in four-cycles term
        sharp_ij = 0
        lambda_ij = 0
        for z in range(N):
            A_z_y = A[z, y] + 0
            A_x_z = A[x, z] + 0
            A2_z_y = A2[z, y] + 0
            A2_x_z = A2[x, z] + 0

            if (z == i) and (y == j):
                A_z_y += 1
            if (x == i) and (z == j):
                A_x_z += 1
            if (z == i) and (A[j, y] != 0):
                A2_z_y += A[j, y]
            if (x == i) and (A[j, z] != 0):
                A2_x_z += A[j, z]
            if (y == j) and (A[z, i] != 0):
                A2_z_y += A[z, i]
            if (z == j) and (A[x, i] != 0):
                A2_x_z += A[x, i]

            TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

            TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

        D[I, J] = (
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
        )
        if lambda_ij > 0:
            D[I, J] += sharp_ij / (d_max * lambda_ij)

def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _balanced_forman_post_delta[blockspergrid, threadsperblock](
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D
```

- **balanced forman post delta**
    
    : _balanced_forman_post_delta 함수는 엣지를 추가하거나 제거한 후 곡률 변화를 계산합니다.
    : 이 함수에서는 degree term, triangle term, 4-cycles term을 고려합니다. 
    : 두 정점 x와 y 사이에 에지를 추가할 경우의 잠재적 개선 효과를 평가하고 추가할 엣지를 결정하는 데 사용됩니다.
    :  balanced_forman_post_delta 함수는 _balanced_forman_post_delta 함수와 함께 rewiring 이후의 곡률 변화를 계산합니다.
    

### 🧑‍💻 Stochastic Discrete Ricci Flow (SDRF)

```python
def sdrf(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.6, # default 0.5
    tau=1, # default 1
    is_undirected=True,
):
    G = data
    A = nx.adjacency_matrix(G)
    A = torch.tensor(A.toarray(), dtype = torch.float).fill_diagonal_(0)
    N = A.shape[0]
    
    if is_undirected:
        G = G.to_undirected()
        #G = G.to_directed()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    for x in range(loops):
        can_add = True
        balanced_forman_curvature(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        
        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break

    return G
```

- **SDRF(Stochastic Discrete Ricci Flow)**
    
    : sdrf 함수는 SDRF 접근 방식을 사용하여 그래프 재구성을 수행하는 주요 알고리즘입니다. 
    : 우선 nx.adjacency_matrix 함수를 이용하여 입력 그래프 데이터를 인접성 행렬 A로 변환합니다. 
    : 그런 다음 인접 행렬을 토치 텐서로 변환하고 더 빠른 계산을 위해 GPU를 활용합니다.
    : 여기서 변수 N은 그래프의 노드 수를 나타냅니다.
    
    : 메인 루프는 지정된 반복 횟수(루프)만큼 진행됩니다. 
    : 반복 내에서 알고리즘은 C 행렬에서 최소 곡률 값을 갖는 엣지(x, y)를 결정합니다. 
    : 이 알고리즘은 x와 y의 이웃을 찾고 그래프에 추가할 수 있는 후보 엣지를 식별합니다.
    
    : 후보 엣지를 식별한 뒤에는 balanced_forman_post_delta 함수를 사용하여 각 후보에 대한 곡률 변화를 계산합니다. 
    : 그런 다음 개선 값의 소프트맥스 확률에 따라 추가할 엣지를 선택하고 그래프에 추가합니다. 
    : 이에 따라서 인접성 행렬 A가 업데이트됩니다.
    

```python
Ghat = sdrf(G)
posGhat = plot_eigenvector_over_graph(Ghat,evecsG,1, pos = posG)
Ahat = nx.adjacency_matrix(Ghat)
plt.imshow(Ahat.todense()-A.todense(), alpha=0.8, cmap="seismic")
plt.colorbar()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9f9ea2b-d14d-4b17-abb0-0d67d7f8bbdd/Untitled.png)

: sdrf를 거친 Ghat을 그려보면 이전과의 큰 차이를 육안으로는 확인하기 어렵습니다.
: 수정된 그래프의 인접행렬(Ahat)과 수정 전 그래프의 인접행렬(A)의 차이를 시작화하면 오른쪽과 같은 그래프를 얻을 수 있습니다.

## ✏️ Diffusion vs Curvature

---

### **🧑‍💻 Degree Distributions**

```python
# Using seaborn
import pandas as pd
import seaborn as sns

degG = list(degree_vector(A))
print(degG)
degGth = list(degree_vector(T_S_th))
print(degGth)
degGhat = list(degree_vector(Ahat))
print(degGhat)

df_degG = pd.DataFrame(list(zip(degG,['G']*len(degG))), columns=['Degree', 'Graph'])
df_degGth = pd.DataFrame(list(zip(degGth,['Gth']*len(degGhat))), columns=['Degree', 'Graph'])
df_degGhat = pd.DataFrame(list(zip(degGhat,['Ghat']*len(degGhat))), columns=['Degree', 'Graph'])
df_deg = pd.concat([df_degG, df_degGth, df_degGhat]).reset_index()
```

: 원본 그래프의 인접행렬 A와 Graph Diffusion(PPR)을 거친 T_S_th(Threshold T_S graph), sdrf를 거친 Ahat의 degree vector를 계산해 비교해보겠습니다.
: 원본 그래프는 G, diffusion rewiring을 거친 그래프는 Gth, curvature rewiring을 거친 그래프는 Ghat이라는 이름으로 데이터프레임을 만듭니다.
: 세 데이터프레임을 df_deg에 concatenate합니다.
: df_deg를 가지고 histogram을 그려봅니다. 

```python
style = {'bins':30, 'kde':True, 'element':"step"}
plt.figure(figsize=(8,6))
sns.histplot(data=df_deg, x='Degree', hue=df_deg[['Graph']].apply(tuple, axis=1), alpha=0.5, palette=["r", "g", "b"], **style)
plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8cd76c07-68bb-48dd-b52e-28c89c26cc6b/Untitled.png)

: 원본 그래프 G와 curvature rewiring을 거친 Ghat의 분포는 큰 차이가 없습니다.
: diffusion rewiring을 거친 Gth는 degree distribution의 차이가 큽니다.

⇒ SDRF는 GDC/DIGL보다 원래 그래프 구조를 더 많이 보존합니다.
⇒ Diffusion은 homophilic graph에서 더 잘 작동하고, SDRF는 heterophilic graph에서 더 잘 작동합니다.

## 📄 Reference

---

1. https://towardsdatascience.com/over-squashing-bottlenecks-and-graph-ricci-curvature-c238b7169e16
2. [Arnaiz-Rodríguez, A., Begga, A., Escolano, F., & Oliver, N. (2022). DiffWire: Inductive Graph Rewiring via the Lovasz Bound. arXiv preprint arXiv:2206.07369.](https://arxiv.org/abs/2206.07369)
3. [Johannes G asteiger, Stefan Weißenberger, and Stephan Günnemann. (2019). Diffusion improves graph learning. arXiv preprint arXiv:1911.05485](https://arxiv.org/abs/1911.05485)
4. [Huda Nassar, Kyle Kloster, and David F. Gleich . Strong Localization in Personalized PageRank Vectors . In International Workshop on Algorithms and Models for the Web Graph (WAW)](https://www.math.purdue.edu/~kkloste/pagerank-localization.pdf)
