# Inductive-Rewiring-CTLayer

## ✏️ TL;DR

---

<aside>
💡 A layer that learns the commute times and uses them as a relevant function for edge re-weighting performs preliminary studies on the use of CT-LAYER for homophilic and heterophilic node classification tasks

Commute Time을 학습하고 Edge re-weighting을 하기 위한 함수로 사용하는 레이어

※ **Commute Time**: The Commute Time between two nodes is defined as the sum of the expected times it takes for the random walker to travel from one node to the other and then return.

</aside>

## ✏️ Introduction

---

- **Background**
    
    : 기존 Massage Passing Neural Network (MPNN) 그래프 고유의 복잡성으로 인해 2가지 제한 사항 발생합니다.
    
    1. 복잡한 그래프 구조에서 최상의 결과는, **적은 수의 레이어에서만 발생**한다는 한계가 있습니다.
        
        ⛔ 레이어가 많은 네트워크는 Over-smoothing 및 Over-Squahsing 문제 발생
        
        - **Over-smoothing:** 그래프 신경망 layer 수가 증가할 수록, 정점의 임베딩이 서로 유사해지는 현상
        - **Over squashing:** 많은 nodes가 bottleneck node를 지나면, messages가 지나치게 squashing/compression이 되어 messages가 손상되는 현상
    2. MPNN의 깊이가 그래프의 직경보다 작을 때, **Under-reaching 현상**이 발생할 수 있습니다.
        - **Under-reaching:** MPNN이 그래프의 전체 구조에 의존하는 정보를 캡처하지 못할 뿐더러, 멀리 있는 노드간에 정보가 제대로 도달하지는 못하는 현상
    
    : 위의 두개 문제를 해결 하기 위해, CT-Layer 와 Gap-Layer가 고안 되었습니다.
    : 본격적으로 CT-Layer에 설명하기에 앞서, The Lovász Bound에 대하여 알고 있어야합니다.
    

- **The Lovász Bound**
    
    : Lovász Bound는 통근 시간(유효 저항 거리)과 네트워크의 스펙트럼 갭 사이의 관계를 수학적으로 표현한 부등식입니다 (Arnaiz-Rodriguez et al. ‘22). 
    : 여기서 통근 시간(CT)은 확산 이론, 스펙트럼 갭은 곡률 이론과 관련이 있습니다. 이 둘은 모두 미분 기하학에 근거하고 있습니다.
    : Lovász Bound 부등식은 다음과 같습니다:
    
    $$
    \left| \frac{CT_{uv}}{vol(G)}- \left( \frac{1}{d_u}+\frac{1}{d_v} \right) \right| \leq \frac{1}{\lambda_2^\prime} \frac{1}{d_{min}}
    $$
    
    : 위의 식에서 $CT_{uv}$는 두 정점 u와 v 사이의 통근 시간(CT)을 나타냅니다. 이 식은 통근 시간이 어떻게 네트워크의 스펙트럼 갭과 관련 되는지 설명해 줍니다. 
    : 수식의 왼쪽 부분은 통근 시간을 전체 그래프의 부피로 나눈 값과 $\left( \frac{1}{d_u}+\frac{1}{d_v} \right)$의 차이의 절댓값이며, 실제 통근 시간과 그래프 이론에서 기대되는 통근 시간 간의 차이를 나타냅니다.
    : 수식의 오른쪽 부분은 두 번째 가장 작은 고유값의 역수인 $\lambda_2^\prime$ (Fiedler value 라고도 불린다)와 최소 차수($d_{min}$)의 역수의 곱으로 나눈 값이며, 스펙트럼 갭과 최소 차수에 대한 정보를 제공합니다. 
    : $\lambda^{\prime}_2$의 값은 그래프가 얼마나 잘 연결되어 있는지 알려줍니다.
    : 따라서, 위의 식은 통근 시간과 스펙트럼 갭 사이의 관계를 제한하고, 두 정점 사이의 통근 시간이 스펙트럼 갭에 어떻게 영향을 받는지를 나타내며, 이를 통해 통근 시간과 그래프의 기하학적 특성 사이의 연결성을 이해할 수 있습니다.
    

- **Commute Time (CT, 통근시간)**
    
    : 그래프 이론에서 통근 시간(*Commute Time; CT*) 은 두 개의 정점사이의 최단 경로 길이가 아닌 **두개 정점사이의 통과하는 기대값**으로 정의 됩니다. 
    : 통근 시간은 보통 랜덤 워크(*Random Walk*) 개념을 사용하여 정의되며, 랜덤 워커가 노드 $i$에서 노드 $j$로 이동하고, 다시 노드 $i$로 돌아오기까지 걸리는 시간을 누적하여 측정할 수 있습니다. 
    
    <aside>
    💡 $CT_{uv}(=H_{uv} + H_{vu})$
    ✅ 두 노드($u, v$) 사이에,  $u$에서 $v$를 치고(*hit*) 다시 $u$로 돌아오는데 걸리는 시간
    
    </aside>
    
    <aside>
    💡 **랜덤 워크 (Random Walk)**
    ✅ 무작위로 정점간을 이동하는 프로세스를 말하며, 각 정점에서 다른 정점으로 이동할 확률이 정점간의 연결성에 의해 결정됩니다.
    
    </aside>
    

## ✏️ CT-Layer (Theory)

---

- **Commute Time Embedding (CTE)**
    
    : CT-Layer에서 저자들은 통근 시간(CT)을 직접 계산한 것이 아니라 임배딩(CTE)을 도입하여 CT를 근사적으로 학습하는 방법을 사용하였습니다. 
    : 따라서, CT-Layer를 이해하기 위해선 CTE에 대한 이해가 먼저 필요합니다.
    : CTE는 각 노드들의 통근시간 정보와 유클리드 공간에서의 거리 정보를반영한 임베딩 벡터이며, CTE행렬 $**\mathbf{Z}**$는 아래와 같은 수식으로 표현 가능합니다:
    
    $$
    \mathbf{Z}\coloneqq \sqrt{vol(G)}\Lambda^{-1/2}\mathbf{F}^T
    $$
    
    $$
    CT_{uv} = \|z_u - z_v\|^2
    $$
    
    : CTE를 학습시킬 때 이용되는 최적화 문제의 목적함수는 인접한 노드 쌍 사이의 CT 차이를 최소화하는 것을 목표로 합니다.
    : 같은 맥락으로, 목적함수가 인접한 노드 사이의 유효 저항을 최소화하거나, 혹은 CTE 간의 차이를 최소화한다고 해석할 수도 있습니다.
    

- **CT-Layer**
    
    : CT-Layer는 CTE를 학습하도록 고안된 뉴럴 네트워크 모듈입니다. 
    : 기존의 그래프재배선(*graph rewiring*) 방식들과는 다르게 CT-Layer는 프리프로세싱 방법이 아닌 end-to-end 방식으로 전체 모델과 같이 트레이닝을 합니다. 
    : 또한 CT-Layer는 재배선과 관련한 하이퍼 파라미터를 일체 사용하지 않는다는 장점이 있습니다.
    : CT-Layer는 single-layer MPNN과 그에 따른 부차적인 연산들로 구성되어있습니다. 
    
    - **유도과정 (technical details)**
        
        $X\in\mathbb{R}^{V\times H_X}$를 그래프 피쳐 입력값이라 하고, $Z\in\mathbb{R}^{V\times H_Z}$를 MPNN의 결과값이라고 할때 CT-Layer의 최종 아웃풋은 $T^{CT}$ 는 다음과 같이 구해집니다.
        
        - $z_u=f(x_u,N(u))$
        - $x_u$와 $z_u$는 각각 $X$와 $Z$의 행(*row*)벡터들이며 (*i.e.* $n\in V$), $N(u)$는 노드 $u$의 이웃 노드들의 집합입니다.
        - $\mathbf{T}^{CT}=\frac{cdist(\mathbf{Z})}{vol(G)}\odot A$
        - $cdist(\cdot)$ 는 거리 매트릭(*metric*)으로써 $cdist(Z)\in\mathbb{R}^{V\times V}$의 임의의 요소(*element*) $c_{ij}$는 $\|z_i - z_j\|^2$로 계산됩니다. $vol(G)$는 그래프 $G$의 볼륨이며 (*i.e.* degree의 총합), $A$는 $G$의 인접행렬(*adjacency matrix*), $\odot$은 Hadamard product (*i.e.* element-wise multiplication)입니다.
        - 즉, CT-Layer의 아웃풋 $\mathbf{T}^{CT}$ 는 일종의 재배선된 그래프(*rewired graph*)의 새 인접행렬인 셈이며, $(\frac{cdist(\mathbf{Z})}{vol(G)},\odot)$ 는 이 재배선 과정의 relevance function이 되는 셈이다.
        - CT-Layer는 다음의 로스함수를 최적화하여 트레이닝 하게 된다.
            - $Loss_{_{CT}} = \frac{Tr[\mathbf{Z}^T\mathbf{LZ}]}{Tr[\mathbf{Z}^T\mathbf{DZ}]} + \|\frac{\mathbf{Z}^T\mathbf{Z}}{\|\mathbf{Z}^T\mathbf{Z}\|_{_F}}-\mathbf{I}\|_{_F}$
            - 첫번째 텀(*term*)은 CT를 최소화 하는 역할을 하며 (*i.e.* $\|z_i-z_j\|^2,\;\;\forall (i,j)$), 두번째 텀은 정규화(*normalizing*) 및 직교성(*orthogonality*)에 대한 제약식(*constraints*) 역할을 한다.

## 🧑‍💻 CT-Layer (Hands-on)

---

```python
def dense_CT_rewiring(x, adj, s):
    """
    Args:
        x (dense): feature matrix: BxNxF
        adj (dense): dense adjacency matrix: BxNxN
        s (dense): CT Embedding: BxNxH (H: size of latent space)
    Returns:
        adj: new adjacency = CTdist/vol(G)
        loss: Cut Loss for CT Embedding (s)
        ortho_loss: Loss regularization orthogonality in CT Embedding (s)
    """
    # Activation
    s = torch.tanh(s) # torch.Size([B,N,H])
		H = s.size(-1)
    
    # CT Rewiring
    ## Calculate CT_dist 
    CT_dist = torch.cdist(s,s) # [B,N,H], [B,N,H]-> [B,N,N]
    ## Calculate degree d_flat and degree matrix d
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([B,N]) 
    d = _rank3_diag(d_flat)  # d torch.Size([B,N,N])
    ## Calculate Vol (volumes): one per graph 
    vol = _rank3_trace(d) # torch.Size([B]) 
    ## Calculate out_adj as CT_dist/vol(G)
    CT_dist = (CT_dist) / vol.unsqueeze(1).unsqueeze(1)
    ## New adjacency matrix (T^{CT})
    adj = CT_dist*adj
    
    # Cut loss
    ## Calculate Laplacian L = D - A 
    L = d - adj
    ## Calculate Tr[S.T*L*S]
    CT_num = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), L), s))
    ## Calculate Tr[S.T*D*S]
    CT_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
		## Cut loss for CT embedding (the first term)
    CT_loss = torch.mean(CT_num / CT_den)

    # Orthogonality regularization loss
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(H).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s)
    ortho_loss = torch.mean(ortho_loss) # Mean over batch!
    
    return adj, CT_loss, ortho_loss
```

- **Arguments**
    
    : `x`: 그래프 피쳐 매트릭스로, BxNxF 차원을 가집니다.
    : `adj`: 원본 그래프의 인접행렬이며, BxNxN 차원을 가집니다.
    : `s`: CTE를 나타내는 행렬이며, BxNxH 차원을 가집니다.
    
    ⛔ B는 배치 사이즈, N은 그래프의 (최대)노드 개수입니다.
    ⛔ 위 코드는 원활한 이해를 위해 원본 CT-Layer 코드를 간소화한 버전입니다.
    
- **Activation**
    
    : 모델 트레이닝의 안정화를 위해 탄젠트 하이퍼볼릭(*tanh*) 활성함수(*activation*)를 취해주었습니다.
    
- **CT Rewiring**
    
    : 재배선된 그래프의 새 인접행렬 $\mathbf{T}^{CT}$를 구하는 과정입니다.
    : $\mathbf{T}^{CT}=\frac{cdist(\mathbf{Z})}{vol(G)}\odot A$
    : `_rank3_diag()` 는 1D 벡터를 대각행렬로 변환해주는 함수의 배치 버전이며, `_rank3_trace()` 는 메트릭스의 트레이스를 구하는 함수의 배치 버전으로, DiffWire 패키지에 정의되어 있습니다.
    
- **Cut loss**
    
    : $Loss_{_{CT}}$에서 첫번째 텀  $*\frac{Tr[\mathbf{Z}^T\mathbf{LZ}]}{Tr[\mathbf{Z}^T\mathbf{DZ}]}*$을 계산하는 코드입니다.
    : $\mathbf{L}$를 계산할 때, 재배선된 그래프의 인접행렬이 쓰이는 것에 주의합니다.
    : 수식에서 $\mathbf{Z}$는 `dense_CT_rewiring()` 내에서 `s`로 표현되었습니다.
    
- **Orthogonality regularization loss**
    
    : $Loss_{_{CT}}$에서 두번째 텀 $*\|\frac{\mathbf{Z}^T\mathbf{Z}}{\|\mathbf{Z}^T\mathbf{Z}\|_{_F}}-\mathbf{I}\|_{_F}*$을 계산하는 코드입니다.
    : $\mathbf{Z}^T\mathbf{Z}$는 `ss`로 $\mathbf{I}$는 `i_s`로 각각 표현되었으며, 계산할때 변수들의 차원을 주의해 주는 것 외에는 특별히 복잡한 부분은 없습니다.
    

## ✏️ GAP-Layer (Theory)

---

- **Main idea**
    
    : GAP-Layer의 핵심 되는 아이디어는 Lovasz bound 부등식의 **우변항**(*right-hand side*)을 조절하여 바운드를 완화시키는 것입니다. 
    : 예를 들어, 재배선을 통해 Fiedler value ($\lambda_2$ 또는  $\lambda_2^{\prime}$)를 줄이게 되면, Lovasz bound 제약식이 완화되어 임의의 두 엣지들(*edges*) 간의 유효저항(*effective resistance*) 차이가 커질 수 있게 됩니다.
    
    : 직관적으로 생각해 보면, Fiedler value를 줄인다는 것은 병목 엣지(*bottleneck edge*)들을 더욱 과장하는 효과를 준다는 것을 의미합니다. 
    : 또는, 같은 맥락으로, 병목 에지의 curvature를 증가시키는 효과를 준다는 것을 의미합니다. 이는 병목 에지의 curvature를 줄이는 CT-Layer의 작용과 상반되는 효과입니다. 
    
    : 이러한 CT-Layer와 GAP-Layer의 효과 차이는 아래의 그림에도 잘 드러나 있습니다. 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/63e98a95-defc-49fe-a5bd-8292aa110d9e/Untitled.png)
    
    : 일반적으로 inter-cluster간 연결다리(*bridge*)가 되는 에지들이 (*i.e.* 병목 에지) 적을수록, 그리고 그 연결다리의 상대적인 넓이가 작을수록 (*i.e.* 에지 웨이트의 상대적 크기가 작을수록; 에지 curvature가 클 수록) 병목현상이 심하다고 볼 수 있습니다. 
    : 위의 그림을 보면 CT-Layer는 병목 에지들의 curvature를 증가하는 방향으로 재배선을 한 반면 GAP-Layer는 병목 에지들의 curvature를 증가시키는 방향으로 재배선을 한 것을 볼 수 있다.
    
- **GAP-Layer**
    
    : 메인 아이디어에서 살펴본 바와 같이, GAP-Layer 문제는 $\lambda_2:\tilde{A}\rightarrow\mathbb{R}^{+}$ 를 최소화하면서 네트워크 위상을 보존하는 인접행렬 $\tilde{A}$를 찾는 문제로 귀결되게 됩니다. 
    : 특히, DiffWire(Arnaiz-Rodriguez et al. ‘22)의 저자들은 이를 $F_{fiedler}$을 경사하강법(*gradient descent*)으로 푸는 방식으로 접근하였습니다.
    : 구체적으로 $F_{fiedler}$는 다음과 같이 정의되며, 이 값을 $\tilde{A}$에 관하여 최적화 시켜 GAP-Layer를 학습시키게 됩니다.
    
    $$
    F_{fiedler}=\|\tilde{A}-A\|_{_F} + \alpha(\lambda_2(\tilde{A}))^2
    $$
    
    - **유도 과정 (technical details)**
        
        $F_{fiedler}$를 경사하강법으로 최적화 하기 위해서는 $\nabla_{\tilde{A}}\lambda_2$의 값을 구할수 있아야 합니다. 그러나, $\tilde{A}$의 함수로 표현되는 $\lambda_2$, 즉, $\lambda_2(\tilde{A})$는 일반적으로 주어지지 않습니다. 따라서, 저자는 $\lambda_2(\tilde{A})$를 구하기 위해 (1) $\tilde{A}$를 $A$와 같은 $\lambda_2$의 공유하는 인접행렬이라 가정하고, (2) spectral clustering을 이용해 ratio cut문제를 풀 수 있다(von Luxburg, '07)는 사실을 활용하였습니다. 
        
        - $\tilde{A}$의 그래프가 연결되어있다(*connected*)고 가정하면, $\tilde{\lambda}_1=0$, $\mathbf{f}_1=\mathbf{1}$, $\tilde{\lambda}_2\neq 0$
        - 따라서, $\tilde{\lambda}_2 = \lambda_2 = Tr[\tilde{U}^T\tilde{L}\tilde{U}]\;\; s.t.\;\; \tilde{U}=[\mathbf{f}_1, \mathbf{f}_2]\in\mathbb{R}^{n\times 2},\;\; \tilde{U}\tilde{U}^T=\mathbf{I}_2$
        - $\nabla_{\tilde{A}}\lambda_2 = diag(\textbf{f}_2\textbf{f}_2^T)\textbf{1}\textbf{1}_2 - \textbf{f}_2\textbf{f}_2^T$  (Kang & Tong. '19)
        - $\nabla_{\tilde{A}}F_{fiedler} = 2(\tilde{A}-A) + \frac{\alpha}{2}(diag(\textbf{f}_2\textbf{f}_2^T)\textbf{1}\textbf{1}_2 - \textbf{f}_2\textbf{f}_2^T)\times\lambda_2$
        
        즉, $\tilde{A}$의 Fiedler vector인 $\textbf{f}_2$를 구할 수 있다면, $F_{fiedler}$를 최소화하는 (따라서, $\lambda_2(\tilde{A})$를 최소화 하게 되는) $\tilde{A}$를 구할 수 있게 됩니다.
        
        위 가정들에 따라 $\textbf{f}_2$는 근사적으로 $A$를 인접행렬로 가지는 그래프의 ratio cut으로 둘 수 있게됩니다. 또한, 그래프 ratio cut 문제는 다음과 같은 최적화 문제로 재구성 될 수 있습니다 (Gallier. ‘22).
        
        - 어떤 그래프가 K개의 그룹($\{V_1, V_2, \cdots, V_k\}$)으로 구성되어 있다고 가정할때, 다음과 같은 파티션 행렬 $S$를 정의해 볼 수 있습니다
            - $S_i^j = \begin{cases}
            1 ,& v_i \in V_j\\
            0 ,& v_i \notin V_j
            \end{cases}$
        - 또한, 위와 같이 정의된 $S$에 대해서 다음이 성립하게 됩니다.
            - $Cut(V_j, \bar{V_j}) = S^jLS^j$
            - $|V_j|=S^jS^j$
        - 따라서, ratio cut문제는 다음과 같은 최적화 문제로 표현됩니다.
            - $min_{_S}\sum_{j=1}^k\frac{Cut(V_j, \bar{V_j})}{|V_j|} = \sum_{j=1}^k\frac{S^jLS^j}{S^jS^j}\;\; s.t.\;\; S^iS^j=0\; \; \forall i\neq j$
            - 또는, $min_{_S}Tr[S^TLS]\;\; s.t.\;\; S^TS=I$
        
        이 관계들을 이용하여, GAP-Layer에서는 입력데이터 $X$로부터 $S$를 추론해 내는 MPNN 레이어 $f(\cdot)$를 ratio cut 로스함수인 $L_{cut}$을 최소화 시키는 방법을 통해 학습시킵니다.
        
        - $L_{cut}=-\frac{Tr[S^TAS]}{Tr[S^TDS]}+\|\frac{S^TS}{\|S^TS\|_{_F}}-\frac{I}{\sqrt{2}}\|_{_F}$
        - $S_{|V|\times 2} = softmax(f(X))$
        - $L_{cut}$의 첫번째 항은 다음과 같은 관계로 부터 나왔습니다.
        - $Tr[S^TLS] = Tr[S^T(D-A)S] = Tr[S^TDS]-Tr[S^TAS]$
        - 최종적으로 GAP Layer의 로스는, $L_{GAP} = L_{cut} + L_{fiedler}$
        

## 🧑‍💻 GAP-Layer (Hands-on)

---

```python
def dense_mincut_rewiring(x, adj, s): 
		"""
    Args:
        x (dense): feature matrix: BxNxF
        adj (dense): dense adjacency matrix: BxNxN
        s (dense): GAP embedding: BxNx2
    Returns:
        Ac: new adjacency 
        mincut_loss: cut loss
        ortho_loss: orthogonality regularization
    """
    # Hyper parameters
	  k = 2 # We want bipartition to compute spectral gap
    mu = 0.01 # learning rate for \tilde{A}
    lambdaReg = 2.0 # regularization wegith: 0.5*\alpha
    
    # Batched Laplacian 
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([B,N]) 
    d = _rank3_diag(d_flat) # d torch.Size([B,N,N]) 
    L = d - adj

		# Fiedler vector approximation
    s = torch.softmax(s, dim=-1)
    fiedlers = approximate_Fiedler(s)

    # Recalculate
    der = derivative_of_lambda2_wrt_adjacency(fiedlers)
    fvalues = fiedler_values(adj, fiedlers)
    
    # Calculate \tilde{A}
    Ac = adj.clone()
    for _ in range(5):
      partialJ = 2*(Ac-adj) + 2*lambdaReg*der*fvalues.unsqueeze(1).unsqueeze(2)
      dJ = partialJ + torch.transpose(partialJ,1,2) - 
					 torch.diag_embed(torch.diagonal(partialJ,dim1=1,dim2=2))
      # Update adjacency
      Ac  = Ac - mu*dJ
      Ac = torch.softmax(Ac, dim=-1)
      Ac = Ac*adj
		
    # MinCUT regularization
		## calculate Tr[S.T*A*S]
		mincut_num = _rank3_trace(
				torch.matmul(torch.matmul(s.transpose(1, 2), adj), s))
		## calculate Tr[S.T*D*S]
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) 
    ## calculate the average cut loss
    mincut_loss = torch.mean(-(mincut_num / mincut_den))
    ## Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss) 
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))  
    ortho_loss = torch.mean(ortho_loss) 
    
    return  Ac, mincut_loss, ortho_loss
```

- **Arguments**
    
    : `x`: 그래프 피쳐 매트릭스로, BxNxF 차원을 가집니다.
    : `adj`: 원본 그래프의 인접행렬이며, BxNxN 차원을 가집니다.
    : `s`: GAP embedding을 나타내는 행렬이며, BxNx2 차원을 가집니다.
    
    ⛔ `s`의 마지막 차원이 2 인것에 주의합니다. 이는 그래프를 2개의 그룹(*i.e.*, bipartite)으로 나누고자 하는 목적에 기인합니다.
    ⛔ 위 코드는 원활한 이해를 위해 원본 GAP-Layer 코드를 간소화한 버전입니다.
    

- **Hyper parameters**
    
    : `k`: 항상 2로 고정된 값으로, bipartition을 하고자 하는 목적을 의미합니다.
    : `mu`: $\tilde{A}$의 러닝레이트(*learning rate*)이다. 즉, $\tilde{A}\leftarrow\tilde{A} -\text{mu}\cdot\nabla_{\tilde{A}}F_{fiedler}$
    : `lambdaReg`: $F_{fiedler}$에서 $\alpha$에 비례하는 값입니다.
    

- **Fiedler vector approximation**
    
    : `s`로부터 $\mathbf{f}_2$를 도출해내는 과정입니다.
    : 우선 `s`를 softmax함수를 통해 표준화 해주게 됩니다. 이 과정은 `approximate_Fiedler()` 에 필요한 것은 아니지만, `s`는 이후 로스 계산에도 쓰이기 때문에, 모델 학습을 안정화 해주는 효과를 줍니다.
    : `approximate_Fiedler()` 는 `s`를 Fiedler vector로 변환해주는 함수이며 $s_i^j$를 `s[b]` $\in\mathbb{R}^{N\times 2}$의 ($i,j$) 컴포넌트라 할 때, 변환 결과는 다음과 같은 값을 가집니다:
    
    - $\mathbf{f}_2(i) \in\mathbb{R}^N= \begin{cases}1/\sqrt{N},\;\;\;\;\; s_i^1\geq s_i^2\\ -1/\sqrt{N},\;\; s_i^1<s_i^2\end{cases}$

- **Recalculate**
    
    : `derivative_of_lambda2_wrt_adjacency()`는 $\nabla_{\tilde{A}}\lambda_2$를 Fiedler vector $\mathbf{f}_2$를 이용해서 계산하는 함수입니다
    
    - $\nabla_{\tilde{A}}\lambda_2 = diag(\textbf{f}_2\textbf{f}_2^T)\textbf{1}\textbf{1}_2 - \textbf{f}_2\textbf{f}_2^T$.
    
    : `fiedler_values()`는 Fiedler value ($\lambda_2$)를 계산하는 함수입니다. 이때도 마찬가지로 $\mathbf{f}_2$를 이용해 계산하며, 구체적으로는 다음과 같이 Dirichlet Energy를 활용해서 구할 수 있습니다.
    
    - $\lambda_2 = N\times \|\frac{\mathbf{f}_2^T L \mathbf{f}_2}{\mathbf{f}_2^T L_c \mathbf{f}_2}\|$
    - $L$은 $A$ 그래프의 Laplacian, $L_c$는 complete graph의 Laplacian.

- **Calculate $\tilde{A}$**
    
    : 재배선된 그래프의 인접행렬 $\tilde{A}$를 경사하강법으로 구하는 부분입니다. 
    
    : 경사하강법의 기본적인 구조인 $\tilde{A}\leftarrow\tilde{A}-\eta\cdot\nabla_{\tilde{A}}L_{GAP}$에서 다음과 같은 유도/변형이 이루어 집니다.
    
    - $\nabla_{\tilde{A}}L_{GAP}=\nabla_{\tilde{A}}(L_{cut} + L_{Fiedler}) = \nabla_{\tilde{A}}L_{Fiedler}$
    - $\nabla_{\tilde{A}}L_{Fiedler}=2(\tilde{A}-A) + \frac{\alpha}{2}(diag(\textbf{f}_2\textbf{f}_2^T)\textbf{1}\textbf{1}_2 - \textbf{f}_2\textbf{f}_2^T)\times\lambda_2$
    - $\nabla_{\tilde{A}}^*L_{Fiedler}=\nabla_{\tilde{A}}L_{Fiedler}+(\nabla_{\tilde{A}}L_{Fiedler})^T - Diag(L_{Fiedler})$
    - $\tilde{A}^{(temp)}\leftarrow\tilde{A}^{(old)}-\eta\cdot\nabla^*_{\tilde{A}}L_{Fiedler}$
    - $\tilde{A}^{(new)}\leftarrow Softmax(\tilde{A}^{(temp)})\odot A$
    
    : $\nabla_{\tilde{A}}^*L_{Fiedler}$로 변환하는 부분은 $\tilde{A}$의 대칭성 보장을 위한 휴리스틱(*heuristic*)한 방법으로 이해됩니다 (즉, 그래프가 undirected라는 가정을 하고 있습니다).
    
    : $\tilde{A}$는 딥러닝 모델의 parameter가 아니기 때문에 autograd를 통해 end-to-end방식으로 최적화 하기가 힘듭니다. 저자는 이 문제를 내부루프(*inner-iteraton or inner-loop*)방식으로 해결하였습니다.
    

- **MinCUT regularization**
    
    : $L_{cut}=-\frac{Tr[S^TAS]}{Tr[S^TDS]}+\|\frac{S^TS}{\|S^TS\|_{F}}-\frac{I}{\sqrt{2}}\|_{_F}$으로 표현되는 Mincut problem의 로스 함수를 구하는 부분입니다.
    
    : `mincut_loss`는 $-\frac{Tr[S^TAS]}{Tr[S^TDS]}$ 부분을 `ortho_loss`는 $\|\frac{S^TS}{\|S^TS\|{F}}-\frac{I}{\sqrt{2}}\|_{_F}$ 부분을 각각 표현합니다.
    

## 📚 참고문헌

---

[1] Arnaiz-Rodríguez, Adrián, Ahmed Begga, Francisco Escolano, and Nuria M. Oliver. (2022). ‘DiffWire: Inductive Graph Rewiring vi33a the LováSz Bound’. In *Proceedings of the First Learning on Graphs Conference*, edited by Bastian Rieck and Razvan Pascanu, 198:15:1-15:27. 

[2] von Luxburg, U. (2007). A tutorial on spectral clustering. *Stat Comput* **17**, 395–416. 

[3] Gallier, Jean H. (2022). “Graph Clustering Using Ratio Cuts,” CIS 5105: Fundamentals of Linear Algebra and Optimization (class lecture, University of Pennsylvania, PA, US, Fall, 2022).

[4] Kang, Jian, and Hanghang Tong. (2019). ‘N2N: Network Derivative Mining’. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)*, 861–70.
