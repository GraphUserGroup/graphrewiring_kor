# Lovasz-Bound-and-CT

## ✏️ Introduction

---

1. **The Road So Far…**
    - 본 튜토리얼에서 소개하는 DiffWire는 GNN에서의 over-smoothing, over-squashing 문제를 해결하기 위한 프레임워크입니다.
    - DiffWire는 그래프 sparsification과 rewiring을 포함하고 있습니다. 이 때, 수정된 그래프를 만들 때는 원래 그래프의 특성을 유지하는 것이 중요합니다. DiffWire는 스펙트럴 그래프 이론과 관련된 특성들을 유지하는 프레임워크입니다.
    - DiffWire는 파라미터 조정이 필요 없는 방식으로 sparsification, rewiring을 수행합니다. 이 프레임워크는 Lovasz Bound라는 개념에 기초를 두고 있습니다.

1. **Lovasz Bound**
    
    $$
    \left| \frac{CT_{uv}}{vol(G)}- \left( \frac{1}{d_u}+\frac{1}{d_v} \right) \right| \leq \frac{1}{\lambda_2^\prime} \frac{1}{d_{min}}
    $$
    
    - Lovasz bound는 그래프 상에서 한 노드 쌍의 히팅 타임 $H_{uv}$와 그래프 스펙트럼 사이의 관계를 나타낸 식입니다.
        - $CT_{uv}$: 노드 $u$와 $v$ 사이의 커뮤트 타임
        - $vol(G)$: 그래프 $G$의 볼륨
        - $d_u$: 노드 $u$의 degree
        - $\lambda_2^{'}$: *피들러 값 (Fiedler value)* = *스펙트럴 갭* (*Spectral Gap)* = 그래프 라플라시안의 두 번째로 작은 고윳값
        - $d_{min}$: 가장 작은 degree 값
    - Spectral gap $\lambda^{'}_2$의 값이 발산(또는 0으로 수렴)함에 따라 유효 저항(= $\frac{CT_{uv}}{vol(G)}$)의 값은 국소 저항(=$\frac{1}{d_u}+\frac{1}{d_v}$)으로 수렴(또는 그로부터 발산)합니다.
    - 본 논문에서는 이 Lovasz Bound를 반영한 GNN 레이어를 제안하고 있습니다. 따라서 Lovasz Bound의 구성 요소들을 이해할 필요가 있습니다.

1. **In this Chapter…**
    
    이번 챕터에서는 Lovasz Bound의 구성 요소들을 이해하기 위해, 다음 주제들을 깊이 파고들어 볼 예정입니다
    
    1. 피들러 벡터란 무엇인가? 어떤 특징을 가지고 있는가?
    2. 커뮤트 타임이란 어떤 특징을 가지고 있는가?
        1. 커뮤트 타임에 대한 정보를 어떻게 임베딩할 수 있을까?
        2. 커뮤트 타임 임베딩은 어떤 특징을 가지고 있을까?
    3. 유효 저항이란 무엇인가?
        1. 유효 저항을 이용해 Graph Sparsification을 하는 방법?
    

## ✏️ **Fiedler Vector**

---

1. **피들러 벡터 (Fiedler Vector)**
    - 그래프 라플라시안 행렬을 고윳값 분해했을 때, 스펙트럴 갭에 대응되는 고유벡터입니다. 다시 말하자면 $\mathbf{L}\mathbf{x}=\lambda \mathbf{x}$의 두 번째로 작은 고윳값에 해당하는 고유벡터입니다.
    - 이 피들러 벡터는 그래프 노드들의 연결 관계 정보를 반영하고 있습니다. 따라서, 피들러 벡터의 값을 바탕으로 스펙트럴 그래프 분할 문제를 풀 수 있습니다
        - 쉽게 말해서 그래프 노드들을 두 그룹으로 나눠야 한다고 하면, 각 노드에 대응하는 피들러 벡터들을 살펴봅니다. 피들러 벡터가 양수/음수인 노드끼리 묶으면 스펙트럴 정보를 반영한 노드 분할을 하게 됩니다.
        - ex) 피들러 벡터가 $(0.415, 0.309, -0.221, 0.221, -0.794)$라면
            - $1, 2, 4$번 노드가 같은 그룹
            - $3, 5$번 노드가 같은 그룹
    - 수학적으로 말하자면, 그래프의 노드들을 1차원의 잠재적 공간에 임베딩한다고 합시다. 거리의 제곱의 총합을 손실 함수로 사용했을 때, 이를 최소화하는 최적의 해가 피들러 벡터입니다.
        - 자세한 설명
            
            그래프를 1차원 선상에 임베딩한다고 할 때, 노드 $i$의 임베딩 값을 $x_i$라고 합시다. 
            
            노드 $i$와 $j$가 엣지로 연결되어 있다면 $x_i$와 $x_j$는 비슷한 값을 가져야 할 것입니다. 반대로 두 노드 사이에 엣지가 없다면  $x_i$와 $x_j$의 차이가 커야 합니다. 
            
            즉, 우리는 다음 the sum of the squared distances가 최소인 임베딩 값을 찾아야 합니다.
            
            $$
            \Delta^2=\frac{1}{2}\sum_{ij} A_{ij} (x_i-x_j)^2
            $$
            
            하지만 이것만으로는 충분치 않습니다.
            
            1. 위 식에서는 $x_i$가 단 하나의 해를 가지지 않습니다. 예를 들어, 모든 $x_i$에 상수를 더한다 해도 $\Delta^2$의 값은 그대로 유지될 겁니다. 
            2. 모든 노드의 임베딩 값이 같으면 자연스럽게 최소값이 만들어집니다 ($\Delta^2=0)$. 하지만 우리가 원하는 임베딩은 이러면 안 되겠죠!
            
            즉, 우리한테는 두 가지 방법이 필요합니다.
            
            1. 노드 임베딩의 값을 한 곳에 고정할 방법
            2. 노드 임베딩이 하나로 수렴하지 않고 잠재적 공간 상에서 퍼지게 만들 방법
            
            그리고 약간의 제약을 더하면 이를 해결할 수 있습니다.
            
            1. The center of mass를 정해줍니다. 예를 들어, $\sum_i x_i=0$. 
            2. The sum of the squares를 어떤 0이 아닌 값으로 고정시켜줍니다. 예를 들어, $\sum_i x^2_i=1$.
            
            ---
            
            이를 바탕으로 the sum of the squared distances를 다시 써봅시다.
            
            $$
            \begin{aligned}\Delta^2=\frac{1}{2}\sum_{ij} A_{ij} (x_i-x_j)^2  &= \frac{1}{2}\sum_{ij} A_{ij} (x_i^2-2x_ix_j+x_j^2) \\&=\frac{1}{2}\sum_{i} k_ix_i^2 - \sum_{ij} A_{ij} x_ix_j + \frac{1}{2}\sum_{j} k_jx_j^2 \\ &=\sum_{ij}(k_i\delta_{ij}-A_{ij})x_ix_j \\ &=\sum_{ij}L_{ij}x_ix_j\end{aligned}
            $$
            
            Lagrange multiplier $\mu$와 $\lambda$를 이용해 앞에서 제시한 2개의 제약을 추가하면, 다음과 같이 편미분을 통해 $\Delta^2$를 최소화할 수 있습니다.
            
            $$
            \frac{\partial}{\partial x_v}\left[ \sum_{ij}L_{ij}x_ix_j + \mu \sum_i x_i + \lambda \left(1- \sum_i x_i^2\right) \right]=0
            $$
            
            이 식을 풀면,
            
            $$
            \begin{equation}\sum_{j} L_{vj}x_j+\frac{1}{2}\mu-\lambda x_v=0 \end{equation}
            $$
            
            양변을 $v$에 대해 sum하면 
            
            $$
            \sum_{vj} L_{vj}x_j+\frac{1}{2}n\mu-\lambda \sum_vx_v=0
            $$
            
            이 때, 앞선 조건에 따라 $\sum_vx_v=0$이고 $\sum_{v}L_{vj}=\sum_{v}A_{ij}-\sum_vk_v\delta_v=k_j-k_j=0$이므로 $\mu=0$입니다.
            
            이를 식 (1)에 대입하면 $\sum_{j} L_{vj}x_j-\lambda x_v=0$. vector notation을 사용하자면
            
            $$
            \mathbf{L}\mathbf{x}=\lambda \mathbf{x}
            $$
            
            즉, 그래프를 1차원 선에 임베딩했을 때, 최적의 임베딩 값은 Laplacian의 eigenvector입니다.
            
            이로부터 $\Delta^2$를 다시 쓰면 $\Delta^2=\lambda\sum_i x_i^2=\lambda$이므로, 가장 적합한 벡터는 smallest non-zero eigenvalue에 대응하는 eigenvector임을 알 수 있습니다.
            
            정리하자면, Fiedler vector는
            
            1. The sum of the squared distance를 loss function으로 사용하여
            2. 그래프를 1차원 latent space에 임베딩한 벡터로
            3. mean (the center of the mass)을 원점(0)으로 설정
            4. 임베딩 값들의 variation은 일정한 상수이도록 설정
            
            했을 때의 optimal solution입니다.
            
    

### 🧑‍💻 Create an Example Graph

```python
# Start to work on the graphs 
sizes = [100, 100]
# SBM with small gap
probsG = [[0.8, 0.01], [0.01, 0.8]]
G = nx.stochastic_block_model(sizes, probsG, seed=0)

# SBM with a larger gap
probsH = [[0.8, 0.5], [0.5, 0.8]]
H = nx.stochastic_block_model(sizes, probsH, seed=0)

# L = nx.normalized_laplacian_matrix(G)
AG = nx.adjacency_matrix(G)
LG = nx.normalized_laplacian_matrix(G)
eG, evecsG = find_eigen(LG)

AH = nx.adjacency_matrix(H)
LH = nx.normalized_laplacian_matrix(H)
eH, evecsH = find_eigen(LH)
```

- Stochastic block model을 이용하여 두 가지 예시 그래프를 생성 (size는 둘 다 `[100, 100]`)
    - $G$는 `probsG = [[0.8, 0.01], [0.01, 0.8]]`. 따라서 두 개의 블록이 잘 구별되고, gap은 작습니다
    - $H$는 `probsH = [[0.8, 0.5], [0.5, 0.8]]`. $G$와 비교해봤을 때, 블록 사이의 엣지 확률이 높으므로 gap이 큽니다

### 🧑‍💻 Plotting the Graph with **Fiedler Vectors**

```python
# 그래프와 eigenvector의 행렬이 주어졌을 때, k번째 eigenvector를 시각화
# 즉, Laplacian의 eigenvector 행렬과 k=1을 넣어주면 Fiedler vector를 시각화할 수 있음
def plot_eigenvector_over_graph(G,evecs,k, cmap="seismic", pos=None, node_size=40):
  # 행렬 evecs의 k번째 열을 뽑아냅니다
  u = np.real(evecs[:,k])
  u = np.transpose(u)
  u2 = np.squeeze(u)
  v = np.zeros(u2.shape[0])
  for i in range(u2.shape[0]):
    # eigenvector의 값이 음수일 수 있으므로, 100을 더해 양수로 바꿔줍니다
    v[i] = u2[i] + 100
    
  # k번째 eigenvector를 colormap으로 사용
  vColor = mcp.gen_color_normalized(cmap,data_arr=v)
  if pos == None: 
    pos = nx.spring_layout(G,seed=63)
  else: 
    pos = pos
  # NetworkX를 draw() 함수를 이용해 그래프 시각화
  nx.draw(G, pos, node_color=vColor, node_size=node_size, width=0.2, cmap=plt.cm.Blues)
  plt.show()
  return pos

# Laplacian L의 eigenvalue와 eigenvector를 찾는 함수입니다
def find_eigen(L): 
  # 행렬의 고윳값 분해를 할 때는 numpy의 함수 linalg.eig를 사용합니다
  e, evecs = np.linalg.eig(L.todense())
  e.shape, evecs.shape
  # e와 evec을 오름차순으로 정렬합니다
  idx =e.argsort()
  e = e[idx]
  evecs = evecs[:,idx]
  return e, evecs
```

- 그래프 라플라시안의 고유벡터
    - 앞서 설명했듯, 피들러 벡터 (그래프 라플라시안 행렬의 가장 작은 양수 고윳값에 대응되는 벡터) 값을 이용해 노드를 분류할 수 있습니다
    - 실제 그래프에서 피들러 벡터의 값이 어떻게 나타나는지, 몇 가지 플롯을 통해 확인해봅시다

```python
# gap이 작은 그래프 G의 노드에 Fiedler vector를 나타내기
degreeG = list(dict(G.degree).values())
posG = plot_eigenvector_over_graph(G,evecsG,1, node_size=degreeG)
print("Bottleneck of G is", eG[1])

# gap이 큰 그래프 H의 노드에 Fiedler vector를 나타내기
degreeH = list(dict(H.degree).values())
posH = plot_eigenvector_over_graph(H,evecsH,1, cmap="BrBG", node_size=degreeH)
print("Bottleneck of H is", eH[1])
```

![그림2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eebf381c-2e2a-4003-acb0-292639a1192b/%EA%B7%B8%EB%A6%BC2.png)

- 각 노드에 해당하는 피들러 벡터의 값이 색으로 표현되어 있습니다
    - 그래프 G의 경우 두 개의 블록이 확연히 구별됩니다. 피들러 벡터의 값도 극단적으로 나뉘어, 두 블록을 잘 분할하는 것을 확인할 수 있습니다
    - 반면 그래프 H에서는 언뜻 봤을 때 두 개의 블록이 구별되지 않습니다. 피들러 벡터 값을 보면, 중간에 해당하는 노드들(흰색)이 눈에 띕니다

```python
# 각 그래프의 Fiedler vector
FiedlerG = evecsG[:,1]
FiedlerH = evecsH[:,1]

FGVis =  np.squeeze(np.array(FiedlerG.copy()))
FHVis =  np.squeeze(np.array(FiedlerH.copy()))

# 두 개의 그래프 정보를 Pandas df로 만듦
df_data_G = pd.DataFrame(list(zip(FGVis,['G']*200,[0]*100+[1]*100)), columns=['Fiedler', 'Graph', 'Class'])
df_data_H = pd.DataFrame(list(zip(FHVis,['H']*200,[0]*100+[1]*100)), columns=['Fiedler', 'Graph', 'Class'])
df_data = pd.concat([df_data_G, df_data_H]).reset_index()

style = {'bins':30, 'kde':True, 'element':"step"}
plt.figure(figsize=(8,6))
sns.histplot(data=df_data, x='Fiedler', hue=df_data[['Class', 'Graph']].apply(tuple, axis=1), alpha=0.5, palette=["r", "b", "brown", "g"], **style)
plt.show()
```

![graph_hist.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/39bf19e1-f0f8-4cee-a429-c90376133488/graph_hist.png)

- 피들러 벡터의 분포를 히스토그램으로 나타내었습니다
    - 그래프 G의 경우, 클래스 0와 클래스 1의 피들러 벡터 값은 극과 극으로 분포되어 있습니다
    - 그래프 H의 경우, 클래스 0와 클래스 1의 피들러 벡터 값이 보다 완만하게 분포되어 있습니다

### 🧑‍💻 Plotting the Oversquashing

```python
# Message passing 함수
# oversquashing 현상을 시각화하기 위해 사용합니다
def message_passing(adj, feat, iters=1, frame_duration=0.1, gif_name='migif', vcolor='y'):
	'''
	Args:
		adj: 그래프의 adjacency matrix
		feat: 노드 피처
		iter: message passing iteration을 반복할 횟수
		vcolor: scatter plot의 colormap으로 사용
		나머지 파라미터는 gif 생성에 쓰임
	Returns:
		None. 다만 함수 실행 중에 iteration들의 gif를 생성해줍니다
	'''
  filenames = []
  for i in range(iters):
    feat = np.matmul(adj, feat)
		# axis=0 따라 정규화
    feat = (feat-feat.mean(axis=0)) /feat.std(axis=0)
    plt.scatter(feat[:,0], feat[:,1], c=vcolor, cmap='viridis')
    plt.title(f'Iteration {i}')
    if i == iters-1:
      plt.show()

		# gif 파일 만들기 위해 iteration별 이미지를 저장
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()
  
	  # gif 파일 생성
	  with imageio.get_writer(f'{gif_name}.gif', mode='I',duration= frame_duration) as writer:
	      for filename in filenames:
	          image = imageio.imread(filename)
	          writer.append_data(image)

	  # 이미지 파일 삭제
	  for filename in set(filenames):
	      os.remove(filename)
```

- 예시 그래프에 message passing 함수를 적용하여, bottleneck이 달라질 때 over-squashing 현상이 어떻게 변화하는지 관찰해봅시다

```python
# 200개의 노드에 할당할 2차원의 feature. size=)200, 2)
X = np.random.normal(size=(200,2), scale=1)
vcolor = np.zeros((200))
vcolor[100:]+=1

# 그래프 G에서의 over-squashing 현상
# G는 bottleneck이 작기 때문에 oversquashing이 크게 일어남
message_passing(AG.toarray(),(X-X.mean(axis=0)) /X.std(axis=0), iters=15,
                       frame_duration=0.5, gif_name='small_bottleneck', vcolor=vcolor)
```

![small_bottleneck.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/85d17d2e-864d-4b53-bb10-697db1cd2927/small_bottleneck.gif)

- bottleneck이 작은 그래프 G의 경우, over-squashing이 심하게 일어납니다
    - 노드 feature가 두 개의 그룹으로 나뉘며, 같은 블록 내에서는 잘 구별되지 않음

```python
# 그래프 H에서의 oversquashing 현상
# 비교적 oversquashing이 작음
message_passing(AH.toarray(),(X-X.mean(axis=0)) /X.std(axis=0), iters=15,
                       frame_duration=0.5, gif_name='big_bottleneck', vcolor=vcolor)
```

![big_bottleneck.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6f5d32e9-b8d4-41de-bb4c-32a8c887edbd/big_bottleneck.gif)

- bottleneck이 큰 그래프 H의 경우, over-squashing이 비교적 덜 일어납니다
    - 노드 feature가 두 개의 그룹으로 나뉘지만, 같은 블록 내에서는 넓게 퍼져있음

<aside>
💡 $\lambda_2{'}$이 작을 수록, 즉 그래프에서의 병목 현상이 심할 수록 over-squashing 현상이 크게 일어납니다.

</aside>

## ✏️ Commute Time

---

1. **히팅 타임(Hitting Time) & 커뮤트 타임(Commute Time)**
    - 커뮤트 타임이란 그래프 상에서 랜덤 워커가 한 노드 $u$에서 다른 노드 $v$ 사이를 왔다갔다 하는 데 걸리는 평균 시간입니다
        - 랜덤 워크란 그래프의 한 노드에서 연결되어 있는 노드 중 하나를 무작위로 골라 이동하는 랜덤 프로세스를 말합니다.
        - 이런 랜덤 워커가 한 노드 $u$에서 다른 노드 $v$로 가는 데 걸리는 시간을 히팅 타임이라고 합니다. 그리고 평균적인 히팅 타임은 $H_{uv}$로 나타냅니다. 즉, $CT_{uv}=H_{uv}+H_{vu}$입니다.
    - 두 노드 사이의 최단 거리와는 달리, 두 노드를 연결하는 경로가 많을 수록 커뮤트 타임은 감소합니다.
2. **유효 저항 (Effective Resistance)**
    - 유효 저항 $R_e$는 두 노드 사이의 bottleneck을 나타내는 지표입니다. 전자공학 분야에서 두 지점 사이의 유효 저항을 구하는 문제에서 비롯된 개념입니다.
    - $R_e$는 두 노드 사이의 커뮤트 타임에 비례합니다. 정확히는 $R_{uv}=\frac{CT_{uv}}{vol(G)}$입니다.
    - 뒤에서 우리는 graph sparsification 문제를 해결하는 데 유효 저항을 사용할 겁니다.

### 🧑‍💻 Calculating Hitting Paths & Effective Resistance

```python
import networkx as nx
import numpy  as np

# 시작 노드와 끝 노드가 주어졌을 때, Hitting path 구하는 함수
def get_hitting_path(G, T, start, stop):
	"""
	Args:
		G: 그래프
		T: 그래프 G의 transition matrix
		start: Hitting path의 시작 노드
		end: Hitting path의 끝 노드 

	Returns:
		hittimg_time: hitting path의 경로 길이 (또는 이동 횟수)
		Visited_nodes: Hitting path에 포함된 노드들
		Visited_edges: Hitting path에 포함된 엣지들
	"""

  # print("start, stop", start, stop)
  # Visited lists: nodes and edges  
  Visited_nodes = list()
  Visited_edges = list()
  # Current node 
  old_node = start
  Visited_nodes.append(old_node)
  # Get all sorted elements and indices in each step 
  hit = False
  while not hit:
    # Choose the old_node-th row of T which is row-stochastic
    q = T[old_node,:]
    new_node = np.random.choice(G.number_of_nodes(), 1, replace = True, p = q)[0]
    # print(new_node)
    Visited_nodes.append(new_node)
    Visited_edges.append((old_node,new_node))
    if new_node == stop:
      hit = True
    else: 
      old_node = new_node
  hitting_time = len(Visited_nodes) - 1
  return hitting_time, Visited_nodes, Visited_edges
```

- 하나의 노드 `start`에서 다른 노드 `stop`까지의 히팅 패스를 구하는 함수입니다.
    - `old_node = start`: 처음 시작 노드를 `start`로 정합니다
    - `new_node = np.random.choice(G.number_of_nodes(), 1, replace = True, p = q)[0]`: `old_node`와 연결된 노드 중에서 하나를 무작위로 선택합니다
    - `Visited_nodes.append(new_node)` / `Visited_edges.append((old_node,new_node))`: 선택한 노드를 `Visited_nodes`에, 원래 노드와의 엣지를 `Visited_edges`에 기록합니다
    - `new_node`가 `stop`이면 iteration을 멈추고 그때까지의 경로 정보를 반환합니다

```python
import networkx as nx
import numpy  as np

# 예시 그래프: 바벨 그래프
G = nx.barbell_graph(10, 0)
nx.draw(G)
# let networkx return the adjacency matrix A
#G = H
A = nx.adjacency_matrix(G)
A = A.todense()
A = np.array(A, dtype = np.float64)
# let's evaluate the degree matrix D
D = np.diag(np.sum(A, axis=0))
# ...and the transition matrix T
T = np.dot(np.linalg.inv(D),A)

nsamples = 100
All_visited_nodes_go = list()
All_visited_nodes_back = list()
All_visited_edges_go = list()
All_visited_edges_back = list()
All_hitting_times_go = list()
All_hitting_times_back = list()

# 총 100번 0->18 hitting path와 18->0 hitting path 구하기
for k in range(nsamples):
  # 시작 -> 끝 가는 hitting path 구하기
  hitting_time_go, Visited_nodes_go, Visited_edges_go = get_hitting_path(G, T, 0, 19)
  All_visited_nodes_go.extend(Visited_nodes_go)
  All_visited_edges_go.extend(Visited_edges_go)
  All_hitting_times_go.append(hitting_time_go)
  
	# 끝 -> 시작 돌아오는 hitting path 구하기
  hitting_time_back, Visited_nodes_back, Visited_edges_back = get_hitting_path(G, T, 19, 0)
  All_visited_nodes_back.extend(Visited_nodes_back)
  All_visited_edges_back.extend(Visited_edges_back)
  All_hitting_times_back.append(hitting_time_back)

print(All_hitting_times_go)
print(All_hitting_times_back)

# Hitting time은 평균 hitting path
Huv = np.mean(All_hitting_times_go)
Hvu = np.mean(All_hitting_times_back)

# Commute time = 왕복 Hitting time의 총합
# Effective Resistance = Commute time을 그래프 볼륨으로 나누어서 구함
Ruv = (Huv + Hvu)/graph_vol(G)
print("Effective resistance", Ruv)
```

- 바벨 그래프를 예시로 히팅 타임과 유효 저항을 구해봅시다
    - `for … loop`: 히팅 타임은 히팅 패스의 평균 길이입니다. 앞서 정의한 `get_hitting_path()` 함수를 100회 반복하여 평균값을 구합니다
    - 위 코드에는 굳이 언급하지 않지만, 커뮤트 타임은 왕복 히팅 타임의 총합입니다. 코드로 나타내자면 `CTuv = Huv + Hvu`로 나타낼 수 있겠죠?
    - `Ruv = (Huv + Hvu)/graph_vol(G)`: 유효 저항은 $R_{uv}=\frac{CT_{uv}}{vol(G)}$입니다.

### 🧑‍💻 Effective Resistance as an Edge Density

```python
# ID가 작은 노드가 앞에 오도록 엣지 변형
All_visited_edges_go_T = list()
for e in All_visited_edges_go: 
  if e[0] > e[1]:
    All_visited_edges_go_T.append((e[1],e[0]))
  else: 
    All_visited_edges_go_T.append((e[0],e[1]))

All_visited_edges_back_T = list()
for e in All_visited_edges_back: 
  if e[0] > e[1]:
    All_visited_edges_back_T.append((e[1],e[0]))
  else: 
    All_visited_edges_back_T.append((e[0],e[1]))

All_visited_edges_T = list()
for e in All_visited_edges_go_T: 
  All_visited_edges_T.append(e)
for e in All_visited_edges_back_T: 
  All_visited_edges_T.append(e)

# 원래 얻은 visited edges
print("Go ", All_visited_edges_go)
print("   ", All_visited_edges_go_T)
print(len(All_visited_edges_go_T))

print("Bck", All_visited_edges_back)
print("   ", All_visited_edges_back_T)
print(len(All_visited_edges_go_T))

print("All", All_visited_edges_T)
print(len(All_visited_edges_T))
print("Ori", G.edges())
print(len(G.edges()))

# 랜덤 워커가 방문한 엣지와 그 횟수를 히스토그램으로 표현
edgeHist_go = list()
for e in G.edges():
  edgeHist_go.append(All_visited_edges_go_T.count(e))
print("Hist go ", edgeHist_go)

edgeHist_back = list()
for e in G.edges():
  edgeHist_back.append(All_visited_edges_back_T.count(e))
print("Hist bck", edgeHist_back)

edgeHist_all = list()
for e in G.edges():
  edgeHist_all.append(All_visited_edges_T.count(e))
print("Hist all", edgeHist_all)

print("Hist_lengths: go, back, all:", len(edgeHist_go), len(edgeHist_back), len(edgeHist_all))
```

- 유효 저항을 확인하기 위해, 랜덤 워커가 방문한 빈도수를 엣지에 표시해봅시다
    - 엣지가 동일한지 확인할 수 있도록, 랜덤 워커가 방문한 엣지의 목록을 약간 변형합니다 (i.e. `(0,19)`와 `(19,0)`는 방향만 다를 뿐 동일한 엣지. 이를 작은 숫자가 앞에 오도록 하여 `(0,19)`로 통일시킴)
    - `All_visited_edges_T`: start → stop 경로와 stop → start 경로를 합친 리스트. 즉, 커뮤트 패스들의 총합

```python
# edge를 visiting histogram에 따라 색칠
# node는 시작점과 끝점만 색칠
# 앞에서 정의했던 plot_node_intensity_over_graph 함수와 비슷합니다
def plot_edge_intensity_over_graph(G,start, stop, edge_Hist):
  # Info to plot in G.edges()
  u = edge_Hist
  n = 0.25*np.ones(G.number_of_nodes())
  n[start] = 1
  n[stop] = -1
   
  # Create the color maps 
  eColor =mcp.gen_color_normalized(cmap="BrBG",data_arr=u)
  nColor =mcp.gen_color_normalized(cmap="seismic",data_arr=n)
  #pos = nx.spring_layout(G,seed=63)
  pos = nx.spring_layout(G)
  options = {
    "node_color": nColor, #"#A0CBE2",
    "edge_color": eColor,
    "width": 2,
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
  }
  # Draw the graph
  nx.draw(G, pos, **options)
	sm = plt.cm.ScalarMappable(cmap='BrBG', norm=plt.Normalize(vmin = min(edge_Hist), vmax= max(edge_Hist)))
  sm._A = []
  plt.colorbar(sm)
  plt.show()
```

- `edge_Hist`라는 엣지별 점수를 받아, 그걸 바탕으로 엣지 색깔을 다르게 하는 함수입니다
- 앞에서 정의했던 `plot_node_intensity_over_graph`의 엣지 버전입니다

```python
# start -> stop의 hitting path 히스토그램
plot_edge_intensity_over_graph(G,start=0, stop=19, edge_Hist=edgeHist_go)

# stop -> start의 hitting path 히스토그램
plot_edge_intensity_over_graph(G,start=19, stop=0, edge_Hist=edgeHist_back)

# 왕복 path를 합친 히스토그램 (즉, commute path) 
# 가운데 엣지의 bottleneck이 눈에 띈다
plot_edge_intensity_over_graph(G,start=0, stop=19, edge_Hist=edgeHist_all)
```

![그림5.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66a0ae20-be83-4dd9-bc83-08f11ae7c1f9/%EA%B7%B8%EB%A6%BC5.png)

- 검붉은 노드가 `start`, 검푸른 노드가 `stop`입니다. 둘 사이의 hitting path를 그려보았을 때 각 엣지를 방문한 횟수를 색으로 표현하였습니다

![commute_path.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2bbfb01e-3d5f-44a4-9683-d6380eda00c5/commute_path.png)

- 둘을 합친 그래프, 즉 커뮤트 패스를 표현한 그래프입니다. 커뮤트 타임을 기준으로 했을 때는 상대적으로 가운데 엣지의 방문 횟수가 높게 나왔습니다

## ✏️ Commute Time Embeddings

---

1. **커뮤트 타임 임베딩 (Commute Time Embeddings)**
    - 커뮤트 타임 임베딩(이하 CTE)은 각 노드들의 커뮤트 타임 정보와 유클리드 공간에서의 거리 정보를 반영한 임베딩 벡터입니다
    - 다른 스펙트럴 임베딩 (i.e. 히트 커널, diffusion map)과는 달리 파라미터 조정이 필요 없다는 장점이 있습니다.
    - CTE는 degree로 down-weight한 라플라시안 정규화의 결과물입니다. 이를 수식으로 나타내면 다음과 같습니다.
        
        $$
        \mathbf{Z}\coloneqq \sqrt{vol(G)}\Lambda^{-1/2}\mathbf{F}^T
        $$
        
        - 증명 (TBA)

### 🧑‍💻 CTE Implementation

```python
# 그래프 볼륨: 노드 degree의 총합 (2 * edge)
def graph_vol(G):
    A = nx.adjacency_matrix(G)
    D = A.sum(axis=1)
    D = D.squeeze()
    d = np.zeros(G.number_of_nodes())
    for i in range(G.number_of_nodes()):
        d[i] = D[i]
    vol = d.sum()
    return vol

# CTE 구하는 함수
# NetworkX를 이용해 위에서 소개한 식을 그대로 구현하였습니다
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
    # Embedding is in the columns
    return CTE
```

### 🧑‍💻 Visualizing CTE in Scatter Plot & KDE Plot

```python
# 데이터 포인트 분포 시각화하는 함수
def visualize(z, color, size=10, save=False, title='img', plot_type='kde', cmap='seismic'):
		'''
		Args:
			z: CTE matrix를 2차원으로 차원축소한 2D array
			size: (scatter plot의 경우) 데이터 포인트의 크기
			plot_type: kde / scatter 중 선택
			cmap: colormap
			다른 변수들은 이미지 저장에 사용
		'''
    plt.figure(figsize=(5,5))
    # KDE plot 그리기
    # KDE에 대한 설명은 https://darkpgmr.tistory.com/147 참조
    vColor =mcp.gen_color_normalized(cmap,data_arr=color)
    if plot_type=='kde':
      sns.kdeplot(x=z[:, 0], y=z[:, 1], s=size, color=vColor, cmap=cmap)
    elif plot_type=='scatter':
      sns.scatterplot(x=z[:, 0], y=z[:, 1], s=size, c=vColor, cmap=cmap)
    else:
      raise NotImplementedError("Wrong plot type")
    # plt.colorbar()
    plt.xticks([], [])
    plt.yticks([], [])
    if save:
        plt.savefig(title+'.png') 
    plt.show()
```

- CTE 행렬을 2차원 공간에 플롯하는 함수입니다
    - t-SNE 등을 이용해 CTE 행렬을 차원 축소하고 이를 변수로 넣어줍니다
    - `plot_type='kde'`인 경우 KDE plot, `plot_type='scatter'`인 경우 scatter plot을 그려줍니다. 플롯을 그리는 데에는 `seaborn` 함수를 사용합니다.

```python
# TSNE Visualization of G
# CAUTION, I had to do the transpose so that each row was a node instead of a column
spectral_CTE_G = np.real(commute_times_embedding(G,eG, evecsG).T)
# change perplexity and early_exageration to get the embedding right
spec_cte_2d = TSNE(n_components=2, learning_rate='auto',
               init='random', perplexity=6, early_exaggeration=100, random_state=200).fit_transform(spectral_CTE_G.real)
# Any way you want to calculate the degree of nodes for color or size
degree = np.diag(LG.todense())
visualize(spec_cte_2d , color = degree)
visualize(spec_cte_2d , color = degree, size= (degree-degree.min())/(degree.max()-degree.min())*50, plot_type='scatter')

# TSNE Visualization of H
spectral_CTE_H = np.real(commute_times_embedding(H,eH, evecsH).T)
spec_cte_2d = TSNE(n_components=2, learning_rate='auto',
               init='random', perplexity=6, early_exaggeration=100, random_state=200).fit_transform(spectral_CTE_H.real) 
degree = np.diag(LG.todense())
visualize(spec_cte_2d , color = degree, size= degree/2)
visualize(spec_cte_2d , color = degree, size= (degree-degree.min())/(degree.max()-degree.min())*40, plot_type='scatter')
```

![그림4.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a42dd960-077c-495c-8552-999396d00a74/%EA%B7%B8%EB%A6%BC4.png)

- 그래프 $G$와 $H$의 CTE를 2차원 공간에 표현해보았습니다
    - 그래프 $G$에서는 노드들이 두 개의 그룹으로 나눠집니다. KDE plot에서도 두 개의 봉우리가 있는 것을 볼 수 있습니다
    - 반면 그래프 $H$에서는 노드들이 그룹으로 잘 나눠지지 않습니다. KDE plot에서도 하나의 커다란 봉우리가 있습니다
- scatter plot을 그릴 때 사용한 변수는 다음과 같습니다
    - `color = degree`: scatter plot의 데이터 포인트 색깔은 각 노드의 degree에 따라 결정됩니다. degree가 높을 수록 빨간색에 가깝고, 낮을 수록 파란색에 가깝습니다
    - `size = (degree-degree.min())/(degree.max()-degree.min())*40`: 데이터 포인트의 사이즈는 각 노드의 degree를 min-max feature scaling 하여 정해집니다

### 🧑‍💻 Visualizing CT Distance in Heatmap & Histogram

```python
# Distance Matrix
CTE = np.real(commute_times_embedding(G,eG, evecsG))
CT = pdist(np.transpose(CTE), 'euclidean')
CT = squareform(CT)
plt.imshow(CT, alpha=0.8, cmap="seismic")

CTE = np.real(commute_times_embedding(H,eH, evecsH))
CT = pdist(np.transpose(CTE), 'euclidean')
CT = squareform(CT)
plt.imshow(CT, alpha=0.8, cmap="seismic")
```

![그림1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51969b9c-5845-4707-86e6-0171c467ef07/%EA%B7%B8%EB%A6%BC1.png)

- 앞서 살펴봤듯, CTE는 CT의 유클리드 거리 정보를 반영한 벡터입니다
- Distance matrix를 그려보면, $G$는 노드들이 두 개의 그룹으로 나눠진 반면 $H$는 그룹의 구분이 확실치 않습니다

```python
import torch 

# get spectral ct dist
spectral_CT_dist_G = pdist(spectral_CTE_G.real, 'euclidean')
spectral_CT_dist_G = torch.Tensor(squareform(spectral_CT_dist_G))

spectral_CT_dist_H = pdist(spectral_CTE_H.real, 'euclidean')
spectral_CT_dist_H = torch.Tensor(squareform(spectral_CT_dist_H))

# get vectorized UPPER TRIANGULAR ct dist
idx = torch.Tensor(spectral_CT_dist_G).triu().nonzero().T
spec_CT_dist_triu_G = spectral_CT_dist_G[idx[0], idx[1]]
spec_CT_dist_triu_G = spec_CT_dist_triu_G / spec_CT_dist_triu_G.sum()

idx = torch.Tensor(spectral_CT_dist_H).triu().nonzero().T
spec_CT_dist_triu_H = spectral_CT_dist_H[idx[0], idx[1]]
spec_CT_dist_triu_H = spec_CT_dist_triu_H / spec_CT_dist_triu_H.sum()

# Plotting histograms using seaborn
SpecGVis =  np.squeeze(np.array(spec_CT_dist_triu_G.clone()))
SpecHVis =  np.squeeze(np.array(spec_CT_dist_triu_H.clone()))

df_prob_G = pd.DataFrame(list(zip(SpecGVis,['G']*len(spec_CT_dist_triu_G))), columns=['EdgeP', 'Graph'])
df_prob_H = pd.DataFrame(list(zip(SpecHVis,['H']*len(spec_CT_dist_triu_G))), columns=['EdgeP', 'Graph'])
df_prob = pd.concat([df_prob_G, df_prob_H]).reset_index()

style = {'bins':30, 'kde':True, 'element':"step"}
plt.figure(figsize=(8,6))
sns.histplot(data=df_prob, x='EdgeP', hue=df_prob[['Graph']].apply(tuple, axis=1), alpha=0.5, palette=["r", "g"], **style)
plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1e5e508a-2985-487b-a1ed-1f8d46c21a57/Untitled.png)

- 같은 내용을 히스토그램으로도 확인해봅시다. CTE의 유클리드 거리를 히스토그램으로 나타냈습니다.
- $G$는 CT가 짧은 노드쌍, CT가 긴 노드쌍이 명백히 나눠집니다. 반면 $H$는 평균적인 CT를 가진 노드쌍이 많고, 너무 크거나 작은 노드쌍은 적습니다.

---

1. **CTE & Eigenvector**
    
    $$
    \mathbf{Z}\coloneqq \sqrt{vol(G)}\Lambda^{-1/2}\mathbf{F}^T
    $$
    
    - 앞서 CTE를 구할 때 사용한 식을 보면, $F^T$ 항이 있는 것을 볼 수 있습니다. 즉, CTE는 고윳값으로 down-scaling한 버전의 고유벡터라고 할 수 있습니다.

### 🧑‍💻 Relationship between CTE and Eigenvectors

```python
# Transpose of commute times and scaled eigenvectors
# spectral_CTE_G: CTE 매트릭스
Theta_G = spectral_CTE_G
Theta_H = spectral_CTE_H
```

- 한번 CTE 행렬과 고유벡터의 관계를 확인해볼까요?

```python
# Disc: Eigen - CTE
DiscG= np.asarray(abs(evecsG-Theta_G)).flatten()
plt.imshow(abs(evecsG-Theta_G), alpha=0.8, cmap="seismic")
plt.colorbar()
print("Min diff is", np.min(DiscG))
print("Max diff is", np.max(DiscG))
meanDiscG = np.mean(DiscG)
stdDiscG = np.std(DiscG)
print(f"Avg {meanDiscG} +- std {stdDiscG}")

DiscH= np.asarray(abs(evecsH-Theta_H)).flatten()
plt.imshow(abs(evecsH-Theta_H), alpha=0.8, cmap="seismic")
plt.colorbar()
print("Min diff is", np.min(DiscH))
print("Max diff is", np.max(DiscH))
meanDiscH = np.mean(DiscH)
stdDiscH = np.std(DiscH)
print(f"Avg {meanDiscH} +- std {stdDiscH}")
```

![그림2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f061db80-65fd-42f4-a33a-f6fa80f3e5fe/%EA%B7%B8%EB%A6%BC2.png)

- `Disc = abs(eigen - CTE)`를 히트맵으로 나타내었습니다
    - 보시다시피 대부분의 값이 0에 가깝습니다
    - 이러한 경향은 그래프 $G$와 $H$ 모두 동일합니다

```python
# Using seaborn
import pandas as pd

df_Disc_G = pd.DataFrame(list(zip(DiscG,['G']*len(DiscG))), columns=['Disc', 'Graph'])
df_Disc_H = pd.DataFrame(list(zip(DiscH,['H']*len(DiscH))), columns=['Disc', 'Graph'])
df_Disc = pd.concat([df_Disc_G, df_Disc_H]).reset_index()

style = {'bins':30, 'kde':True, 'element':"step"}
plt.figure(figsize=(8,6))
sns.histplot(data=df_Disc, x='Disc', hue=df_prob[['Graph']].apply(tuple, axis=1), alpha=0.2, palette=["r", "g"], **style)
plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a5a4b0c9-5312-487e-97f0-2e2c2278beef/Untitled.png)

- 두 그래프의 `Disc`를 가지고 히스토그램을 그려보면 마찬가지 경향성을 확인할 수 있습니다
    - 대부분의 `Disc` 값은 0이고, 전체적인 분포도 0에 치우쳐져 있습니다
    - $G$와 $H$는 굉장히 다른 그래프임에도, 고유벡터와 CTE가 비슷하다는 점은 동일합니다

## ✏️ Graph Sparsification

---

1. **Graph Sparsification**
    - Graph Sparsification은 어떤 그래프 $G$를 근사할 수 있는 sparse한 그래프 $G'$를 찾는 과정입니다. 이 때, 그래프 $G=(V,E)$와 그래프 $G'=(V,E')$는 다음과 같은 관계를 가집니다.
        - 그래프 $G'$의 node set은 그래프 $G$와 동일합니다.
        - 그래프 $G'$의 edge set은 그래프 $G$의 edge set의 부분집합입니다. $(E' \sube E)$
        - 적절한 metric에 대해 $G$와 $G'$의 값이 비슷해야 합니다. 즉, $G'$는 $G$에 대한 non-trivial statement를 할 수 있어야 합니다. 예를 들자면…
            - size of cuts (bottleneck)
            - clusters (communities)
            - distances
            - random walks
            - single / multi-commodity flows
            - electrical flows + other physical processes
            - coloring
            - Hamiltonian / Eulerian cycle
            - subgraph counts
        - $G'$는 $G$에 비해 훨씬 적은 수의 엣지를 가지고 있습니다. 따라서, $G'$를 이용한 연산은 훨씬 효율적입니다.
    - 본 튜토리얼의 목표는 spectrally similar하도록 sparsification을 하는 것입니다. 즉, $\mathbf{L}_G$와 $\mathbf{L}_{G'}$가 비슷해야 합니다.
2. **Sparsification Leads to Commute Times**
    
    <aside>
    💡 Spectrally similar sparse graph를 찾는 문제는 커뮤트 타임을 이용해 풀 수 있습니다
    
    </aside>
    
    - $G=(V,E)$로부터 $G'$를 찾아내는 샘플링 알고리즘을 생각해봅시다. 이 알고리즘은 아래와 같은 조건에 따라 엣지를 샘플링합니다.
        - $q \propto R_e$의 확률로 엣지를 샘플링
        - 노드의 갯수 $|V|$는 충분히 큼
        - $1/\sqrt{n} < \epsilon \leq 1$을 만족
        - 이 때, $O\left( \frac{n\; log\: n}{\epsilon^2}\right)$ 개의 엣지를 샘플링하면 $1/2$ 이상의 확률로 아래 근사식을 만족합니다.
        
        $$
        \forall x \in \mathbb{R}^n:(1-\epsilon) \mathbf{x}^\intercal \mathbf{L}_G \mathbf{x} \leq \mathbf{x}^\intercal \mathbf{L}_{G'} \mathbf{x} \leq (1+\epsilon) \mathbf{x}^\intercal \mathbf{L}_G \mathbf{x}
        $$
        
    - 앞서 설명했듯, $R_e$는 커뮤트 타임에 비례합니다. 따라서 CT distance에 비례해 엣지를 샘플링하면 같은 효과를 얻을 수 있습니다.

### 🧑‍💻 Sparsification by Sampling

```python
# 각 엣지의 CT distance에 비례하는 확률로 엣지 샘플링
def sparsify_graph(G, spectral_CT_dist_G):
	"""
	Args:
		G: Sparsification을 할 NetworkX 그래프
		spectral_CT_dist_G: G의 노드들 사이의 CT distance 정보를 담고 있는 행렬 (사이즈 V*V)
	Returns:
		Ghat: sparsified 된 NetworkX 그래프
	"""

  # CT 거리 마스킹
  print(G.number_of_edges())
  AG = np.asarray(nx.adjacency_matrix(G).todense())
  CTmaskedG = np.asarray(AG * spectral_CT_dist_G.triu().numpy()).nonzero()

  # 확률 분포 계산
  pGvalues = np.zeros(G.number_of_edges())
  pGvalues = spectral_CT_dist_G[CTmaskedG[0], CTmaskedG[1]] # Get the i and j cols of indexes
  print(pGvalues)
  pG =pGvalues/(pGvalues.sum())
  print(pG.sum())

  # CT 거리에 비례하여 남길 엣지 선택: O(n logn)개의 엣지만 남김
  n = G.number_of_nodes()
  selectedEdgesG = np.random.choice(G.number_of_edges(),
                                  int(n*int(np.log(n))), 
                                  replace = True, p = pG.numpy())
  selectedEdgesG = np.unique(selectedEdgesG)
  print(len(selectedEdgesG))
  print(int(G.number_of_nodes()*int(np.log(G.number_of_nodes()))))

  # 새로운 그래프 생성
  aux = torch.zeros(G.number_of_nodes()*G.number_of_nodes())
  AhatG = aux.scatter_(0,torch.Tensor(selectedEdgesG).long(),1).reshape(G.number_of_nodes(), G.number_of_nodes()).numpy()

  Ghat = nx.from_numpy_array(AhatG)
  return Ghat
```

- CT 거리 마스킹
    - `spectral_CT_dist_G` 행렬에는 모든 노드쌍의 CT distance가 들어있습니다
    - 엣지로 연결된 노드쌍의 거리 정보만 남기기 위해, $G$의 adjacency matrix으로 마스킹을 합니다.
    - 넘파이 `nonzero()` 함수를 이용해 마스킹한 결과가 0이 아닌 원소들의 인덱스를 뽑아냅니다.
- 확률 분포 계산
    - `pGvalues = spectral_CT_dist_G[CTmaskedG[0], CTmaskedG[1]]`: 엣지들의 CT 거리가 담긴 넘파이 array를 만듦
    - `pG = pGvalues/(pGvalues.sum())`: 전체 합이 1.0이 되도록 변환
- CT 거리에 비례하여 남길 엣지 선택
    - `pG` 확률값에 비례하여 $n \:log (n)$ 개의 엣지를 샘플링
- 새로운 그래프 생성
    - 선택된 엣지들을 바탕으로 새로운 adjacency matrix 생성
    - 이 새로운 adjacency matrix로 그래프 `AhatG`를 만듦

```python
Ghat = sparsify_graph(G, spectral_CT_dist_G)
posGhat = plot_eigenvector_over_graph(Ghat,evecsG,1, pos = posG)

Hhat = sparsify_graph(H, spectral_CT_dist_H)
posHhat = plot_eigenvector_over_graph(Hhat,evecsH,1, cmap="BrBG", pos = posH)
```

![그림5.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00b581e8-cd33-48ca-baa7-0250f94327e0/%EA%B7%B8%EB%A6%BC5.png)

- CT를 이용해 그래프 $G$와 $H$를 sparsification 했습니다.
    - 그래프 $G$의 경우 두 클러스터를 연결하는 (= bottleneck에 해당하는) 엣지들은 $R_e$가 높습니다. 반면 클러스터 내의 엣지는 $R_e$가 낮습니다. Sparsification을 했을 때, 주로 클러스터 내의 엣지들이 사라졌음을 확인할 수 있습니다.

---

## 📄 Reference

1. [Newman, M. (2018). *Networks*. Oxford university press.](https://www.google.co.kr/books/edition/Networks/YdZjDwAAQBAJ?hl=en&gbpv=0)
2. [Arnaiz-Rodríguez, A., Begga, A., Escolano, F., & Oliver, N. (2022). DiffWire: Inductive Graph Rewiring via the Lovasz Bound. arXiv preprint arXiv:2206.07369.](https://arxiv.org/abs/2206.07369)
3. [Spielman, D. A., & Srivastava, N. (2008, May). Graph sparsification by effective resistances. In *Proceedings of the fortieth annual ACM symposium on Theory of computing* (pp. 563-568).](https://dl.acm.org/doi/abs/10.1145/1374376.1374456)
    - [Lecture on Graph Sparsification](https://youtu.be/qXRs8-LouSQ) by Nikhil Srivastava
4. [Hahn, H.-I. (2014, January 31). Analysis of Commute Time Embedding Based on Spectral Graph. *Journal of Korea Multimedia Society*. Korea Multimedia Society. https://doi.org/10.9717/kmms.2014.17.1.034](http://www.koreascience.or.kr/article/JAKO201409864556102.page)
