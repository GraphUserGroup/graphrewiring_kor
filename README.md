# graphrewiring_kor
translate graph-rewiring into korean. original project is https://github.com/ellisalicante/GraphRewiring-Tutorial.

**Official site** : [https://ellisalicante.org/tutorials/GraphRewiring]

**Tutorial video** : [https://www.youtube.com/watch?v=AumdG5bazhg&t=3782s]

**Graph-rewiring code** : [https://github.com/ellisalicante/GraphRewiring-Tutorial]

# Main goal of this tutorial (official view)

The main goal of this tutorial is to teach the fundamentals of **graph rewiring** and its current challenges. We will motivate the need for mathematically sound graph rewiring methods as a solution to address the main limitations of GNNs: *under-reaching*, *over-smoothing* and *over-squashing*. We will explain the two main approaches proposed in the literature to achieve graph rewiring:

- **Transductive** methods compute a new convolution matrix of each graph as a pre-processing step in order to improve the performance of the task at hand. Examples are parameterized diffusion or curvature-based approaches.
- **Inductive** methods learn new convolution matrices from training on subgraphs/graphs and then predict those in unseen graphs. Ideally, this process is fully differentiable and parameter free. We will delve into the implementation of this methods. **Hands-on** section.

In addition, we will discuss the potential that graph rewiring has to address social and ethical challenges posed by AI, and particularly as a tool to achieve **algorithmic fairness**.


| Section                                       | Content                                             |
|-----------------------------------------------|-----------------------------------------------------|
| **Motivation**                                | Graph Classification and Expressiveness             |
|                                               | Node Classification and Over-smoothingDesiderates   |
| **Introduction to Spectral Theory**           | Average Cut  Problem                                |
|                                               | Fiedler Vector                                      |
|                                               | Graph Laplacian and Dirichlet Energies              |
|                                               | Laplacian Eigenfunctions and Spectrum               |
|                                               | Spectral Theorem                                    |
|                                               | Spectral Commute Times                              |
| **Transductive Rewiring**                     | Diffusive Rewiring                                  |
|                                               | Cheeger Constant                                    |
|                                               | Curvature-Based Rewiring                            |
| **Inductive Rewiring**                        | CT and the Lovász Bound                             |
|                                               | CT and Sparsification                               |
|                                               | CT and Directional Graph Networks                   |
|                                               | CT-Layer                                            |
|                                               | - CT-Layer and its Loss function                    |
|                                               | - Learned CT Embedding and CT distances             |
|                                               | - Experiments in Graph and Node Classification      |
|                                               | - CT-Layer as Differentiable Curvature              |
|                                               | - CT and Cheeger Constant                           |
| **GAP-Layer**                                 | - Spectral Derivatives                              |
|                                               | - Learning the Fiedler vector                       |
| **Panel**                                     | 강승완 , 남궁민상, 안혜민 , 정이태 , 최민수         |
