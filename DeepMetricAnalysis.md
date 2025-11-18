# Theoretical Deep Metric Learning
## Task 1
### 1.1
>Show analytically how the choice of the margin 𝑚 affects the average inter-class distance in
the embedding space.
Derive the condition under which the expected loss 𝔼[ℒ] becomes zero, and interpret this
condition.

Loss Function:
$$
L = max(0, d(a, p) - d(a, n) + m)
$$
Distance inter class A and class B:
$$
d_{avg}(A, B) = \frac{1}{|A||B|}\sum_{a\in{A}, n\in{B}}d(a, n)
$$
Let 
$$
L' = d(a, p) - d(a, n) + m
$$ 

To make $\mathbb{E}{L} = 0$, we need $\mathbb{E}{L'}<=0$
$$
\mathbb{E}{L'} =\mathbb{E}[d(a, p)] - \mathbb{E}[d(a, n)] + m <= 0 \\
\Rightarrow \mathbb{E}[d(a, n)] - \mathbb{E}[d(a, p)] >= m
$$

With big enough m, the training process will enlarge the inter-class distance $\mathbb{E}[d(a, n)]$, and reduce the inner-class distance $\mathbb{E}[d(a, p)]$.

### 1.2
>Prove or formally justify under which circumstances a too large margin 𝑚 can lead to a
situation where no triplets produce a positive loss (i.e., ℒ = 0 for all triplets) and the training
stops.
Also discuss the opposite case, in which the margin is so small that the loss remains positive
for all triplets, preventing convergence.


For the first case:
$$
L_{max} = max(0, d_{max}(a,p) - d_{min}(a,n)+m) \\
L_{max} < 0 \implies d_{max}(a,p) - d_{min}(a,n)+m < 0 \\
\implies d_{min}(a, n) > d_{max}(a,p) + m \tag{1}
$$
if m is a large number, but formula (1) still holds, that means:
1. The data separability is larger than we assume, model can easily find a embedding space that no hard or semi-hard triplets to be found. 
2. If the training process didn't implement L2-Normalization for embedding vectors distance compute, in that situation, model can find a direction that $d_min(a, n) > d_max(a, p), and scale the vector, to make the magnitude of $d_min(a, n)-d_max(a, p)>m$.

For the second case:
$$
L_{min} = max(0, d_{min}(a,p)-d_{max}(a, n)+m) \\
L_{min} >0 \implies d_{min}(a,p)-d_{max}(a, n)+m >0 \tag{2}
$$
if (2) holds for very small m, that implies:
1. The model found a trivial solution that **collapse** all the vector to same point or a very small space:
$$
d_{min}(a,p)-d_{max}(a, n) \approx 0 \\
L_{min} \approx m
$$
2. Still we forget to set L2-Normalization for vectors, and model just scale down all the vectors.