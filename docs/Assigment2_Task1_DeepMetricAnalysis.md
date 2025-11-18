# Theoretical Deep Metric Learning
## Task 1
### 1.1
Loss Function:
$$
L = max(0, d(a, p) - d(a, n) + m)
$$
Distance inter class A and class B:
$$
d_{avg}(A, B) = \frac{1}{|A||B|}\sum_{x\in{A}, y\in{B}}d(x, y)
$$
Let 
$$
L' = d(a, p) - d(a, n) + m
$$ 
then
$$
d(a, n) = d(a, p) + m - L'
$$
Let
$$
a=x, p=x' \in A, n=y \in B
$$
we got
$$
d(x,y) = d(x, x') + m - L'
$$
we know $d(x, x') > 0$, and the training process try to reduce L' to 0, so if we set bigger m, we will get bigger inter-class distance.

To make $\mathbb{E}{L} = 0$, we need $\mathbb{E}{L'}<=0$, that means for all triplets,
$$
d(a, p) - d(a, n) + m <= 0 \\
\Rightarrow d(a, n) - d(a, p) >= m
$$
### 1.2
For the first case:
$$
L_{max} = max(0, d_{max}(a,p) - d_{min}(a,n)+m) \\
L_{max} < 0 \implies d_{max}(a,p) - d_{min}(a,n)+m < 0 \\
\implies d_{min}(a, n) > d_{max}(a,p) + m \tag{1}
$$
if m is a large number, but formula (1) still holds, that means:
1. The data separability is larger than we assume, model can easily find a embedding space that no hard or semi-hard triplets to be found. 
2. If the training process didn't implement L2-Normalization for embedding vectors distance compute, in that situation, model can find a direction that $d_{min}(a, n) > d_{max}(a, p)$, and scale the vector, to make the magnitude of $d_{min}(a, n)-d_{max}(a, p)>m$.

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