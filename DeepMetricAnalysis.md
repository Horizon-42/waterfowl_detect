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

To make $\mathbb{E}{L} = 0$, we need $\mathbb{E}{L'}<=0$
$$
d(a, p) - d(a, n) + m <= 0 \\
\Rightarrow d(a, n) - d(a, p) >= m
$$
That means the average inter-class distance must be bigger then inner-class distance plus margin value m.
### 1.2
