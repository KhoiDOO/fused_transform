# Fused Transform with CUDA

# Setup

```bash
conda create -c conda-forge -n fusedtrans python=3.10 gxx_linux-64=13 gcc_linux-64=13

conda activate fusedtrans

conda install nvidia::cuda-toolkit==12.8.2

conda install conda-forge::cmake
```

# Run

```bash
cmake .
make

./main # simple performation
./bench # simple benchmarking
```

# 3D Homogeneous Transformation Matrix Calculation

This document outlines the detailed step-by-step derivation of the composite transformation matrix $M$ in 3D homogeneous coordinates, defined as:

$$M = S(s_x, s_y, s_z) \cdot T(t_x, t_y, t_z) \cdot R_z(\alpha) \cdot R_y(\beta) \cdot R_x(\gamma)$$

Step 1: Define the Individual Matrices

In 3D homogeneous coordinates, transformations are represented by $4 \times 4$ matrices. We first define each individual transformation matrix based on the given parameters.

1. Scaling Matrix ($S$)

$$S(s_x, s_y, s_z) = \begin{bmatrix} 
s_x & 0 & 0 & 0 \\ 
0 & s_y & 0 & 0 \\ 
0 & 0 & s_z & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

2. Translation Matrix ($T$)

$$T(t_x, t_y, t_z) = \begin{bmatrix} 
1 & 0 & 0 & t_x \\ 
0 & 1 & 0 & t_y \\ 
0 & 0 & 1 & t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

3. Rotation around Z-axis ($R_z$)

$$R_z(\alpha) = \begin{bmatrix} 
\cos\alpha & -\sin\alpha & 0 & 0 \\ 
\sin\alpha & \cos\alpha & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

4. Rotation around Y-axis ($R_y$)

$$R_y(\beta) = \begin{bmatrix} 
\cos\beta & 0 & \sin\beta & 0 \\ 
0 & 1 & 0 & 0 \\ 
-\sin\beta & 0 & \cos\beta & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

5. Rotation around X-axis ($R_x$)

$$R_x(\gamma) = \begin{bmatrix} 
1 & 0 & 0 & 0 \\ 
0 & \cos\gamma & -\sin\gamma & 0 \\ 
0 & \sin\gamma & \cos\gamma & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Step 2: Compute the Composite Rotation Matrix ($R$)

Because matrix multiplication is associative, we can group the operations. We will first compute the total rotation matrix $R = R_z \cdot R_y \cdot R_x$. We evaluate this from right to left.

2a. Multiply $R_y(\beta)$ and $R_x(\gamma)$

Let $R_{yx} = R_y \cdot R_x$.

$$R_{yx} = \begin{bmatrix} 
\cos\beta & 0 & \sin\beta & 0 \\ 
0 & 1 & 0 & 0 \\ 
-\sin\beta & 0 & \cos\beta & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix} \begin{bmatrix} 
1 & 0 & 0 & 0 \\ 
0 & \cos\gamma & -\sin\gamma & 0 \\ 
0 & \sin\gamma & \cos\gamma & 0 \\ 
0 & 0 & 0 & 1 \end{bmatrix}$$

Performing the dot product for rows and columns:

$$R_{yx} = \begin{bmatrix} 
\cos\beta & \sin\beta \sin\gamma & \sin\beta \cos\gamma & 0 \\ 
0 & \cos\gamma & -\sin\gamma & 0 \\ 
-\sin\beta & \cos\beta \sin\gamma & \cos\beta \cos\gamma & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

2b. Multiply $R_z(\alpha)$ and $R_{yx}$

Now, multiply the Z-rotation matrix by the result of the previous step: $R = R_z \cdot R_{yx}$.

$$R = \begin{bmatrix} 
\cos\alpha & -\sin\alpha & 0 & 0 \\ 
\sin\alpha & \cos\alpha & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix} \begin{bmatrix} 
\cos\beta & \sin\beta \sin\gamma & \sin\beta \cos\gamma & 0 \\ 
0 & \cos\gamma & -\sin\gamma & 0 \\ 
-\sin\beta & \cos\beta \sin\gamma & \cos\beta \cos\gamma & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Resulting in the composite rotation matrix $R$:

$$R = \begin{bmatrix} 
\cos\alpha \cos\beta & \cos\alpha \sin\beta \sin\gamma - \sin\alpha \cos\gamma & \cos\alpha \sin\beta \cos\gamma + \sin\alpha \sin\gamma & 0 \\ 
\sin\alpha \cos\beta & \sin\alpha \sin\beta \sin\gamma + \cos\alpha \cos\gamma & \sin\alpha \sin\beta \cos\gamma - \cos\alpha \sin\gamma & 0 \\ 
-\sin\beta & \cos\beta \sin\gamma & \cos\beta \cos\gamma & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Step 3: Compute Scaling and Translation ($S \cdot T$)

Next, we group the Scaling and Translation matrices together from the left side of our main equation: $M = (S \cdot T) \cdot R$.

$$S \cdot T = \begin{bmatrix} 
s_x & 0 & 0 & 0 \\ 
0 & s_y & 0 & 0 \\ 
0 & 0 & s_z & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix} \begin{bmatrix} 
1 & 0 & 0 & t_x \\ 
0 & 1 & 0 & t_y \\ 
0 & 0 & 1 & t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Multiplying a scaling matrix by a translation matrix scales the respective axes including the translation components:

$$S \cdot T = \begin{bmatrix} 
s_x & 0 & 0 & s_x t_x \\ 
0 & s_y & 0 & s_y t_y \\ 
0 & 0 & s_z & s_z t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Step 4: Final Matrix Assembly ($M$)

Finally, we multiply the combined Scale/Translate matrix by our combined Rotation matrix: $M = (S \cdot T) \cdot R$.

$$M = \begin{bmatrix} 
s_x & 0 & 0 & s_x t_x \\ 
0 & s_y & 0 & s_y t_y \\ 
0 & 0 & s_z & s_z t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix} \begin{bmatrix} 
R_{11} & R_{12} & R_{13} & 0 \\ 
R_{21} & R_{22} & R_{23} & 0 \\ 
R_{31} & R_{32} & R_{33} & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Due to the structure of the left matrix (a diagonal matrix plus a translation vector in the 4th column), multiplying it by $R$ (which has $[0, 0, 0, 1]^T$ in its 4th column) results in:

Every row $i$ of the rotational $3 \times 3$ submatrix being multiplied by $s_i$.

The 4th column simply adopting the $[s_x t_x, s_y t_y, s_z t_z, 1]^T$ vector.

Expanding this fully using our explicitly written terms from Step 2b gives the final composite transformation matrix:

$$M = \begin{bmatrix}
s_x \cos\alpha \cos\beta & s_x (\cos\alpha \sin\beta \sin\gamma - \sin\alpha \cos\gamma) & s_x (\cos\alpha \sin\beta \cos\gamma + \sin\alpha \sin\gamma) & s_x t_x \\
s_y \sin\alpha \cos\beta & s_y (\sin\alpha \sin\beta \sin\gamma + \cos\alpha \cos\gamma) & s_y (\sin\alpha \sin\beta \cos\gamma - \cos\alpha \sin\gamma) & s_y t_y \\
-s_z \sin\beta & s_z \cos\beta \sin\gamma & s_z \cos\beta \cos\gamma & s_z t_z \\
0 & 0 & 0 & 1
\end{bmatrix}$$
