========
Concepts
========

.. math::

   \boldsymbol{d}^\ast_{hkl}=h\boldsymbol{a}^\ast+k\boldsymbol{b}^\ast+l\boldsymbol{c}^\ast

.. plot:: concepts/laue_trajectories.py
   :include-source: false
   :caption: Illustration of uncertainty line segements.

.. math::

   \begin{bmatrix}
   Q_x \\
   Q_y \\
   Q_z
   \end{bmatrix}=2\pi
   \begin{bmatrix}
   R_{11} & R_{12} & R_{13} \\
   R_{21} & R_{22} & R_{23} \\
   R_{31} & R_{32} & R_{33} \\
   \end{bmatrix}
   \begin{bmatrix}
   U_{11} & U_{12} & U_{13} \\
   U_{21} & U_{22} & U_{23} \\
   U_{31} & U_{32} & U_{33} \\
   \end{bmatrix}
   \begin{bmatrix}
   B_{11} & B_{12} & B_{13} \\
   0 & B_{22} & B_{23} \\
   0 & 0 & B_{33} \\
   \end{bmatrix}
   \begin{bmatrix}
   h \\
   k \\
   l
   \end{bmatrix}