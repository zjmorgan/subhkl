.. subhkl documentation master file

======
subhkl
======

*Solving* :math:`UB` and indexing :math:`hkl` from Laue diffraction data.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   optimization/index
   concepts/index
   installation/index

-----
About
-----

`subhkl` is a Python utility for determining the crystal orientation from a Laue diffraction image.
It currently works if the lattice parameters are already known.
Future versions may allow for determining the lattice parameters.

-----------
Conventions
-----------


In dealing with crystallographic and scattering data, several conventions are chosen out of convenience.
In crystal coordinates, the real space vector is given by

.. math:: \boldsymbol{r}=u\boldsymbol{a}+v\boldsymbol{b}+w\boldsymbol{c}

whereas the reciprocal lattice vector is

.. math:: \frac{\boldsymbol{G}_{hkl}}{2\pi}=h\boldsymbol{a}^\ast+k\boldsymbol{b}^\ast+l\boldsymbol{c}^\ast.

Cartesian axes are chosen such that the first crystal axis coincides with the first orthonormal axis.
The second orthogonal axis is in the plane of the first two crystal axes.
Finally, the third forms a right-hand perpendicular triplet.
The symbols :math:`x`, :math:`y`, and :math:`z` are reserved for the Cartesian system associated with the laboratory coordinates.

- Incident beam direction: :math:`z`
- Horizontal directional perpendicula to beam: :math:`x`
- Vertical direction: :math:`y`
