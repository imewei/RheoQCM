:orphan:

# Useful QCM Info

Converted from {download}`QCMnotes.pdf <../_static/QCMnotes.pdf>` for easier
search and cross-referencing inside the Sphinx documentation. Figures from the
original PDF are referenced but not reproduced here.

## Introduction

This document accompanies the open source rheoQCM software written and
maintained by Dr. Qifeng Wang at Northwestern University.

## Background Theory

As the name implies, the quartz crystal microbalance (QCM) is most often
employed as a high sensitivity mass-balance. The instrument relies on the
piezoelectric nature of quartz to measure changes in resonance frequency caused
by the application of a material to the quartz surface. An adsorbed mass or
deposited film results in a decrease in resonant frequency of the quartz
crystal by an amount, $\Delta f_{sn}$, that is given by the well-known Sauerbrey
expression [1]:

$$
\Delta f_{sn} = \frac{2 n f_1^2}{Z_q} \Delta M_A
             = \frac{2 n f_1^2}{Z_q} \rho d
\tag{1}
$$

Here, $n$ is the order of the resonance harmonic, $f_1$ is the fundamental
resonance frequency of quartz (5 MHz), $Z_q$ is its acoustic shear impedance
(8.84 x 10^6 kg/m^2 s for AT cut quartz), $\Delta M_A$ is the mass per unit
area of the film, $\rho$ is its density and $d$ is its thickness. Note that
$\rho d$ is the mass per unit area.

Originally, the technique was designed for use in an air or vacuum environment,
but today the QCM is often used to characterize molecules, particles and
thin-films in liquid media. Applications include studies of adsorption [2-10],
self-assembly [2-4], cell-substrate interactions [11], and electrochemistry
[12, 13].

The QCM consists of a single-crystal quartz disc sandwiched between two
electrodes. Due to the piezoelectric nature of quartz, the material will
oscillate transversely if an alternating potential is applied across the
electrodes, propagating a shear wave through the disc. If the frequency of
oscillation is near to the acoustic resonance frequency of quartz, a standing
shear wave will generate across the disc, inducing a peak in the system
conductance [14, 15]. For the AT-cut crystals used throughout this work, the
fundamental resonant frequency is near 5 MHz. The conductance also peaks at odd
harmonics ($n$) of the fundamental resonant frequency, with resonances at
15 MHz ($n=3$), 25 MHz ($n=5$), etc. If the conductance is plotted in the
frequency domain, Lorentzian curves can be fit to the conductance peaks to
provide two values: the resonance frequency ($f_n$), where the conductance is
maximum at the $n$th harmonic, and the half-bandwidth ($\Gamma_n$), the
half-width at half-maximum of the conductance peak. This half-bandwidth is
equal to $2D/f_n$, where $D$ is the dissipation factor introduced by the
Chalmers group and utilized in the commonly employed time-domain QCM-D
technique [16]. We refer to $\Gamma_n$ simply as the dissipation, recognizing
that our definition of the dissipation differs from the QCM-D definition by a
factor of $2/f_n$.

If a load is applied to the QCM surface, $f_n$ and $\Gamma_n$ shift relative to
the bare crystal and together make up a complex frequency shift [14, 15]:

$$
\Delta f_n^* = \Delta f_n + i \Delta \Gamma_n
\tag{2}
$$

Note that in our notation, starred quantities are complex, and quantities
subscripted with an $n$ depend on the harmonic at which they are measured. When
the frequency shift is small in comparison to the resonance frequency of
quartz, which is true in nearly all applications of the QCM, $\Delta f_n^*$ is
related to the shift in complex load impedance ($\Delta Z_{nL}^*$) through the
small load approximation [14]:

$$
\Delta f_n^* = \frac{i f_1 \Delta Z_{nL}^*}{\pi Z_q}
\tag{3}
$$

### Sauerbrey Limit

The Sauerbrey equation (Eq. 1) is obtained from Eq. 3, using the following
inertial load impedance, $\Delta Z_{sn}^*$, for $\Delta Z_{nL}^*$:

$$
\Delta Z_{sn}^* = 2 \pi i n f_1 \rho_f d_f
\tag{4}
$$

### Bulk Limit

For thicker films, the complex resonant frequency deviates from the Sauerbrey
relationship and depends on the rheological properties of the material in
contact with the crystal surface. For films that are sufficiently thick (well
beyond the decay length of the acoustic shear wave), the load impedance is
equal to the acoustic impedance of a bulk material:

$$
Z_{n,\text{bulk}}^* = (\rho G_n^*)^{1/2}
\tag{5}
$$

Here $G_n^*$ is the complex shear modulus at $f_n$, expressed here in terms of
its magnitude and phase angle:

$$
G_n^* = |G_n^*| \exp(i \phi_n)
\tag{6}
$$

For a variety of reasons it is more convenient to express the complex shear
modulus in terms of its magnitude and phase. The storage and loss moduli,
$G_n'$ and $G_n''$ respectively, can easily be obtained from the magnitude and
phase:

$$
G_n' = |G_n^*| \cos(\phi_n)
\tag{7a}
$$

$$
G_n'' = |G_n^*| \sin(\phi_n)
\tag{7b}
$$

Use of $Z_{n,\text{bulk}}^*$ for the load impedance gives the following for
$\Delta f_n$ and $\Delta \Gamma_n$ in the bulk limit:

$$
\Delta f_n = -\frac{f_1}{\pi Z_q} (\rho |G_n^*|)^{1/2} \sin(\phi_n/2)
\tag{8}
$$

$$
\Delta \Gamma_n = \frac{f_1}{\pi Z_q} (\rho |G_n^*|)^{1/2} \cos(\phi_n/2)
\tag{9}
$$

These equations can be inverted to give the following for $\rho |G_n^*|$ and
$\phi$:

$$
\rho |G_n^*| = \left( \frac{\pi Z_q |\Delta f_n^*|}{f_1} \right)^2
\tag{10}
$$

$$
\phi = 2 \arctan\left(\frac{\Delta f_n}{\Delta \Gamma_n}\right)
\tag{11}
$$

The decay length is given by the following expression:

$$
\delta_{n,\rho} =
\frac{\rho}{\operatorname{Im}(k^*)}
= \frac{(\rho |G_n^*|)^{1/2}}{2 \pi n f_1 \sin(\phi_n/2)}
\tag{12}
$$

We can use Eq. 8 to eliminate the sin term:

$$
\delta_{n,\rho} = \frac{\rho |G_n^*|}{2 \pi^2 n Z_q \Delta f_n}
\tag{13}
$$

Now we can use Eq. 10 to eliminate $\rho |G_n^*|$:

$$
\delta_{n,\rho} = -\frac{Z_q |\Delta f_n^*|^2}{2 n f_1^2 \Delta f_n}
= -\frac{Z_q (\Delta f_n^2 + \Delta \Gamma_n^2)}{2 n f_1^2 \Delta f_n}
\tag{14}
$$

As a general rule of thumb, the bulk limit applies when the film thickness is
larger than approximately $2 \delta_n$.

### Generalized Single Layer

The complex wave vector, $k_n^*$, for shear wave propagation in the material
can be obtained directly from $Z_n^*$ [17]:

$$
k_n^* = \frac{2 \pi f_n \rho}{Z_n^*}
\tag{15}
$$

The wavelength of the propagating shear wave is then obtained from the real
part of $k^*$ [17]:

$$
\lambda_n = \frac{2 \pi}{\operatorname{Re}(k_n^*)}
;\quad
\lambda_{n,\rho} = \frac{(\rho |G_n^*|)^{1/2}}{f_n \cos(\phi_n/2)}
\tag{16}
$$

Finally, we define a quantity, $D_n^*$, which plays an important role in the
expressions given below:

$$
D_n^* = k_n^* d
= \frac{2 \pi d}{\lambda_n} \left(1 - i \tan(\phi_n/2)\right)
= \frac{2 \pi f_n d \rho}{Z_n^*}
\tag{17}
$$

The impedance of a layer (which we refer to here as layer 1) of mass thickness
$d \rho$ is given by the following expression:

$$
Z_{n,\text{film}}^* = i Z_{n,\text{bulk}}^* \tan(D_n^*)
\tag{18}
$$

Note that we define $\Delta f_{sn}$ as a positive number (see Eq. 1), so in
the case where the Sauerbrey limit applies ($d/\lambda_n \ll 1$), and when
there is sufficient acoustic contrast between the film and the surrounding
liquid ($Z_{n,f}^* \gg Z_{n,\ell}^*$), we have
$\Delta f_n^* = -\Delta f_{sn}$.

Figures in the original PDF:

- Figure 1: Two-layer geometry involving a "film" sandwiched between the
  electrode surface and a "membrane".
- Figure 2: Result from the Lu-Lewis model for the parameters listed in Eqs. 19
  and 21 varying the values of $\phi$ for the film as indicated.

### Bilayer

We can apply Eq. 3, with $\Delta Z_{nL}^* = \Delta Z_{nfm}^*$ to get an
expression for the complex frequency shift. After a bunch of algebra, we get
to the following form of the master equation:

$$
\frac{\Delta f_n^*}{f_{sn}}
= -\frac{\tan(D_{n1}^*)}{D_{n1}^*}
\left[
\frac{1 - (r_{n12}^*)^2}{1 + i r_{n12}^* \tan(D_{n1}^*)}
\right]
\tag{19}
$$

Where $r^*$ is the ratio of the overall membrane impedance to the material
impedance for the film:

$$
r_{n12}^* = \frac{Z_{n2,\text{film}}^*}{Z_{n1,\text{bulk}}^*}
\tag{20}
$$

Written in this way it is more obvious that there is no net response for the
trivial case when a thin film and thick membrane have identical properties
($r_{n12}^* = 1$), and that the single layer master equation is recovered for
experiments done in air ($r_{n12}^* = 0$).

In a liquid medium, the "membrane" actually corresponds to a bulk, infinitely
thick liquid. The impedance ratio is now a ratio of pure material properties:

$$
r_{n12}^* = \left(\frac{\rho_2 G_{n2}^*}{\rho_1 G_{n1}^*}\right)^{1/2}
\tag{21}
$$

## Lu-Lewis Equation

The original PDF includes figures comparing Lu-Lewis and SLA model responses.
Refer to the PDF for the plotted curves and parameter definitions.

## Thicker Films

Things look quite a bit different if we use thicker films. Suppose we use a
5 um glassy polymer film, representative of some of our experiments. Here is
the set of film properties:

$$
(d \rho)_{\text{film}} = 5 \, \text{g/m}^2
$$

$$
(|G_3^*| \rho)_{\text{film}} = 10^9 \, \text{Pa} \cdot \text{g/cm}^3
$$

$$
\phi_{\text{film}} = 1.5 \, \text{deg}
\tag{22}
$$

Figures in the original PDF:

- Figure 3: Comparison of the LL and SLA models for the electrode properties
  given in the PDF and the film properties in Eq. 22.
- Figure 4: Same comparison plotted against linear $n$, focusing on the first
  few harmonics.

## Solution Method and Error Analysis

The numerical solution obtains three quantities (typically $d \rho$,
$|G_3| \rho$ and $\phi$) that give predicted three measured quantities in
agreement with experimental values. These three measured quantities are two
frequency shifts and one dissipation shift. In a $n_1:n_2,n_3$ calculation,
these quantities are $\Delta f_{n1}$, $\Delta f_{n2}$ and $\Delta \Gamma_{n3}$.
MATLAB and Python both return the Jacobian, $J$, at the solution point, which
is given by the following expression:

$$
J =
\begin{bmatrix}
\partial \Delta f_{n1} / \partial(d \rho)
  & \partial \Delta f_{n1} / \partial(|G_3| \rho)
  & \partial \Delta f_{n1} / \partial \phi \\
\partial \Delta f_{n2} / \partial(d \rho)
  & \partial \Delta f_{n2} / \partial(|G_3| \rho)
  & \partial \Delta f_{n2} / \partial \phi \\
\partial \Delta \Gamma_{n3} / \partial(d \rho)
  & \partial \Delta \Gamma_{n3} / \partial(|G_3| \rho)
  & \partial \Delta \Gamma_{n3} / \partial \phi
\end{bmatrix}
\tag{23}
$$

Once we have the matrix, we invert it to get $J^{-1}$, the Jacobian of the
inverse function, which is what we need:

$$
J^{-1} =
\begin{bmatrix}
\partial(d \rho) / \partial \Delta f_{n1}
  & \partial(d \rho) / \partial \Delta f_{n2}
  & \partial(d \rho) / \partial \Delta \Gamma_{n3} \\
\partial(|G_3| \rho) / \partial \Delta f_{n1}
  & \partial(|G_3| \rho) / \partial \Delta f_{n2}
  & \partial(|G_3| \rho) / \partial \Delta \Gamma_{n3} \\
\partial \phi / \partial \Delta f_{n1}
  & \partial \phi / \partial \Delta f_{n2}
  & \partial \phi / \partial \Delta \Gamma_{n3}
\end{bmatrix}
\tag{24}
$$

The uncertainties in the experimental quantities are $\Delta f_{n1}^{\text{err}}$,
$\Delta f_{n2}^{\text{err}}$ and $\Delta \Gamma_{n3}^{\text{err}}$. We represent
these as a three component error vector:

$$
\text{err} =
\begin{bmatrix}
\Delta f_{n1}^{\text{err}} \\
\Delta f_{n2}^{\text{err}} \\
\Delta \Gamma_{n3}^{\text{err}}
\end{bmatrix}
\tag{25}
$$

In our linearized analysis we just multiply the uncertainties by the
appropriate partial derivatives, and sum the three different uncertainties in
quadrature to get the associated experimental uncertainty in our extracted
property:

$$
(d \rho)_{\text{err}} =
\left(
(J^{-1}_{11} \, \text{err}(1))^2
+ (J^{-1}_{12} \, \text{err}(2))^2
+ (J^{-1}_{13} \, \text{err}(3))^2
\right)^{1/2}
$$

$$
(|G_3| \rho)_{\text{err}} =
\left(
(J^{-1}_{21} \, \text{err}(1))^2
+ (J^{-1}_{22} \, \text{err}(2))^2
+ (J^{-1}_{23} \, \text{err}(3))^2
\right)^{1/2}
$$

$$
\phi_{\text{err}} =
\left(
(J^{-1}_{31} \, \text{err}(1))^2
+ (J^{-1}_{32} \, \text{err}(2))^2
+ (J^{-1}_{33} \, \text{err}(3))^2
\right)^{1/2}
\tag{26}
$$

This same analysis can be applied to more complicated models as well.

## Bibliography

[1] G. Sauerbrey, Verwendung von Schwingquarzen zur Wagung dunner Schichten
und zur Mikrowagung, Zeitschrift fur Physik A Hadrons and Nuclei 155 (2)
(1959) 206-222. doi:10.1007/BF01337937.

[2] K. A. Marx, Quartz Crystal Microbalance: A Useful Tool for Studying Thin
Polymer Films and Complex Biomolecular Systems at the Solution-Surface
Interface, Biomacromolecules 4 (5) (2003) 1099-1120. doi:10.1021/bm020116i.

[3] M. A. Cooper, V. T. Singleton, A survey of the 2001 to 2005 quartz crystal
microbalance biosensor literature: applications of acoustic physics to the
analysis of biomolecular interactions, Journal of Molecular Recognition 20 (3)
(2007) 154-184. doi:10.1002/jmr.826.

[4] B. Becker, M. A. Cooper, A survey of the 2006-2009 quartz crystal
microbalance biosensor literature, Journal of Molecular Recognition 24 (5)
(2011) 754-787. doi:10.1002/jmr.1117.

[5] A. Tiraferri, P. Maroni, D. Caro Rodriguez, M. Borkovec, Mechanism of
Chitosan Adsorption on Silica from Aqueous Solutions, Langmuir 30 (17) (2014)
4980-4988. doi:10.1021/la500680g.

[6] P. Yi, K. L. Chen, Release Kinetics of Multiwalled Carbon Nanotubes
Deposited on Silica Surfaces: Quartz Crystal Microbalance with Dissipation
(QCM-D) Measurements and Modeling, Environmental Science and Technology 48 (8)
(2014) 4406-4413. doi:10.1021/es405471u.

[7] J. Song, W. E. Krause, O. J. Rojas, Adsorption of polyalkyl glycol ethers
and triblock nonionic polymers on PET, Journal of Colloid and Interface Science
420 (2014) 174-181. doi:10.1016/j.jcis.2014.01.012.

[8] I. E. Salama, B. P. Binks, P. D. Fletcher, D. I. Horsup, Adsorption of
benzyldimethyldodecylammonium chloride onto stainless steel using the quartz
crystal microbalance and the depletion methods: an optimisation study,
Colloids and Surfaces A: Physicochemical and Engineering Aspects 447 (2014)
155-165. doi:10.1016/j.colsurfa.2014.01.034.

[9] A. Rudrake, K. Karan, J. H. Horton, A combined QCM and XPS investigation of
asphaltene adsorption on metal surfaces, Journal of Colloid and Interface
Science 332 (1) (2009) 22-31.

[10] Z. Liu, H. Choi, P. Gatenholm, A. R. Esker, Quartz Crystal Microbalance
with Dissipation Monitoring and Surface Plasmon Resonance Studies of
Carboxymethyl Cellulose Adsorption onto Regenerated Cellulose Surfaces,
Langmuir 27 (14) (2011) 8718-8728. doi:10.1021/la200628a.

[11] J. Wegener, A. Janshoff, C. Steinem, The quartz crystal microbalance as a
novel means to study cell-substrate interactions in situ, Cell Biochemistry
and Biophysics 34 (1) (2001) 121-151.

[12] D. A. Buttry, M. D. Ward, Measurement of Interfacial Processes at
Electrode Surfaces with the Electrochemical Quartz Crystal Microbalance,
Chem. Rev. 92 (1992) 1355-1379.

[13] A. Hillman, The Electrochemical Quartz Crystal Microbalance, in:
A. J. Bard, M. Stratmann, P. R. Unwin (Eds.), Encyclopedia of Electrochemistry,
Volume 3, Instrumentation and Electroanalytical Chemistry, Vol. 3, Wiley-VCH,
Weinheim, 2003, pp. 230-289.

[14] D. Johannsmann, Viscoelastic, mechanical, and dielectric measurements on
complex samples with the quartz crystal microbalance, Physical Chemistry
Chemical Physics 10 (31) (2008) 4516-4534.

[15] G. C. DeNolf, L. Haack, J. Holubka, A. Straccia, K. Blohowiak,
K. R. Shull, High Frequency Rheometry of Viscoelastic Coatings with the Quartz
Crystal Microbalance, Langmuir 27 (16) (2011) 9873-9879.
doi:10.1021/la200646h.

[16] M. V. Voinova, M. Rodahl, M. Jonson, B. Kasemo, Viscoelastic Acoustic
Response of Layered Polymer Films at Fluid-Solid Interfaces: Continuum
Mechanics Approach, Physica Scripta 59 (5) (1999) 391-396.
doi:10.1238/Physica.Regular.059a00391.

[17] F. Simonetti, P. Cawley, On the nature of shear horizontal wave
propagation in elastic plates coated with viscoelastic materials, Proceedings
of the Royal Society A: Mathematical, Physical and Engineering Sciences
460 (2048) (2004) 2197-2221. doi:10.1098/rspa.2004.1284.
