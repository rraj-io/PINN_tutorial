# PINNs from Scratch — Tutorial Series

**When Solvers Learn · HalfQubiX**

Companion notebooks for the *PINNs from Scratch* tutorial series: a practitioner's walkthrough of Physics-Informed Neural Networks, from autograd fundamentals to nonlinear PDEs — one small, debuggable problem at a time.

Each notebook is paired with a full write-up on Substack: [rajrohit403.substack.com](https://rajrohit403.substack.com)

## What is a PINN?

A Physics-Informed Neural Network is a neural network trained to satisfy a differential equation rather than a labelled dataset. Instead of minimizing prediction vs. ground truth, it minimizes:

$$\mathcal{L} = \underbrace{\|\text{PDE residual}\|^2}_{\text{physics}} + \underbrace{\|\text{BC/IC error}\|^2}_{\text{boundary conditions}}$$

No mesh, no labelled simulation data — derivatives come exactly from automatic differentiation, and the network learns a function that satisfies the governing equation everywhere in the domain.

## The tutorials

Work through them in order — each one adds exactly one layer of difficulty on top of the last, and the code structure barely changes between them.

| # | Notebook | Problem | What's new |
|---|---|---|---|
| T0 | `PINN_Tutorial_00_Background_and_Setup.ipynb` | — | Autograd from first principles, why tanh (not ReLU), collocation sampling strategies, environment check |
| T1 | `PINN_Tutorial_01_1D_First_Order_ODE.ipynb` | $du/dx = \cos(\pi x)$, $u(0)=0$ | The simplest possible PINN: one `autograd.grad` call, residual + IC loss, training loop, validation against a closed form |
| T2 | `PINN_Tutorial_02_1D_Linear_ODE.ipynb` | 1D Poisson: $-u'' = \sin(\pi x)$, $u(0)=u(1)=0$ | Second derivatives (chained `autograd.grad`), two boundary conditions, `create_graph` vs `retain_graph`, collocation-count study |
| T3 | `PINN_Tutorial_03_1D_Burgers.ipynb` | Steady Burgers: $u\,u_x - \nu u_{xx} = 0$, $u(\pm 1) = \pm 1$ | First **nonlinear** residual, SciPy `solve_bvp` reference (no closed form), Xavier init, two-phase **Adam → L-BFGS** training, viscosity study |
| T4 | `PINN_Tutorial_04_2D_Poisson.ipynb` | 2D Poisson: $-\nabla^2 u = f$ on the unit square | Two spatial inputs, `meshgrid` collocation, four boundary edges, 2D error heatmaps |
| T5 | *(planned)* | 2D Navier–Stokes (lid-driven cavity) | Vector-valued output, pressure coupling, incompressibility constraint |

## How to follow the tutorials

Each tutorial exists in two forms that are meant to be used together:

1. **The Substack post** explains the math, every design decision, and how to read the training logs — including the deliberate-break experiments and what the failure modes look like.
2. **The notebook** (this repo) is the runnable version: interleaved markdown and code, in run order, with the plots and real training logs included.

The workflow that works best:

- Read the post (or the notebook markdown) for a section, **run the corresponding cells**, and check your output against the recorded output before moving on. Every tutorial validates against a known reference solution, so you always know whether your run worked.
- Do the **exercises** at the end of each tutorial. Each one breaks something on purpose — removing the IC term, dropping `create_graph=True`, cutting the training budget — so you see the failure mode with your own eyes instead of just reading about it.
- If your run doesn't converge, compare against the committed notebook: it's a known-working version with matching seeds (`torch.manual_seed(42)`), so differences point at what changed.

Every tutorial follows the same six-step recipe: **write the PDE → sample collocation points → build a network → define the loss → minimize → validate.** T0 introduces the recipe; T1–T4 apply it to progressively harder problems.

## Coming next: T5 — 2D Navier–Stokes (lid-driven cavity)

The series capstone solves the steady incompressible Navier–Stokes equations in the **lid-driven cavity** — the standard benchmark of CFD: a square box of fluid, three stationary walls, and a top lid sliding at constant velocity, driving a recirculating vortex.

$$(\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0$$

Everything the earlier tutorials built shows up at once, plus the genuinely new challenges of a real flow problem:

- **Vector-valued output**: the network predicts three fields — $u(x,y)$, $v(x,y)$, $p(x,y)$ — instead of a single scalar
- **Coupled equations**: two momentum residuals plus the continuity equation, all enforced simultaneously in one loss
- **Incompressibility as a constraint**: $\nabla \cdot \mathbf{u} = 0$ becomes its own loss term (and where the divergence residual refuses to vanish is a diagnostic in itself)
- **Pressure without pressure BCs**: pressure appears only through its gradient, so it's determined up to a constant — and handling that is half the battle
- **The Burgers convection term, now for real**: $(\mathbf{u}\cdot\nabla)\mathbf{u}$ is the 2D version of the $u\,u_x$ nonlinearity from T3, and the Reynolds number plays exactly the role the viscosity study previewed
- **Validation against classical CFD**: the benchmark reference for cavity flow (Ghia et al. centerline velocity profiles), the same way T3 validated against `solve_bvp`

Subscribe on [Substack](https://rajrohit403.substack.com/) to catch it the day it lands.

## Setup

Everything runs with four packages. **No GPU needed** — every notebook trains in minutes on a laptop CPU (T3's full two-phase run is the slowest at ~5 minutes).

```bash
git clone https://github.com/rraj-io/pinn-tutorial.git
cd pinn-tutorial

python3 -m venv pinn-env
source pinn-env/bin/activate   # Windows: pinn-env\Scripts\activate

pip install torch numpy matplotlib scipy
```

Tested versions: `torch` 2.x (CPU build is enough), `numpy` 1.x/2.x, `matplotlib` 3.x, `scipy` 1.x (used from T3 onward for the reference BVP solver).

Then launch Jupyter and open the tutorial you're on:

```bash
pip install jupyter
jupyter notebook
```

Verify your environment with T0's final cell — if the autograd demo prints derivatives matching the analytical values, you're ready for Tutorial 1.

## What "working" looks like

Each notebook ends with validation against a reference solution and reports L2 / max error. Representative results from the committed runs (seed 42):

| Tutorial | Reference | L2 error |
|---|---|---|
| T1 — first-order ODE | Analytical $\sin(\pi x)/\pi$ | ~2.5e-03 |
| T2 — 1D Poisson | Analytical $\sin(\pi x)/\pi^2$ | ~7.3e-05 |
| T3 — steady Burgers ($\nu=0.1$) | SciPy `solve_bvp` | ~1.9e-04 |

Your numbers won't match to the last digit across PyTorch versions and platforms, but they should land in the same order of magnitude. If they're wildly off, the failure-mode tables in each tutorial are the place to start.

## Ebook

The series is also available as a single self-contained PDF — ***Physics-Informed Neural Networks, from Ground Zero*** (T0–T3, ~51 pages), with every script reproduced in full, in run order, plus the derivations and design decisions behind each one.

**Get it on Gumroad:** [rajrohit403.gumroad.com/l/pinn_tutorial](https://rajrohit403.gumroad.com/l/pinn_tutorial)

## Support the series

- **Subscribe** on Substack to catch new tutorials the day they land: [rajrohit403.substack.com](https://rajrohit403.substack.com/)
- **Buy the ebook** on Gumroad to support the notebooks and the writing that goes with them: [rajrohit403.gumroad.com/l/pinn_tutorial](https://rajrohit403.gumroad.com/l/pinn_tutorial)
- **Open an issue** if a notebook doesn't run for you — include your package versions and the first cell that misbehaves.

---

*Rohit Raj · HalfQubiX — classical numerical methods meet scientific machine learning.*
