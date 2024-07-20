from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
import sympy.vector as spv
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
import pygmsh
import meshio
from typing import Callable
from jaxtyping import Float, Array
from functools import partial
from copy import copy

from jax import vmap

from fipy import TransientTerm, DiffusionTerm, ConvectionTerm, SphericalGrid1D, CellVariable, LinearLUSolver, Variable

# Enable 64-bit precision
from jax import config
config.update("jax_enable_x64", True)

class PhysicalSystem(ABC):
    """An advection-diffusion system in a 1D sphere."""

    def __init__(
        self,
        R: float,
        n_grid: float,
        exact_solution: Callable = None,
        **kwargs
    ):
        self.R = R
        self.n_grid = n_grid
        self._physical_parameters = kwargs
        self.exact_solution = exact_solution
        if exact_solution is not None:
            self.create_source_for_verification()
        self.sol = None
    
    def create_source_for_verification(self):
        """Find the source term that results in the provided solution.
        
        Uses symbolic differentiation to find the source term and saves it. Useful for verification."""
        self._check_exact_solution_is_provided()

        # Create symbolic variables
        t = sp.Symbol('t')
        args_sym = {k: sp.Symbol(k) for k in self.args.keys()}
        S = spv.CoordSys3D('S', transformation='spherical')

        c = self.exact_solution(t, S.r, args_sym)  # The solution
        D = self.diff_coef(S.r, args_sym)  # Diffusion coefficient
        v = self.conv_coef(S.r, args_sym)*S.i  # Velocity
        rhs_no_source = ( spv.divergence( D*spv.gradient(c) + v*c ) ).simplify()
        time_derivative = sp.diff(c, t)
        source = ( time_derivative - rhs_no_source ).simplify()
        
        args_sym_lst = list(args_sym.values())
        args_vals_lst = list(self.args.values())
        f = sp.lambdify((*args_sym_lst, t, S.r), source.factor(S.r), 'numpy')
        self.source_for_verification = lambda t, r: f(*args_vals_lst, t, r)
    
    @property
    def args(self):
        return dict(R=self.R, **self._physical_parameters)

    @abstractmethod
    def source(self, t, r, symbolic_args=None):
        """Source term for the right-hand side of the PDE."""
        pass

    @abstractmethod
    def diff_coef(self, r, symbolic_args=None):
        """Diffusion coefficient."""
        pass

    @abstractmethod
    def conv_coef(self, r, symbolic_args=None):
        """Convection coefficient (i.e., the velocity)."""
        pass
    
    def _check_exact_solution_is_provided(self):
        if self.exact_solution is None:
            raise ValueError('Exact solution not provided.')

    def _check_numerical_solution_is_computed(self):
        if self.sol is None:
            raise ValueError('Solution not computed yet.')

    def solve(self, final_time, dt):
        """Solve the PDE using FiPy."""

        # Create mesh
        mesh = SphericalGrid1D(nr=self.n_grid, Lr=self.R)
        cell_centers = mesh.cellCenters.value[0]
        face_centers = mesh.faceCenters.value[0]

        # Set up the physical system
        D = CellVariable(name='D(r)', mesh=mesh, value=vmap(partial(self.diff_coef, args=self.args))(cell_centers))
        v = CellVariable(name='v(r)', mesh=mesh, value=vmap(partial(self.conv_coef, args=self.args))(cell_centers)[None, :])
        r = CellVariable(name='r', mesh=mesh, value=cell_centers)
        t = Variable(0.)
        u = CellVariable(mesh=mesh, name='concentration', value=0.)
        u.faceGrad.constrain([0.], mesh.facesRight) # Fixed flux boundary condition
        u.faceGrad.constrain([0.], mesh.facesLeft)
        eq = TransientTerm() == DiffusionTerm(coeff=D) + ConvectionTerm(coeff=v) + self.source(t, r)
        
        # Pick a solver
        solver = LinearLUSolver()

        # Solve
        t_lst = [copy(t.value)]
        u_lst = [copy(u.faceValue.value)]
        while t.value <= final_time:
            t.setValue(t.value + dt)
            eq.solve(var=u, dt=dt, solver=solver)
            t_lst.append(copy(t.value))
            u_lst.append(copy(u.faceValue.value))
        
        # Save result
        self.sol = dict(
            t=np.hstack(t_lst),
            r=face_centers,
            u=np.stack(u_lst, axis=1)
        )
        self._interpolate_solution = RegularGridInterpolator((self.sol['t'], self.sol['r']), self.sol['u'].T)
    
    def evaluate_solution(
        self, 
        t: Float[Array, "#N"], 
        r: Float[Array, "#N"]
    ) -> Float[Array, "N"]:
        """Interpolate the solution against the grid-values."""
        self._check_numerical_solution_is_computed()
        return self._interpolate_solution(np.stack(np.broadcast_arrays(t, r), axis=-1))
        
    def plot_solution_evolution(self, plot_times):
        """Create a plot of the solution at different times."""
        self._check_numerical_solution_is_computed()

        is_label_set = False
        fig, ax = plt.subplots(figsize=(4, 4))
        
        for t in plot_times:
            u = self.evaluate_solution(t, self.sol['r'])
            if not is_label_set:
                is_label_set = True
                label_numerical = 'Numerical'
                label_exact = 'Exact'
            else:
                label_numerical = None
                label_exact = None
            ax.plot(self.sol['r'], u, linewidth=3, color='tab:blue', label=label_numerical)
            if self.exact_solution is not None:
                u_exact = vmap(partial(self.exact_solution, t, args=self.args))(self.sol['r'])
                ax.plot(self.sol['r'], u_exact, linewidth=2, linestyle='--', color='tab:orange', label=label_exact)
            ax.annotate(f"$t={t:.1f}$", (self.R, u[-1] + 0.5), xycoords='data')
        
        ax.set_xlabel('$r$')
        ax.set_ylabel('$u$')
        ax.legend()
        sns.despine(trim=True)

    def plot_solution_vs_ground_truth(self):
        """Plot the solution against the ground truth."""
        self._check_exact_solution_is_provided()
        self._check_numerical_solution_is_computed()

        # Plot the result
        fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
        ax[0].set_title('Numerical solution')
        p1 = ax[0].contourf(self.sol['t'], self.sol['r'], self.sol['u'], 20)
        vmin, vmax = copy(p1.get_clim())

        # Plot exact solution
        u_exact = vmap(vmap(partial(self.exact_solution, args=self.args), (0, None)), (None, 0))(self.sol['t'], self.sol['r'])
        ax[1].set_title('Exact solution')
        p2 = ax[1].contourf(self.sol['t'], self.sol['r'], u_exact, 20, vmin=vmin, vmax=vmax)

        # Plot the difference
        diff = (self.sol['u'] - u_exact)
        ax[2].set_title('Difference')
        p3 = ax[2].contourf(self.sol['t'], self.sol['r'], diff, 20)

        ax[0].set_xlabel('Time')

        # add colorbar to each plot
        fig.colorbar(p1, ax=ax[0])
        fig.colorbar(p1, ax=ax[1])
        fig.colorbar(p3, ax=ax[2])
        
        return fig, ax
    
    def create_solution_video(self, filename):
        """Create a video of the solution."""

        with pygmsh.occ.Geometry() as geom:
            geom.add_ball([0.0, 0.0, 0.0], self.R, mesh_size=0.2)
            _mesh = geom.generate_mesh()
        meshio.write("mesh.vtk", _mesh)
        
        mesh = pv.read("mesh.vtk")
        mesh = mesh.clip_box([0, self.R, 0, self.R, 0, self.R])
        points = np.sqrt(np.sum(mesh.points**2, axis=1))
        mesh.add_field_data(self.evaluate_solution(0.0, points), 'u')
        t_plt = np.linspace(0, self.sol['t'][-1], 100)
        cmap = plt.get_cmap('viridis', 256)

        p = pv.Plotter()
        p.add_mesh(mesh, scalars=mesh['u'], cmap=cmap, clim=[0, np.ceil(self.sol['u'].max())])
        p.open_movie(filename)
        for t in t_plt:
            text = p.add_text(f"t={t:.1f}", font_size=10)
            vals = self.evaluate_solution(t, points)
            p.mesh['u'][:] = vals
            p.write_frame()
            p.remove_actor(text)
        p.close()