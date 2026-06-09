#!/usr/bin/env python3

import argparse
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from polaris.config import PolarisConfigParser
from polaris.mesh.planar import compute_planar_hex_nx_ny
from polaris.ocean.vertical import init_vertical_coord

# Physics parameters (no-slip variant, ν₂=400 gives δ_m ≈ 34 km)
NU_2  = 400.0    # Laplacian viscosity [m² s⁻¹]
F_0   = 1e-4     # reference Coriolis [s⁻¹]
BETA  = 1e-11    # beta-plane gradient [s⁻¹ m⁻¹]
TAU_0 = 0.1      # wind stress amplitude [N m⁻²]
H     = 5000.0   # resting depth [m]
LX    = 1200.0   # domain width  [km]
LY    = 1200.0   # domain height [km]


def exact_ssh_solution(
        ds, tau_0=TAU_0, rho=1.e3, g=9.81, nu_2=NU_2, beta=BETA, f_0=F_0):
    """Exact Munk-layer SSH for the linearised barotropic gyre."""
    get_extent = lambda x: float(x.max() - x.min())

    L_x = get_extent(ds.xCell)
    L_y = get_extent(ds.yCell)
    layerThickness = ds.restingThickness.squeeze()

    pi     = np.pi
    sqrt3  = np.sqrt(3)
    delta_m = (nu_2 / beta) ** (1. / 3.)
    gamma   = (sqrt3 * ds.xCell) / (2. * delta_m)

    ssh = (tau_0 / (rho * g * layerThickness)) * (ds.fCell / beta) * \
          (1. - ds.xCell / L_x) * pi * np.sin(pi * ds.yCell / L_y) * \
          (1. - np.exp(-1. * ds.xCell / (2. * delta_m)) *
           (np.cos(gamma) + (1. / sqrt3) * np.sin(gamma)))

    return ssh


def create_initial_state(resolution_km, output_dir):
    """Create initial_state.nc and forcing.nc for one resolution.

    Parameters
    ----------
    resolution_km : float
        Grid cell spacing in kilometres.
    output_dir : str
        Directory where output files are written.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Munk-layer width check: δ_m = (2π/√3)(ν₂/β)^(1/3) must span ≥ 3 cells
    delta_m = (2 * np.pi / np.sqrt(3)) * (NU_2 / BETA) ** (1. / 3.)
    if delta_m < 3. * resolution_km * 1e3:
        raise ValueError(
            f"Resolution {resolution_km} km too coarse: Munk layer "
            f"({delta_m/1e3:.1f} km) < 3 cells ({3*resolution_km:.1f} km)"
        )

    dc = resolution_km * 1e3   # cell spacing in metres
    nx, ny = compute_planar_hex_nx_ny(LX, LY, resolution_km)
    ds_mesh = make_planar_hex_mesh(
        nx=nx, ny=ny, dc=dc, nonperiodic_x=True, nonperiodic_y=True
    )
    ds_mesh = cull(ds_mesh)
    ds_mesh = convert(ds_mesh)

    write_netcdf(ds_mesh, os.path.join(output_dir, 'culled_mesh.nc'))

    # Load vertical grid config from config.cfg (vertical_grid section only)
    config = PolarisConfigParser()
    config.add_from_file("config.cfg")

    # Build IC dataset on top of the mesh
    ds = ds_mesh.copy()
    ds["ssh"]         = xr.zeros_like(ds.xCell)
    ds["bottomDepth"] = H * xr.ones_like(ds.xCell)

    init_vertical_coord(config, ds)

    # Beta-plane Coriolis on cells, edges, and vertices
    for loc in ["Cell", "Edge", "Vertex"]:
        ds[f"f{loc}"] = F_0 + BETA * ds[f"y{loc}"]

    # Zero initial velocity
    ds["normalVelocity"] = xr.zeros_like(ds.xEdge).expand_dims(
        ["Time", "nVertLevels"], axis=[0, -1]
    )

    # Bug fix: write ds (with ICs) not ds_mesh (mesh-only) to initial_state.nc
    # Also exclude wind-stress fields — those go in forcing.nc
    write_netcdf(ds, os.path.join(output_dir, 'initial_state.nc'))

    # Wind stress on cells, y-varying: τ_x = −τ₀ cos(π y / L_y)
    # Shape (nCells, 1) to match ForcingVars.jl expectation of (nCells, nVertLevels)
    Ly_m = LY * 1e3
    wind_stress_zonal = (
        -TAU_0 * np.cos(np.pi * ds.yCell.values / Ly_m)
    ).reshape(-1, 1)

    ds_forcing = xr.Dataset(
        {
            "windStressZonal": (["nCells", "nVertLevels"], wind_stress_zonal),
            "windStressMeridional": (
                ["nCells", "nVertLevels"],
                np.zeros((ds.dims["nCells"], 1))
            ),
        }
    )
    write_netcdf(ds_forcing, os.path.join(output_dir, 'forcing.nc'))

    print(f"Done: {output_dir}/initial_state.nc and forcing.nc")
    print(f"  resolution={resolution_km} km, Munk layer={delta_m/1e3:.1f} km "
          f"({delta_m/(resolution_km*1e3):.1f} cells wide)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate barotropic gyre ICs for Moka.jl"
    )
    parser.add_argument("--res", type=float, required=True,
                        help="Grid resolution in km (e.g. 40, 20, 10)")
    parser.add_argument("--dir", type=str, required=True,
                        help="Output directory (e.g. 40km)")
    args = parser.parse_args()

    create_initial_state(args.res, args.dir)
