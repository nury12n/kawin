import numpy as np

from pycalphad import variables as v

from kawin.plot_utils import _get_axis, _adjust_kwargs
from kawin.thermo.mobility import u_to_x_frac, expand_u_frac, interstitials
from kawin.diffusion.diffusion import DiffusionState, DiffusionModel, LookupTable
from kawin.diffusion.homogenization import compute_mobility
from kawin.diffusion.mesh.fvm import FiniteVolume1D, Cartesian2D

def _get_1d_mesh(model: DiffusionModel):
    mesh: FiniteVolume1D = model.mesh
    if not isinstance(mesh, FiniteVolume1D):
        raise ValueError('Diffusion mesh must be Cartesian1D, Cylindrical1D, Spherical1D or a subclass of FiniteVolume1D')
    return mesh

def _offset(x, z_scale, z_offset):
    return (x+z_offset)/z_scale

def _set_1d_xlim(ax, mesh: FiniteVolume1D, z_scale, z_offset):
    ax.set_xlim([_offset(mesh.edges[0], z_scale, z_offset), _offset(mesh.edges[-1], z_scale, z_offset)])
    ax.set_xlabel(f'Distance*{z_scale:.0e} (m)')

def plot_1d(state: DiffusionState, model: DiffusionModel, elements=None, z_scale=1, z_offset=0, ax=None, plot_ufrac=False, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    z_scale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. z_scale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so z_scale represents the desired unit/meters
        Defaults to 0
    z_offset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1d_mesh(model)
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    # Convert y to full composition space and convert to composition if plotting
    y_full = expand_u_frac(state.u, model.all_elements, model.ref_element, interstitials)
    if not plot_ufrac:
        y_full = u_to_x_frac(y_full, model.all_elements, interstitials)
    for e in elements:
        y = y_full[:,model.all_elements.index(e)]
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot(_offset(mesh.z, z_scale, z_offset), y, *args, **plot_kwargs)

    _set_1d_xlim(ax, mesh, z_scale, z_offset)
    if len(elements) > 1:
        ax.legend()
    ax.set_ylim([0,1])
    ax.set_ylabel(f'Composition (at.)')
    return ax

def plot_1d_two_axis(state: DiffusionState, model: DiffusionModel, left_elements, right_elements, z_scale=1, z_offset=0, axl=None, axr=None, plot_ufrac=False, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh on left and right axes

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    left_elements: list[str]
        List of elements to plot on left axis (can include the dependent/reference element)
    right_elements: list[str]
        List of elements to plot on right axis (can include the dependent/reference element)
    z_scale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. z_scale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so z_scale represents the desired unit/meters
        Defaults to 0
    z_offset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    axl: matplotlib axis (optional)
        Left axis
        Will be created if None
    axr: matplotlib axis (optional)
        Right axis
        Will be created if None

    Returns
    -------
    matplotlib Axis for left axis
    matplotlib Axis for right axis
    '''
    axl = _get_axis(axl)
    if axr is None:
        axr = axl.twinx()
    mesh = _get_1d_mesh(model)
    if isinstance(left_elements, str):
        left_elements = [left_elements]
    if isinstance(right_elements, str):
        right_elements = [right_elements]

    # Convert y to full composition space and convert to composition if plotting
    y_full = expand_u_frac(state.u, model.all_elements, model.ref_element, interstitials)
    if not plot_ufrac:
        y_full = u_to_x_frac(y_full, model.all_elements, interstitials)

    # Plotting is the same for each axis, so we can just loop across the two
    i = 0
    for ax, elements in zip([axl, axr], [left_elements, right_elements]):
        for e in elements:
            y = y_full[:,model.all_elements.index(e)]
            plot_kwargs = _adjust_kwargs(e, {'label': e, 'color': f'C{i}'}, kwargs)
            ax.plot(_offset(mesh.z, z_scale, z_offset), y, *args, **plot_kwargs)
            i += 1

        # If the list of elements is small, we can add them to the y label
        if len(elements) <= 3:
            element_list = ', '.join(elements)
            element_label = f'[{element_list}] '
        else:
            element_label = ''
        ax.set_ylabel(f'Composition {element_label}(at.)')
        ax.set_ylim([0,1])

    _set_1d_xlim(ax, mesh, z_scale, z_offset)
    left_lines, left_labels = axl.get_legend_handles_labels()
    right_lines, right_labels = axr.get_legend_handles_labels()
    axl.legend(left_lines+right_lines, left_labels+right_labels, framealpha=1)
    return axl, axr

def plot_1d_phases(state: DiffusionState, model: DiffusionModel, phases=None, z_scale=1, z_offset=0, ax=None, *args, **kwargs):
    '''
    Plots phase fractions over z

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    phases : list[str]
        Plots phases. If None, all phases in model are plotted
    z_scale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. z_scale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so z_scale represents the desired unit/meters
        Defaults to 0
    z_offset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax : matplotlib Axes object
        Axis to plot on

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1d_mesh(model)

    # Compute phase fraction
    x_full = expand_u_frac(state.u, model.all_elements, model.ref_element, interstitials)
    x_full = u_to_x_frac(x_full, model.all_elements, interstitials)[:,1:]
    y_diff, z_diff = model.mesh.get_diffusivity_coordinates(x_full)
    conditions = {v.X(e): y_diff[:,model.elements.index(e)] for e in model.elements}
    conditions[v.T] = model.temperature(state.time, z_diff)
    conditions[v.P] = kwargs.get(v.P, 101325)
    conditions[v.GE] = kwargs.get(v.GE, 0)
    mob_data = LookupTable(compute_mobility)(model.therm, conditions)
    phases = model.phases if phases is None else phases
    if isinstance(phases, str):
        phases = [phases]

    # plot phase fraction
    for p in phases:
        pf = []
        for md in mob_data:
            pf.append(np.sum(md.phase_fractions[md.phases==p]))
        plot_kwargs = _adjust_kwargs(p, {'label': p}, kwargs)
        ax.plot(_offset(mesh.z, z_scale, z_offset), pf, *args, **plot_kwargs)

    _set_1d_xlim(ax, mesh, z_scale, z_offset)
    if len(phases) > 1:
        ax.legend()
    ax.set_ylim([0,1])
    ax.set_ylabel(f'Phase Fraction')
    return ax

def plot_1d_fluxes(state: DiffusionState, model: DiffusionModel, elements=None, z_scale=1, z_offset=0, ax=None, *args, **kwargs):
    '''
    Plots flux of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    z_scale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. z_scale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so z_scale represents the desired unit/meters
        Defaults to 0
    z_offset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1d_mesh(model)
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    fluxes = model.mesh.compute_fluxes(model.compute_pairs(state))

    # Sum of fluxes for substitutional elements = 0
    fluxes_sub = [fluxes[:,i] for i,e in enumerate(model.elements) if e not in interstitials]
    # If all dependent elements are interstitial, then flux of dependent element (substitutional) is 0
    if len(fluxes_sub) == 0:
        fluxes_sum = np.zeros((len(fluxes), 1))
    else:
        fluxes_sum = 0-np.sum(fluxes_sub,axis=0)[:,np.newaxis]
    fluxes_full = np.concatenate((fluxes_sum, fluxes), axis=1)
    for e in elements:
        y = fluxes_full[:,model.all_elements.index(e)]
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot(_offset(mesh.edges, z_scale, z_offset), y, *args, **plot_kwargs)

    _set_1d_xlim(ax, mesh, z_scale, z_offset)
    if len(elements) > 1:
        ax.legend()
    ax.set_ylabel(f'$J/V_m$ ($m/s$)')
    return ax

def plot_2d(state: DiffusionState, model: DiffusionModel, element, z_scale=1, ax=None, plot_ufrac=False, *args, **kwargs):
    '''
    Plots a composition profile on a 2D mesh

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    element: str
        Element to plot
    z_scale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')

    # make sure zScale has 2 elements (for x and y axis)
    z_scale = np.atleast_1d(z_scale)
    if z_scale.shape[0] == 1:
        z_scale = z_scale[0]*np.ones(2)

    y_full = expand_u_frac(state.u, model.all_elements, model.ref_element, interstitials)
    if not plot_ufrac:
        y_full = u_to_x_frac(y_full, model.all_elements, interstitials)
    # Reshape composition to mesh shape (Cartesian2D.unflattenResponse will give (Nx, Ny, 1))
    y = np.reshape(y_full[:,model.all_elements.index(element)], (mesh.Nx, mesh.Ny))
    #y = mesh.unflattenResponse(y_full[:,model.all_elements.index(element)], 1)[...,0]

    # plot element
    # TODO: there should be a way to do contour and contourf (maybe separate plot functions?)
    plot_kwargs = _adjust_kwargs(element, {'vmin': 0, 'vmax': 1}, kwargs)
    cm = ax.pcolormesh(mesh.z_mesh[...,0]/z_scale[0], mesh.z_mesh[...,1]/z_scale[1], y, *args, **plot_kwargs)
    ax.set_title(element)
    ax.set_xlabel(f'Distance x*{z_scale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{z_scale[1]:.0e} (m)')
    return ax, cm

def plot_2d_phases(state: DiffusionState, model: DiffusionModel, phase, z_scale=1, ax=None, *args, **kwargs):
    '''
    Plots a composition profile on a 2D mesh

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    phase: str
        phase to plot
    z_scale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    # make sure zScale has 2 elements (for x and y axis)
    z_scale = np.atleast_1d(z_scale)
    if z_scale.shape[0] == 1:
        z_scale = z_scale[0]*np.ones(2)

    x_full = expand_u_frac(state.u, model.all_elements, model.ref_element, interstitials)
    x_full = u_to_x_frac(x_full, model.all_elements, interstitials)[:,1:]
    y_diff, z_diff = model.mesh.get_diffusivity_coordinates(x_full)
    conditions = {v.X(e): y_diff[:,model.elements.index(e)] for e in model.elements}
    conditions[v.T] = model.temperature(state.time, z_diff)
    conditions[v.P] = kwargs.get(v.P, 101325)
    conditions[v.GE] = kwargs.get(v.GE, 0)
    mob_data = LookupTable(compute_mobility)(model.therm, conditions)
    pf = []
    for md in mob_data:
        pf.append(np.sum(md.phase_fractions[md.phases==phase]))

    # Reshape phase fraction to mesh shape (Cartesian2D.unflattenResponse will give (Nx, Ny, 1))
    #pf = mesh.unflattenResponse(np.array(pf), 1)[...,0]
    pf = np.reshape(pf, (mesh.Nx, mesh.Ny))
    plot_kwargs = _adjust_kwargs(phase, {'vmin': 0, 'vmax': 1}, kwargs)
    cm = ax.pcolormesh(mesh.z_mesh[...,0]/z_scale[0], mesh.z_mesh[...,1]/z_scale[1], pf, *args, **plot_kwargs)
    ax.set_title(phase)
    ax.set_xlabel(f'Distance x*{z_scale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{z_scale[1]:.0e} (m)')
    return ax, cm

def plot_2d_fluxes(state: DiffusionState, model: DiffusionModel, element, direction, z_scale=1, ax=None, *args, **kwargs):
    '''
    Plots flux in x or y direction of element

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    element: str
        Element to plot
    direction: str
        'x' or 'y'
    z_scale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    # make sure zScale has 2 elements (for x and y axis)
    z_scale = np.atleast_1d(z_scale)
    if z_scale.shape[0] == 1:
        z_scale = z_scale[0]*np.ones(2)

    if direction != 'x' and direction != 'y':
        raise ValueError("direction must be \'x\' or \'y\'")

    fluxes = model.mesh.compute_fluxes(model.compute_pairs(state))
    if direction == 'x':
        fluxes = fluxes[0]
        z = (mesh.corners[:,:-1] + mesh.corners[:,1:]) / 2
    elif direction == 'y':
        fluxes = fluxes[1]
        z = (mesh.corners[:-1,:] + mesh.corners[1:,:]) / 2

    flux_shape = fluxes.shape
    fluxes_flat = np.reshape(fluxes, (flux_shape[0]*flux_shape[1], flux_shape[2]))
    fluxes_sub = [fluxes_flat[:,i] for i,e in enumerate(model.elements) if e not in interstitials]
    # If all dependent elements are interstitial, then flux of dependent element (substitutional) is 0
    if len(fluxes_sub) == 0:
        fluxes_sum = np.zeros((len(fluxes_flat), 1))
    else:
        fluxes_sum = 0-np.sum(fluxes_sub, axis=0)[:,np.newaxis]
    fluxes_flat = np.concatenate((fluxes_sum, fluxes_flat), axis=1)
    y = fluxes_flat[:,model.all_elements.index(element)]
    y = np.reshape(y, (flux_shape[0], flux_shape[1]))
    cm = ax.pcolormesh(z[...,0]/z_scale[0], z[...,1]/z_scale[1], y, *args, **kwargs)
    ax.set_title(f'$J_x/V_m$ {element} ($m/s$)')
    ax.set_xlabel(f'Distance x*{z_scale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{z_scale[1]:.0e} (m)')
    return ax, cm