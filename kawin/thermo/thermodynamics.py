import copy
from collections import namedtuple

import numpy as np
from tinydb import where

from pycalphad import Workspace, Model, Database, calculate, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.utils import extract_parameters
from pycalphad.core.solver import Solver

#from kawin.thermo.utils import _process_xT_arrays, _process_x
#from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.thermo.free_energy_hessian import total_dmudx
import kawin.thermo.mobility as mob_funcs

setattr(v, 'GE', v.IndependentPotential('GE'))

SampledPointsCache = namedtuple('SampledPointsCache',
                               ['temperature', 'pressure', 'samples', 'ordered_samples'],
                               defaults=(None, None, None))

def local_equilibrium(dbf, comps, phases, conds, models, phase_records, composition_sets=None, pdens=10):
    '''
    Local equilibrium calculation

    Chemical potential in a miscibility gap will be constant
    This method allows the user to get the free energy at the specified composition
    ignoring possible miscibility gaps

    Parameters
    ----------
    dbf: Database
    comps: list[str]
        List of elements to consider
    phases: list[str]
        List of phases to consider
    conds: dict[v.StateVariable, float]
        Dictionary of conditions (v.N needs to be included)
    models: dict[str, Model]
    phase_records : PhaseRecordFactory
    composition_sets: list[CompositionSet] (optional)
        If None, then composition sets will be determined through sampling
    pDens: int (optional)
        Sampling density when composition sets are not supplied

    Returns
    -------
    Dataset containing free energy and chemical potential
    '''
    # Broadcasting conditions not supported
    cur_conds = {str(k): float(v) for k, v in conds.items()}
    cur_conds = {k: v for k, v in conds.items()}

    # State variables are in alphabetical order
    if v.GE in cur_conds:
        state_variables = np.array([cur_conds[v.GE], cur_conds[v.N], cur_conds[v.P], cur_conds[v.T]], dtype=np.float64)
    else:
        state_variables = np.array([0, cur_conds[v.N], cur_conds[v.P], cur_conds[v.T]], dtype=np.float64)
    if composition_sets is None:
        # Note: filter_phases() not called, so all specified phases must be valid
        composition_sets = []

        # Special case for single phase where we can set a local phase condition for better sampling
        if len(phases) == 1:
            local_phase_conds = {v.X(phases[0], var.species): conds[var] for var in conds if isinstance(var, v.X)}
            calc_p = calculate(dbf, comps, phases[0], T=cur_conds[v.T], P=cur_conds[v.P], N=cur_conds[v.N], GE=cur_conds[v.GE],
                               pdens=pdens, model=models, phase_records=phase_records, conditions=local_phase_conds)
            idx_p = np.argmin(calc_p.GM.values.squeeze())
            compset = CompositionSet(phase_records[phases[0]])
            #For phases with a single site fraction (e.g. stoichiometric at a pure composition),
            #squeezing will result in a 0D array, so we make sure the site fractions are at least 1D
            site_fractions = np.array(calc_p.Y.isel(points=idx_p).values.squeeze())
            if len(site_fractions.shape) == 0:
                site_fractions = np.array([site_fractions])
            compset.update(site_fractions, 1.0, state_variables)
            composition_sets.append(compset)
        else:
            # Choose a naive starting point for each phase
            # only one composition set per phase is chosen
            # here, we just choose the point with the minimum Gibbs energy
            # mass balance does not have to be preserved at the starting point
            for phase in phases:
                # arbitrary guess
                phase_amt = 1./len(phases)
                calc_p = calculate(dbf, comps, phase, T=cur_conds[v.T], P=cur_conds[v.P], N=cur_conds[v.N], GE=cur_conds[v.GE],
                                pdens=pdens, model=models, phase_records=phase_records)
                idx_p = np.argmin(calc_p.GM.values.squeeze())
                compset = CompositionSet(phase_records[phase])
                site_fractions = np.array(calc_p.Y.isel(points=idx_p).values.squeeze())
                compset.update(site_fractions, phase_amt, state_variables)
                composition_sets.append(compset)
    else:
        #Update state variables in composition sets if supplied
        #pycalphad doesn't seem to update the temperature if it changes
        for cs in composition_sets:
            cs.dof[:state_variables.shape[0]] = state_variables


    # Calculate a local equilibrium for the specified phases
    solver = Solver()
    #print('initial', composition_sets)
    result = solver.solve(composition_sets, cur_conds)
    return result, composition_sets

class ExtraFreeEnergyType(v.IndependentPotential):
    implementation_units = 'joules'
    display_units = 'joules'
    display_name = 'Extra Gibbs Free Energy'

    def __init__(self):
        super().__init__('GE')

    def __reduce__(self):
        return self.__class__, ()

class ExtraGibbsModel(Model):
    """
    Subclass of pycalphad Model with extra variable GE
        GE represents any extra contribution to the Gibbs free energy
        such as the Gibbs-Thomson contribution
    """
    energy = GM = property(lambda self: self.ast + v.GE)
    formulaenergy = G = property(lambda self: (self.ast + v.GE) * self._site_ratio_normalization)
    orderingContribution = OCM = property(lambda self: self.models['ord'])

class Thermodynamics:
    """
    Class for defining driving force and essential functions for
    binary and multicomponent systems using pycalphad for equilibrium
    calculations

    Parameters
    ----------
    database : Database or str
        pycalphad Database or file name for database
    elements : list
        Elements to consider
        Note: reference element must be the first index in the list
    phases : list
        Phases involved
        Note: matrix phase must be first index in the list
    drivingForceMethod : str (optional)
        Method used to calculate driving force
        Options are 'tangent' (default), 'approximate', 'sampling' and 'curvature' (not recommended)
    parameters : list [str] or dict {str : float} or None
        List of parameters to keep symbolic in the thermodynamic or mobility models
        If None, then parameters are fixed
    """
    g_offset = 1 #Small value to add to precipitate phase for when order/disorder models are used
    state_variables = sorted([v.GE, v.N, v.P, v.T], key=str)

    def __init__(
            self,
            database: Database | str,
            elements: list[str],
            matrix: str | list[str],
            precipitates: str | list[str] = [],
            parameters: list[str] | dict[str, float] = None
            ):
        # database
        if isinstance(database, str):
            database = Database(database)
        self.dbf = database

        # elements
        self.elements = sorted(elements)
        self.nonvacant_elements = sorted(set(self.elements) - {'VA'})
        self.num_elements = len(self.nonvacant_elements)
        self.sort_indices = np.argsort(np.argsort(self.nonvacant_elements))

        # parameters, convert to dictionary mapping parameter name to value
        if parameters is None:
            self._parameters = {}
        else:
            if isinstance(parameters, list):
                self._parameters = {p: 0 for p in parameters}
            else:
                self._parameters = parameters

        if type(matrix) == str:  # check if a single phase was passed as a string instead of a list of phases.
            matrix = [matrix]
        if type(precipitates) == str:
            precipitates = [precipitates]
        self.matrix = matrix
        self.precipitates = precipitates
        self.phases = list(self.matrix) + list(self.precipitates)

        self._build_thermo_models()
        self._build_mobility_models()

        self.sampling_pdens = 2000      # Sampling density for driving force sampling method
        self.pdens = 500                # Sampling density for equilibrium
        self.local_pdens = 10           # Sampling density for local equilibrium
        self.mobility_correction = {}   # scale to increase/decrease mobility (proxy for quenched-in vacancies)
        self.vacancy_poor_interstitial_sublattice = {} # removes assumption that interstitial sublattice is mostly vacant

    def _build_thermo_models(self):
        """
        Builds thermodynamic models for each phase

        For each precipitate phase, it checks whether the phase has an order/disorder contribution
            If so and if the disordered contribution is the parent phase (ex. gamma and gamma prime in Ni alloys),
            then we add an new phase to the database that is only the disordered contribution as DIS_<phase name>
        """
        self.is_ordered_phase = {p: False for p in self.precipitates}
        for p in self.precipitates:
            if self.dbf.phases[p].model_hints.get('disordered_phase') in self.matrix:
                self.is_ordered_phase[p] = True
                self._force_disorder(self.dbf.phases[p].model_hints.get('disordered_phase'))

        # Build phase models assuming first phase is parent phase and rest of precipitate phases
        # If the same phase is used for matrix and precipitate phase, then force the matrix phase to remove the ordering contribution
        # This may be unnecessary as already disordered phase models will not be affected, but I guess just in case the matrix phase happens to be an ordered solution
        param_keys, _ = extract_parameters(self._parameters)
        self.models = {p: Model(self.dbf, self.elements, p, parameters=param_keys) for p in self.matrix}
        self.models.update({p: ExtraGibbsModel(self.dbf, self.elements, p, parameters=param_keys) for p in self.precipitates})
        for p in self.models:
            self.models[p].state_variables = self.state_variables

        self.phase_record_factory = PhaseRecordFactory(self.dbf, self.elements, self.state_variables, self.models, parameters=self._parameters)

    def _build_mobility_models(self):
        """
        Builds mobility models for phases that have model parameters
        """
        self.mobility_models = {}
        self.mobility_callables = {}
        self.diffusivity_callables = {}
        param_keys, _ = extract_parameters(self._parameters)
        phase_mob_params = {}
        phase_diff_params = {}
        for p in self.phases:
            # Get mobility/diffusivity of phase p if exists
            param_search = self.dbf.search
            param_query_mob = (
                (where('phase_name') == p) & \
                ((where('parameter_type') == 'MQ') | (where('parameter_type') == 'MF'))
            )

            param_query_diff = (
                (where('phase_name') == p) & \
                ((where('parameter_type') == 'DQ') | (where('parameter_type') == 'DF'))
            )
            phase_mob_params[p] = param_search(param_query_mob)
            phase_diff_params[p] = param_search(param_query_diff)

            if len(phase_mob_params[p]) > 0 or len(phase_diff_params[p]) > 0:
                self.mobility_models[p] = mob_funcs.MobilityModel(self.dbf, self.elements, p, parameters=param_keys)
                self.mobility_models[p].state_variables = self.state_variables

        mob_phases = list(self.mobility_models.keys())
        if len(mob_phases) > 0:
            self.mobility_phase_record_factory = PhaseRecordFactory(self.dbf, self.elements, self.state_variables, self.mobility_models, parameters=self._parameters)

        for p in self.mobility_models:
            if len(phase_mob_params[p]) > 0:
                self.mobility_callables[p] = {
                    c: self.mobility_phase_record_factory.get_phase_property(p, 'MOB_'+c, include_grad=False, include_hess=False).func
                    for c in self.phase_record_factory[p].nonvacant_elements
                    }
            if len(phase_diff_params[p]) > 0:
                self.diffusivity_callables[p] = {
                    c: self.mobility_phase_record_factory.get_phase_property(p, 'DIFF_'+c, include_grad=False, include_hess=False).func
                    for c in self.phase_record_factory[p].nonvacant_elements
                    }

    def update_parameters(self, parameters: dict[str, float]):
        """
        Update parameter dictionary with new values

        Parameters
        ----------
        parameters : dict {str : float}
            Dictionary of parameters
            NOTE: this does not have to be the full list and can also have other parameters in it
                Only the parameters that are stored upon initialization will be changed
        """
        for pm in parameters:
            if pm in self._parameters:
                self._parameters[pm] = parameters[pm]

        param_keys, param_values = extract_parameters(self._parameters)
        for p in self.phases:
            self.phase_record_factory[p].parameters[:] = np.asarray(param_values, dtype=np.float64)
            if p in self.mobility_models:
                self.mobility_phase_record_factory[p].parameters[:] = np.asarray(param_values, dtype=np.float64)

    def _force_disorder(self, phase: str):
        """
        For phases using an order/disorder model, pycalphad will neglect the disordered phase unless
        it is the only phase set active, so the order and disordered portion of the phase will use the same model

        For the Gibbs-Thomson effect to be applied, the ordered and disordered parts of the model will need to be kept separate
        As a fix, a new phase is added to the database that uses only the disordered part of the model
        We keep the original name of the disordered phase, but the disorder contribution will be the DIS_<phase name> phase
        """
        new_phase = 'DIS_' + phase
        self.dbf.phases[new_phase] = copy.deepcopy(self.dbf.phases[phase])
        self.dbf.phases[new_phase].name = new_phase

        # Remove order/disorder model hints on original name
        del self.dbf.phases[phase].model_hints['ordered_phase']
        del self.dbf.phases[phase].model_hints['disordered_phase']

        # Rename disordered phase contribution to new phase
        self.dbf.phases[new_phase].model_hints['disordered_phase'] = new_phase
        ordered_phase = self.db.phases[new_phase].model_hints['ordered_phase']
        self.dbf.phases[ordered_phase].model_hints['disordered_phase'] = new_phase

        #Copy database parameters with new name
        param_query = where('phase_name') == phase
        params = self.db.search(param_query)
        for p in params:
            #We have to create a new dictionary since p is a TinyDB.Document
            new_p = {}
            for entry in p:
                new_p[entry] = p[entry]
            new_p['phase_name'] = new_phase
            self.dbf._parameters.insert(new_p)

def _get_phase(phases, phase = None):
    return phases[0] if phase is None else phase

def _process_conditions(conditions: dict[v.StateVariable, float | list[float]]):
    """
    Adjusts conditions to have all the same lengths
    Note: we do not support broadcasting
    """
    conds = {key: value for key, value in conditions.items()}
    conds.setdefault(v.GE, 0)
    conds.setdefault(v.N, 1)
    conds.setdefault(v.P, 101325)
    max_size = 1
    for key in conds:
        conds[key] = np.atleast_1d(conds[key])
        if len(conds[key].shape) > 1:
            raise ValueError("Conditions must be scalar to 1D array")
        if len(conds[key]) > 1:
            if max_size == 1 or max_size == len(conds[key]):
                max_size = len(conds[key])
            else:
                raise ValueError("Broadcasting conditions is not supported, all arrays must be the same length")
    for key in conds:
        if max_size > 1 and len(conds[key]) == 1:
            conds[key] = conds[key][0]*np.ones(max_size)
    return conds, max_size

def generate_model_subset(therm: Thermodynamics, phases: str | list[str] = None):
    """
    Creates a subset of phases and models and updates the phase records accordingly

    Options for precPhase
        -1           -> [parent phase]
        None         -> [parent phase, 1st precipitate phase]
        str          -> [parent phase, precPhase]
        list[phases] -> precPhase
    """
    if isinstance(phases, str):
        phases = [phases]
    sub_models = {p: therm.models[p] for p in phases}
    # pycalphad seems to use the phase record model list to generate composition sets
    # so we need to make sure that the phase record models align with the subset of phases
    therm.phase_record_factory.models = sub_models
    return phases, sub_models

def get_eq(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], phases: str | list[str] = None):
    """
    Calculates equilibrium at specified x, T, gExtra

    This is separated from the interfacial composition function so that this can be used for getting curvature for interfacial composition from mobility

    Parameters
    ----------
    x : float or array
        Composition
        Needs to be array for multicomponent systems
    T : float
        Temperature
    gExtra : float
        Gibbs-Thomson contribution (if applicable)
    precPhase : str, int, list or None
        Precipitate phase (default is first precipitate)
        Options:
            None - first precipitate phase in phase list
            str  - specific precipitate phase by name
            list - all phases by name in list
            -1   - no precipitate phase

    Returns
    -------
    Workspace object
    """
    conds, num_conds = _process_conditions(conditions)
    if num_conds > 1:
        raise ValueError("Only one set of conditions is supported for global eq")
    for key in conds:
        conds[key] = conds[key][0]
    phases, model_subset = generate_model_subset(therm, phases)
    wks = Workspace(
        therm.dbf, therm.elements, phases, conds,
        models=model_subset, phase_record_factory=therm.phase_record_factory,
        calc_opts={'pdens': therm.pdens}
        )
    return wks

def get_local_eq(therm: Thermodynamics, conditions: dict[v.StateVariable, float], phases: str | list[str] = None, composition_sets: list[CompositionSet] = None):
    """
    Calculates local equilibrium at specified x, T, gExtra

    Local equilibrium is defined as all phases have a single composition set
        Miscibility gaps are ignored
        In the case of a single phase with a miscibility gap, this returns the composition
        set at x

    Parameters
    ----------
    x : float or array
        Composition
        Needs to be array for multicomponent systems
    T : float
        Temperature
    gExtra : float
        Gibbs-Thomson contribution (if applicable)
    precPhase : str, int, list or None
        Precipitate phase (default is first precipitate)
        Options:
            None - first precipitate phase in phase list
            str  - specific precipitate phase by name
            list - all phases by name in list
            -1   - no precipitate phase
    composition_sets : list[CompositionSet] or None
        Composition sets to use in equilibrium, this will override
        the phase list and use only the supplied composition sets
        If None, then composition sets are generated for each phase

    Returns
    -------
    result - equilibrium convergence and chemical potentials
    composition_sets - list of CompositionSet for phases in "equilibrium"
        Note - "equilibrium" in terms of the matrix and singled out precipitate phase (or just matrix if precPhase is -1)
    """
    conds, num_conds = _process_conditions(conditions)
    if num_conds > 1:
        raise ValueError("Only one set of conditions is supported for local eq")
    for key in conds:
        conds[key] = conds[key][0]
    phases, model_subset = generate_model_subset(therm, phases)
    return local_equilibrium(
        therm.dbf, therm.elements, phases, conds,
        model_subset, therm.phase_record_factory,
        composition_sets=composition_sets, pdens=therm.local_pdens
        )

def compute_interdiffusivity(therm: Thermodynamics, ref_element: str, conditions: dict[v.StateVariable, float | list[float]], phase: str = None, cache: dict[str, list[CompositionSet]] = {}):
    """
    Gets interdiffusivity at unique composition and temperature

    Parameters
    ----------
    x : float or array
        Composition
    T : float
        Temperature
    removeCache : boolean
    phase : str

    Returns
    -------
    Interdiffusivity as a matrix (will return float in binary case)
    """
    conds, num_conds = _process_conditions(conditions)
    phase = _get_phase(therm.phases, phase)
    composition_sets = None
    if cache is not None:
        composition_sets = cache.get(f'{phase}_diff', None)
    Dnkjs = np.zeros((num_conds, therm.num_elements-1, therm.num_elements-1))
    for i in range(num_conds):
        single_conds = {key: value[i] for key, value in conds.items()}
        result, composition_sets = get_local_eq(therm, single_conds, [phase], composition_sets=composition_sets)
        cs_matrix = composition_sets[0]
        chemical_potentials = result.chemical_potentials

        # Get interdiffusivity from mobility or diffusivity models whichever is available
        # If both mobility and diffusivity models exist, then favor the mobility model
        if therm.mobility_callables[phase] is None:
            Dnkj = mob_funcs.interdiffusivity_from_diff(
                cs_matrix, ref_element, therm.diffusivity_callables[phase],
                diffusivity_correction=therm.mobility_correction, parameters=therm._parameters
                )
        else:
            Dnkj = mob_funcs.interdiffusivity(
                chemical_potentials, cs_matrix, ref_element, therm.mobility_callables[phase],
                mobility_correction=therm.mobility_correction,
                vacancy_poor_interstitial_sublattice=therm.vacancy_poor_interstitial_sublattice.get(phase, False),
                parameters=therm._parameters
                )

        if cache is not None:
            cache[f'{phase}_diff'] = composition_sets

        Dnkjs[i] = Dnkj

    return np.squeeze(Dnkjs)

def compute_tracer_diffusivity(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], phase: str = None, cache: dict[str, list[CompositionSet]] = {}):
    """
    Gets tracer diffusivity at unique composition and temperature

    Parameters
    ----------
    x : float or array
        Composition
    T : float
        Temperature
    el : str
        Element to calculate diffusivity

    Returns
    -------
    Tracer diffusivity as a float
    """
    conds, num_conds = _process_conditions(conditions)
    phase = _get_phase(therm.phases, phase)
    composition_sets = None
    if cache is not None:
        composition_sets = cache.get(f'{phase}_diff', None)
    diffs = np.zeros((num_conds, therm.num_elements))
    for i in range(num_conds):
        single_conds = {key: value[i] for key, value in conds.items()}
        result, composition_sets = get_local_eq(therm, single_conds, [phase], composition_sets=composition_sets)
        cs_matrix = composition_sets[0]

        # Get interdiffusivity from mobility or diffusivity models whichever is available
        # If both mobility and diffusivity models exist, then favor the mobility model
        if therm.mobility_callables[phase] is None:
            #NOTE: This is not tested yet
            dtrace = mob_funcs.tracer_diffusivity_from_diff(
                cs_matrix, therm.diffusivity_callables[phase],
                diffusivity_correction=therm.mobility_correction, parameters=therm._parameters
                )
        else:
            dtrace = mob_funcs.tracer_diffusivity(
                cs_matrix, therm.mobility_callables[phase],
                mobility_correction=therm.mobility_correction, parameters=therm._parameters
                )

        if cache is not None:
            cache[f'{phase}_diff'] = composition_sets

        diffs[i] = dtrace

    return np.squeeze(diffs)

def _get_cs_eq(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_phase, prec_phase, cache = {}):
    """
    Gets composition set from x and T by global equilibrium

    Steps
        1. Compute equilibrium at x and T
            If equilibrium did not converge or matrix phase is not stable, then return None
        2. Get composition sets and add to cache
            If precipitate is not stable, the return None
        3. Resolve possible issues with miscibility gaps
        4. Return values

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)

    Returns
    -------
    chemical_potentials
    cs_matrix - composition set of matrix phase
    cs_precip - composition set of precipitate phase
    """
    # Takes list of composition sets and gets matrix composition, precipitate composition,
    # and whether there is a miscibility gap
    # If matrix or precipitate is unstable, then the composition set will be None
    def _process_composition_sets(composition_sets):
        cs_list_matrix = [cs for cs in composition_sets if cs.phase_record.phase_name == matrix_phase]
        cs_list_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == prec_phase]
        cs_matrix = None if len(cs_list_matrix) == 0 else cs_list_matrix[0]
        cs_precip = None if len(cs_list_precip) == 0 else cs_list_precip[0]
        miscibility_gap = len(cs_list_matrix) > 1 or len(cs_list_precip) > 1
        return cs_matrix, cs_precip, miscibility_gap

    # Updates a list of composition sets with new conditions
    # Return chemical potential, matrix comp set, precipitate comp set, and whether there is a miscibility gap
    def _update_composition_sets(composition_sets):
        conditions = _process_conditions(conditions)
        phases, sub_models = generate_model_subset(therm, [matrix_phase, prec_phase])
        result, composition_sets = local_equilibrium(
            therm.dbf, therm.elements, phases,
            conditions, sub_models, therm.phase_record_factory, composition_sets,
            therm.local_pdens
            )
        cs_matrix, cs_precip, miscibility_gap = _process_composition_sets(composition_sets)
        return result.chemical_potentials, cs_matrix, cs_precip, miscibility_gap

    # If no cache exists, then compute global equilibrium, else, update cached composition sets
    cache_id = frozenset({matrix_phase, prec_phase})
    if cache.get(cache_id, None) is None:
        wks = get_eq(therm, conditions, [matrix_phase, prec_phase])
        cs_matrix, cs_precip, miscibility_gap = _process_composition_sets(wks.get_composition_sets())
        chemical_potentials = np.squeeze(wks.eq.MU)
    else:
        chemical_potentials, cs_matrix, cs_precip, miscibility_gap = _update_composition_sets(cache[cache_id])

    # If invalid equilibrium, then return None to denote that we cannot use this calculation
    if any(np.isnan(chemical_potentials)):
            return None

    # If the matrix or precipitate is unstable, then we return everything as usual
    # This is in case we want to attempt to find a condition where the two phases are stable
    if cs_matrix is None or cs_precip is None:
        return chemical_potentials, cs_matrix, cs_precip

    # Check for miscibility gaps, if so, then compute local equilibrium with a single comp set of matrix and precipitate
    if miscibility_gap:
        chemical_potentials, cs_matrix, cs_precip, miscibility_gap = _update_composition_sets([cs_matrix, cs_precip])

    return chemical_potentials, cs_matrix, cs_precip

def _get_cs_for_df(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_phase, prec_phase, cache = {}):
    """
    Wrapper for getting composition set from x and T by either global equilibrium or local from a cached composition set

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)

    Returns
    -------
    chemical_potentials
    cs_matrix - composition set of matrix phase
    cs_precip - composition set of precipitate phase
    """
    eq_results = _get_cs_eq(therm, conditions, matrix_phase, prec_phase, cache)
    cache_id = frozenset({matrix_phase, prec_phase})
    if eq_results is None:
        cache[cache_id] = None
        return None
    else:
        chemical_potentials, cs_matrix, cs_precip = eq_results
        if cs_matrix is None or cs_precip is None:
            cache[cache_id] = None
            return None
        cache[cache_id] = [cs_matrix, cs_precip]
        return chemical_potentials, cs_matrix, cs_precip


def _sample_precipitate_cs(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_chem_pot, prec_phase, local_phase_sampling_conditions = None, cache = {}):
    """
    Gets samples for precipitate phase for use in sampling driving force method and returns driving force and precipitate composition

    This is also use in tangent driving force method for when equilibrium is not (yet) cached

    Steps
        1. Compute local equilibrium at x and T of only the matrix phase
        2. Sample precipitate phase
            If ordered contribution to matrix phase, then sample ordering contribution
            and remove points on the matrix free energy surface
        3. Compute energy difference between precipitate samples and chemical potential hyperplane
        4. Find sample that maximizes energy difference and return sample composition and driving force

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)

    Returns
    -------
    driving force - max free energy difference
    precipitate composition set - corresponds to max driving force
    """
    orderTol = -1e-8
    state_cond = {v.GE: therm.g_offset, v.N: 1, v.P: conditions.get(v.P, 101315), v.T: conditions[v.T]}
    str_cond = {str(key): val for key,val in state_cond.items()}

    #Sample precipitate phase and get driving force differences at all points -------------------------------------------------------------------
    #Sample points of precipitate phase
    cache_id = f"{prec_phase}_samples"
    phases, sub_models = generate_model_subset(therm, [prec_phase])
    sample_data = cache.get(cache_id, SampledPointsCache())
    prev_T = sample_data.temperature
    prev_P = sample_data.pressure
    prec_points = sample_data.samples
    ordered_points = sample_data.ordered_samples

    if prec_points is None or prev_T != conditions[v.T] or prev_P != conditions[v.P]:
        prec_points = calculate(therm.dbf, therm.elements, phases[0],
                                pdens=therm.sampling_pdens, model=sub_models, output='GM',
                                phase_records=therm.phase_record_factory, conditions=local_phase_sampling_conditions,
                                to_xarray=False, **str_cond)
        if therm.is_ordered_phase[prec_phase]:
            ordered_points = calculate(therm.dbf, therm.elements, phases[0],
                                        pdens=therm.sampling_pdens, model=sub_models, output='OCM',
                                        phase_records=therm.phase_record_factory, to_xarray=False, **str_cond)
        cache[cache_id] = SampledPointsCache(temperature=conditions[v.T], pressure=conditions[v.P], samples=prec_points, ordered_samples=ordered_points)

    #For phases at fixed composition, there will only be 1 set of site fractions
    #So we force composition and site fractions to be 2D
    prec_comp = np.atleast_2d(np.squeeze(prec_points.X))
    y = np.atleast_2d(np.squeeze(prec_points.Y))
    gm = np.atleast_1d(np.squeeze(prec_points.GM))
    mu = np.atleast_2d(matrix_chem_pot)

    #Difference between the chemical potential hyperplane and the free energy of the samples
    #   Chemical potential hyperplane is the free energy on the plane at the composition of the samples
    #The max driving force is the same as when the chemical potentials of the two phases are parallel
    mult = prec_comp * mu
    diff = np.sum(mult, axis=1) - np.squeeze(gm)

    #Find maximum driving force and corresponding composition -----------------------------------------------------------------------------------
    #For phases with order/disorder transition, a filter is applied such that it will only use points that are below the disordered energy surface
    if therm.is_ordered_phase[prec_phase]:
        indices = np.squeeze(ordered_points.OCM) < orderTol
        diff = diff[indices]
        y = y[indices]

    dg = np.amax(diff)
    idx = np.argmax(diff)

    prec_cs = CompositionSet(therm.phase_record_factory[prec_phase])
    state_variables = np.array([state_cond[v.GE], state_cond[v.N], state_cond[v.P], state_cond[v.T]], dtype=np.float64)
    y = np.array(y[idx][:prec_cs.phase_record.phase_dof], dtype=np.float64)
    prec_cs.update(y, 1, state_variables)

    return dg, prec_cs


def compute_driving_force_sampling(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_phase, prec_phase, local_phase_sampling_conditions = None, cache = {}):
    """
    Gets driving force for nucleation by sampling

    Steps
        1. Compute local equilibrium at x and T of only the matrix phase
        2. Sample precipitate phase
            If ordered contribution to matrix phase, then sample ordering contribution
            and remove points on the matrix free energy surface
        3. Compute energy difference between precipitate samples and chemical potential hyperplane
        4. Find sample that maximizes energy difference and return sample composition and driving force

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    # Calculate equilibrium with only the parent phase
    # Return (None, None) if equilibrium was not successful
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    result, matrix_cs = get_local_eq(therm, conditions, [matrix_phase], composition_sets=matrix_cs)
    #result, self._matrix_cs = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=self._matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    # Get precipitate composition set that maximizes driving force
    #dg, prec_cs = self._getPrecCompositionSetSamplingDF(x, T, result.chemical_potentials, precPhase, local_phase_sampling_conditions)
    dg, prec_cs = _sample_precipitate_cs(therm, conditions, result.chemical_potentials, prec_phase, local_phase_sampling_conditions, cache)

    # Sort precipitate composition from alphabetical to input order of elements
    #sortIndices = np.argsort(self.elements[:-1])
    #unsortIndices = np.argsort(sortIndices)
    beta_x = np.array(prec_cs.X, dtype=np.float64)
    #beta_x = beta_x[unsortIndices]
    beta_x = beta_x[therm.sort_indices[1:]]

    cache[f'{matrix_phase}_dg'] = matrix_cs
    return np.squeeze(dg), np.squeeze(beta_x)


def compute_driving_force_approximate(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_phase, prec_phase, local_phase_sampling_conditions = None, cache = {}):
    """
    Approximate method of driving force calculation
    Assumes equilibrium composition of precipitate phase

    Sampling method is used if driving force is negative

    Steps:
        1. Compute equilibrium and get composition sets for matrix and precipitate phase
        2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
            If not, then resort to sampling method
        3. Compute equilibrium at matrix composition and get chemical potential hyperplane
        4. Driving force is the difference between the free energy of the precipitate (from step 1)
            and the free energy on the chemical potential hyperplane (from step 3) at the precipitate composition

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    cs_results = _get_cs_for_df(therm, conditions, matrix_phase, prec_phase)
    if cs_results is None:
        return compute_driving_force_sampling(therm, conditions, matrix_phase, prec_phase, local_phase_sampling_conditions, cache)
    #cs_results = self._getCompositionSetsForDF(x, T, precPhase)
    #if cs_results is None:
    #    return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache, local_phase_sampling_conditions=local_phase_sampling_conditions)
    chemical_potentials, cs_matrix, cs_precip = cs_results

    # Calculate equilibrium with only the parent phase
    # Return (None, None) if equilibrium was not successful
    result, matrix_cs = get_local_eq(therm, conditions, [matrix_phase], composition_sets=matrix_cs)
    #result, self._matrix_cs = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=self._matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    xP = np.array(cs_precip.X, dtype=np.float64)
    dg = np.sum(xP * result.chemical_potentials) - np.sum(xP * chemical_potentials)

    #sortIndices = np.argsort(self.elements[:-1])
    #unsortIndices = np.argsort(sortIndices)

    #self._resetDrivingForceCache(precPhase, removeCache)
    #return np.squeeze(dg), np.squeeze(xP[unsortIndices[1:]])
    cache[f'{matrix_phase}_dg'] = matrix_cs
    return np.squeeze(dg), np.squeeze(xP[therm.sort_indices][1:])

def compute_driving_force_curvature(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_phase, prec_phase, local_phase_sampling_conditions = None, cache = {}):
    """
    Gets driving force from curvature of free energy function
    Assumes small saturation

    Steps:
        1. Compute equilibrium and get composition sets for matrix and precipitate phase
        2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
            If not, then resort to sampling method
        3. Get dmu/dx (free energy curvature)
        4. Compute (x_infty - x_matrix) * dmu/dx * (x_prec - x_matrix)^T
            This does a first (or second?) order approximation of the driving force based off the curvature at x_infty

    Sampling method is used if driving force is negative

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    cs_results = _get_cs_for_df(therm, conditions, matrix_phase, prec_phase)
    if cs_results is None:
        return compute_driving_force_sampling(therm, conditions, matrix_phase, prec_phase, local_phase_sampling_conditions, cache)
    #cs_results = self._getCompositionSetsForDF(x, T, precPhase)
    #if cs_results is None:
    #    return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache, local_phase_sampling_conditions=local_phase_sampling_conditions)

    chemical_potentials, cs_matrix, cs_precip = cs_results
    non_va_elements = list(cs_matrix.phase_record.nonvacant_elements)
    ref_index = non_va_elements.index(therm.elements[0])
    x_matrix = np.array(cs_matrix.X, dtype=np.float64)
    x_precip = np.array(cs_precip.X, dtype=np.float64)

    #If in two phase region, then get curvature of parent phase and use it to calculate driving force
    #sortIndices = np.argsort(self.elements[1:-1])
    #unsortIndices = np.argsort(sortIndices)

    dmudx_parent = mob_funcs.total_dmudx(chemical_potentials, cs_matrix, therm.elements[0])
    xM = np.delete(x_matrix, ref_index)
    xP = np.delete(x_precip, ref_index)
    xBar = np.array([xP - xM])

    #x = x[sortIndices]
    x = x[therm.sort_indices[1:]]
    xD = np.array([x - xM])

    dg = np.matmul(xD, np.matmul(dmudx_parent, xBar.T))

    #self._resetDrivingForceCache(precPhase, removeCache)
    return np.squeeze(dg), np.squeeze(xP[therm.sort_indices[1:]])


def compute_driving_force_tangent(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], matrix_phase, prec_phase, local_phase_sampling_conditions = None, cache = {}):
    """
    Gets driving force from parallel tangent calculation

    Steps
        1. Compute equilibrium to get composition sets (or used previous cached CS)
        2. Compute equilibrium of matrix phase at matrix composition
        3. Remove composition and extra free energy from conditions
        4. Add chemical potential for each component to conditions
        5. Compute equilibrium of precipitate phase with new conditions
            The calculated v.GE is the driving force

    This will work for positive and negative driving forces

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    # Calculate equilibrium with only the parent phase
    # Return (None, None) if equilibrium was not successful
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    result, matrix_cs = get_local_eq(therm, conditions, [matrix_phase], composition_sets=matrix_cs)
    #result, self._matrix_cs = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=self._matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    # Setup conditions, which include T, P, N and MU (for all elements)
    # The nonvacant elements list in the matrix phase record will be sorted
    cond = {v.T: conditions[v.T], v.P: conditions.get(v.P, 101325), v.N: conditions.get(v.N, 1)}
    non_va_elements = matrix_cs[0].phase_record.nonvacant_elements
    cond.update({v.MU(e): result.chemical_potentials[i] for i,e in enumerate(non_va_elements)})
    #cond = {v.T: T, v.P: 101325, v.N: 1}
    #non_va_elements = self._matrix_cs[0].phase_record.nonvacant_elements
    #cond.update({v.MU(e): result.chemical_potentials[i] for i,e in enumerate(non_va_elements)})

    # If we do not have a precipitate composition set, then find one by sampling
    if cache.get(f'{prec_phase}_dg', None) is None:
        dg, prec_cs = _sample_precipitate_cs(therm, conditions, result.chemical_potentials, prec_phase, local_phase_sampling_conditions)
        cache[f'{prec_phase}_dg'] = [prec_cs]
    #if self._compset_cache_df.get(precPhase, None) is None:
    #    dg, prec_cs = self._getPrecCompositionSetSamplingDF(x, T, result.chemical_potentials, precPhase, local_phase_sampling_conditions)
    #    self._compset_cache_df[precPhase] = [prec_cs]

    #Solving for local equilibrium on precipitate
    #The fixed conditions are T, P and MU, so this should solve for precipitate composition and GE
    #   Rather than solving for parallel tangent where the driving force is the difference between the chemical potentials of matrix and precipitate phase
    #   This instead solves for the offset in the precipitate energy surface to make the precipitate lie on the chemical potential hyperplane of the matrix phase
    phases, sub_models = generate_model_subset(therm, [prec_phase])
    prec_eq_results, prec_cs = local_equilibrium(
        therm.dbf, therm.elements, phases,
        cond, sub_models, therm.phase_record_factory,
        composition_sets=cache[f'{prec_phase}_dg'], pdens=therm.local_pdens
        )
    #phases, sub_models = self._setupSubModels([precPhase])
    #prec_eq_results, prec_cs = local_equilibrium(self.db, self.elements, phases, cond,
    #                                                sub_models, self.phase_records,
    #                                                composition_sets=self._compset_cache_df[precPhase], pDens=self.local_pDens)
    if any(np.isnan(prec_eq_results.chemical_potentials)):
        return None, None

    # NOTE: we assum that v.GE is the first state variable in alphabetical order
    dg = prec_eq_results.x[0]
    xb = np.array(prec_cs[0].X)

    #Check if precipitate composition at equilibrium is the matrix composition
    #This can occur in order/disordered models where the miscibility gap is small enough that the parallel tangent can only be found at the matrix composition
    #In this case, switch to sampling for the driving force
    #This still seems to be an improvement over approximate and curvature methods since this occurs after the driving force becomes negative
    mat_comps = np.array(matrix_cs[0].X, dtype=np.float64)
    if np.allclose(xb, mat_comps, 1e-6):
        cache[f'{prec_phase}_dg'] = None
        return compute_driving_force_sampling(therm, conditions, matrix_phase, prec_phase, local_phase_sampling_conditions, cache)
    #mat_comps = np.array(self._matrix_cs[0].X, dtype=np.float64)
    #if np.allclose(xb, mat_comps, 1e-6):
    #    self._compset_cache_df[precPhase] = None
    #    return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache, local_phase_sampling_conditions=local_phase_sampling_conditions)

    # If all goes well, then we can store the cache
    cache[f'{prec_phase}_dg'] = prec_cs
    cache[f'{matrix_phase}_dg'] = matrix_cs
    #self._compset_cache_df[precPhase] = prec_cs

    #sortIndices = np.argsort(self.elements[:-1])
    #unsortIndices = np.argsort(sortIndices)

    #self._resetDrivingForceCache(precPhase, removeCache)

    #return np.squeeze(dg), np.squeeze(xb[unsortIndices[1:]])
    return np.squeeze(dg), np.squeeze(xb[therm.sort_indices[1:]])

def getDrivingForce(self, x, T, precPhase = None, removeCache = False, local_phase_sampling_conditions = None):
    """
    Gets driving force using method defined upon initialization

    Parameters
    ----------
    x : float, array or 2D array
        Composition of minor element in bulk matrix phase
        For binary system, use an array for multiple compositions
        For multicomponent systems, use a 2D array for multiple compositions
            Where 0th axis is for indices of each composition
    T : float or array
        Temperature in K
        Must be same length as x if x is array or 2D array
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    x, T = _process_xT_arrays(x, T, self.numElements == 2)
    precPhase = _getPrecipitatePhase(self.phases, precPhase)
    dgArray, compArray = zip(*[self._drivingForce(xi, Ti, precPhase, removeCache, local_phase_sampling_conditions) for xi, Ti in zip(x, T)])
    return np.squeeze(dgArray), np.squeeze(compArray)

def _resetDrivingForceCache(self, phase, removeCache):
    if removeCache:
        self._compset_cache_df[phase] = None
        self._matrix_cs = None
        self._points_cache[phase] = SampledPointsCache()

def _getDrivingForceSampling(self, x, T, precPhase, removeCache = False, local_phase_sampling_conditions = None):
    """
    Gets driving force for nucleation by sampling

    Steps
        1. Compute local equilibrium at x and T of only the matrix phase
        2. Sample precipitate phase
            If ordered contribution to matrix phase, then sample ordering contribution
            and remove points on the matrix free energy surface
        3. Compute energy difference between precipitate samples and chemical potential hyperplane
        4. Find sample that maximizes energy difference and return sample composition and driving force

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    # Calculate equilibrium with only the parent phase
    # Return (None, None) if equilibrium was not successful
    result, self._matrix_cs = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=self._matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    # Get precipitate composition set that maximizes driving force
    dg, prec_cs = self._getPrecCompositionSetSamplingDF(x, T, result.chemical_potentials, precPhase, local_phase_sampling_conditions)

    # Sort precipitate composition from alphabetical to input order of elements
    sortIndices = np.argsort(self.elements[:-1])
    unsortIndices = np.argsort(sortIndices)
    beta_x = np.array(prec_cs.X, dtype=np.float64)
    beta_x = beta_x[unsortIndices]

    self._resetDrivingForceCache(precPhase, removeCache)
    return np.squeeze(dg), np.squeeze(beta_x[1:])

def _getDrivingForceApprox(self, x, T, precPhase, removeCache = False, local_phase_sampling_conditions = None):
    """
    Approximate method of driving force calculation
    Assumes equilibrium composition of precipitate phase

    Sampling method is used if driving force is negative

    Steps:
        1. Compute equilibrium and get composition sets for matrix and precipitate phase
        2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
            If not, then resort to sampling method
        3. Compute equilibrium at matrix composition and get chemical potential hyperplane
        4. Driving force is the difference between the free energy of the precipitate (from step 1)
            and the free energy on the chemical potential hyperplane (from step 3) at the precipitate composition

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    cs_results = self._getCompositionSetsForDF(x, T, precPhase)
    if cs_results is None:
        return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache, local_phase_sampling_conditions=local_phase_sampling_conditions)
    chemical_potentials, cs_matrix, cs_precip = cs_results

    # Calculate equilibrium with only the parent phase
    # Return (None, None) if equilibrium was not successful
    result, self._matrix_cs = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=self._matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    xP = np.array(cs_precip.X, dtype=np.float64)
    dg = np.sum(xP * result.chemical_potentials) - np.sum(xP * chemical_potentials)

    sortIndices = np.argsort(self.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    self._resetDrivingForceCache(precPhase, removeCache)
    return np.squeeze(dg), np.squeeze(xP[unsortIndices[1:]])

def _getDrivingForceCurvature(self, x, T, precPhase, removeCache = False, local_phase_sampling_conditions = None):
    """
    Gets driving force from curvature of free energy function
    Assumes small saturation

    Steps:
        1. Compute equilibrium and get composition sets for matrix and precipitate phase
        2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
            If not, then resort to sampling method
        3. Get dmu/dx (free energy curvature)
        4. Compute (x_infty - x_matrix) * dmu/dx * (x_prec - x_matrix)^T
            This does a first (or second?) order approximation of the driving force based off the curvature at x_infty

    Sampling method is used if driving force is negative

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    cs_results = self._getCompositionSetsForDF(x, T, precPhase)
    if cs_results is None:
        return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache, local_phase_sampling_conditions=local_phase_sampling_conditions)

    chemical_potentials, cs_matrix, cs_precip = cs_results
    non_va_elements = list(cs_matrix.phase_record.nonvacant_elements)
    refIndex = non_va_elements.index(self.elements[0])
    x_matrix = np.array(cs_matrix.X, dtype=np.float64)
    x_precip = np.array(cs_precip.X, dtype=np.float64)

    #If in two phase region, then get curvature of parent phase and use it to calculate driving force
    sortIndices = np.argsort(self.elements[1:-1])
    unsortIndices = np.argsort(sortIndices)

    dMudxParent = dMudX(chemical_potentials, cs_matrix, self.elements[0])
    xM = np.delete(x_matrix, refIndex)
    xP = np.delete(x_precip, refIndex)
    xBar = np.array([xP - xM])

    x = x[sortIndices]
    xD = np.array([x - xM])

    dg = np.matmul(xD, np.matmul(dMudxParent, xBar.T))

    self._resetDrivingForceCache(precPhase, removeCache)
    return np.squeeze(dg), np.squeeze(xP[unsortIndices])

def _getDrivingForceTangent(self, x, T, precPhase, removeCache = False, local_phase_sampling_conditions = None):
    """
    Gets driving force from parallel tangent calculation

    Steps
        1. Compute equilibrium to get composition sets (or used previous cached CS)
        2. Compute equilibrium of matrix phase at matrix composition
        3. Remove composition and extra free energy from conditions
        4. Add chemical potential for each component to conditions
        5. Compute equilibrium of precipitate phase with new conditions
            The calculated v.GE is the driving force

    This will work for positive and negative driving forces

    Parameters
    ----------
    x : float or array
        Composition of minor element in bulk matrix phase
        Use float for binary systems
        Use array for multicomponent systems
    T : float
        Temperature in K
    precPhase : str (optional)
        Precipitate phase to consider (default is first precipitate phase in list)
    removeCache : bool (optional)
        If True, this will not cache any equilibrium
        This is used for training since training points may not be near each other

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    # Calculate equilibrium with only the parent phase
    # Return (None, None) if equilibrium was not successful
    result, self._matrix_cs = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=self._matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    # Setup conditions, which include T, P, N and MU (for all elements)
    # The nonvacant elements list in the matrix phase record will be sorted
    cond = {v.T: T, v.P: 101325, v.N: 1}
    non_va_elements = self._matrix_cs[0].phase_record.nonvacant_elements
    cond.update({v.MU(e): result.chemical_potentials[i] for i,e in enumerate(non_va_elements)})

    # If we do not have a precipitate composition set, then find one by sampling
    if self._compset_cache_df.get(precPhase, None) is None:
        dg, prec_cs = self._getPrecCompositionSetSamplingDF(x, T, result.chemical_potentials, precPhase, local_phase_sampling_conditions)
        self._compset_cache_df[precPhase] = [prec_cs]

    #Solving for local equilibrium on precipitate
    #The fixed conditions are T, P and MU, so this should solve for precipitate composition and GE
    #   Rather than solving for parallel tangent where the driving force is the difference between the chemical potentials of matrix and precipitate phase
    #   This instead solves for the offset in the precipitate energy surface to make the precipitate lie on the chemical potential hyperplane of the matrix phase
    phases, sub_models = self._setupSubModels([precPhase])
    prec_eq_results, prec_cs = local_equilibrium(self.db, self.elements, phases, cond,
                                                    sub_models, self.phase_records,
                                                    composition_sets=self._compset_cache_df[precPhase], pDens=self.local_pDens)
    if any(np.isnan(prec_eq_results.chemical_potentials)):
        return None, None

    # NOTE: we assum that v.GE is the first state variable in alphabetical order
    dg = prec_eq_results.x[0]
    xb = np.array(prec_cs[0].X)

    #Check if precipitate composition at equilibrium is the matrix composition
    #This can occur in order/disordered models where the miscibility gap is small enough that the parallel tangent can only be found at the matrix composition
    #In this case, switch to sampling for the driving force
    #This still seems to be an improvement over approximate and curvature methods since this occurs after the driving force becomes negative
    mat_comps = np.array(self._matrix_cs[0].X, dtype=np.float64)
    if np.allclose(xb, mat_comps, 1e-6):
        self._compset_cache_df[precPhase] = None
        return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache, local_phase_sampling_conditions=local_phase_sampling_conditions)

    # If all goes well, then we can store the cache
    self._compset_cache_df[precPhase] = prec_cs

    sortIndices = np.argsort(self.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    self._resetDrivingForceCache(precPhase, removeCache)
    return np.squeeze(dg), np.squeeze(xb[unsortIndices[1:]])

# def _getCompositionSetsForDF(self, x, T, precPhase):
#     """
#     Wrapper for getting composition set from x and T by either global equilibrium or local from a cached composition set

#     Parameters
#     ----------
#     x : float or array
#         Composition of minor element in bulk matrix phase
#         Use float for binary systems
#         Use array for multicomponent systems
#     T : float
#         Temperature in K
#     precPhase : str (optional)
#         Precipitate phase to consider (default is first precipitate phase in list)

#     Returns
#     -------
#     chemical_potentials
#     cs_matrix - composition set of matrix phase
#     cs_precip - composition set of precipitate phase
#     """
#     eq_results = self._getCompositionSetsEq(x, T, precPhase, self._compset_cache_df)
#     if eq_results is None:
#         self._compset_cache_df[precPhase] = None
#         return None
#     else:
#         chemical_potentials, cs_matrix, cs_precip = eq_results
#         if cs_matrix is None or cs_precip is None:
#             self._compset_cache_df[precPhase] = None
#             return None
#         self._compset_cache_df[precPhase] = [cs_matrix, cs_precip]
#         return chemical_potentials, cs_matrix, cs_precip

# def _getCompositionSetsEq(self, x, T, precPhase, cached_composition_sets = {}):
#     """
#     Gets composition set from x and T by global equilibrium

#     Steps
#         1. Compute equilibrium at x and T
#             If equilibrium did not converge or matrix phase is not stable, then return None
#         2. Get composition sets and add to cache
#             If precipitate is not stable, the return None
#         3. Resolve possible issues with miscibility gaps
#         4. Return values

#     Parameters
#     ----------
#     x : float or array
#         Composition of minor element in bulk matrix phase
#         Use float for binary systems
#         Use array for multicomponent systems
#     T : float
#         Temperature in K
#     precPhase : str (optional)
#         Precipitate phase to consider (default is first precipitate phase in list)

#     Returns
#     -------
#     chemical_potentials
#     cs_matrix - composition set of matrix phase
#     cs_precip - composition set of precipitate phase
#     """
#     # Takes list of composition sets and gets matrix composition, precipitate composition,
#     # and whether there is a miscibility gap
#     # If matrix or precipitate is unstable, then the composition set will be None
#     def _process_composition_sets(composition_sets):
#         cs_list_matrix = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]]
#         cs_list_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase]
#         cs_matrix = None if len(cs_list_matrix) == 0 else cs_list_matrix[0]
#         cs_precip = None if len(cs_list_precip) == 0 else cs_list_precip[0]
#         miscibility_gap = len(cs_list_matrix) > 1 or len(cs_list_precip) > 1
#         return cs_matrix, cs_precip, miscibility_gap

#     # Updates a list of composition sets with new conditions
#     # Return chemical potential, matrix comp set, precipitate comp set, and whether there is a miscibility gap
#     def _update_composition_sets(composition_sets):
#         cond = self._getConditions(x, T)
#         phases, sub_models = self._setupSubModels([self.phases[0], precPhase])
#         result, composition_sets = local_equilibrium(self.db, self.elements, phases, cond,
#                                                         sub_models, self.phase_records,
#                                                         composition_sets=composition_sets, pDens=self.local_pDens)
#         cs_matrix, cs_precip, miscibility_gap = _process_composition_sets(composition_sets)
#         return result.chemical_potentials, cs_matrix, cs_precip, miscibility_gap

#     # If no cache exists, then compute global equilibrium, else, update cached composition sets
#     if cached_composition_sets.get(precPhase, None) is None:
#         wks = self.getEq(x, T, 0, precPhase)
#         cs_matrix, cs_precip, miscibility_gap = _process_composition_sets(wks.get_composition_sets())
#         chemical_potentials = np.squeeze(wks.eq.MU)
#     else:
#         chemical_potentials, cs_matrix, cs_precip, miscibility_gap = _update_composition_sets(cached_composition_sets[precPhase])

#     # If invalid equilibrium, then return None to denote that we cannot use this calculation
#     if any(np.isnan(chemical_potentials)):
#             return None

#     # If the matrix or precipitate is unstable, then we return everything as usual
#     # This is in case we want to attempt to find a condition where the two phases are stable
#     if cs_matrix is None or cs_precip is None:
#         return chemical_potentials, cs_matrix, cs_precip

#     # Check for miscibility gaps, if so, then compute local equilibrium with a single comp set of matrix and precipitate
#     if miscibility_gap:
#         chemical_potentials, cs_matrix, cs_precip, miscibility_gap = _update_composition_sets([cs_matrix, cs_precip])

#     return chemical_potentials, cs_matrix, cs_precip

# def _getPrecCompositionSetSamplingDF(self, x, T, matrix_chem_pot, precPhase, local_phase_sampling_conditions = None):
#     """
#     Gets samples for precipitate phase for use in sampling driving force method and returns driving force and precipitate composition

#     This is also use in tangent driving force method for when equilibrium is not (yet) cached

#     Steps
#         1. Compute local equilibrium at x and T of only the matrix phase
#         2. Sample precipitate phase
#             If ordered contribution to matrix phase, then sample ordering contribution
#             and remove points on the matrix free energy surface
#         3. Compute energy difference between precipitate samples and chemical potential hyperplane
#         4. Find sample that maximizes energy difference and return sample composition and driving force

#     Parameters
#     ----------
#     x : float or array
#         Composition of minor element in bulk matrix phase
#         Use float for binary systems
#         Use array for multicomponent systems
#     T : float
#         Temperature in K
#     precPhase : str (optional)
#         Precipitate phase to consider (default is first precipitate phase in list)

#     Returns
#     -------
#     driving force - max free energy difference
#     precipitate composition set - corresponds to max driving force
#     """
#     orderTol = -1e-8
#     state_cond = {v.GE: self.gOffset, v.N: 1, v.P: 101325, v.T: T}
#     str_cond = {str(key): val for key,val in state_cond.items()}

#     #Sample precipitate phase and get driving force differences at all points -------------------------------------------------------------------
#     #Sample points of precipitate phase
#     phases, sub_models = self._setupSubModels([precPhase])
#     sample_data = self._points_cache.get(precPhase, SampledPointsCache())
#     prevT = sample_data.temperature
#     precPoints = sample_data.samples
#     orderedPoints = sample_data.ordered_samples

#     if precPoints is None or prevT != T:
#         precPoints = calculate(self.db, self.elements, phases[0],
#                                 pdens=self.sampling_pDens, model=sub_models, output='GM',
#                                 phase_records=self.phase_records, conditions=local_phase_sampling_conditions,
#                                 to_xarray=False, **str_cond)
#         if self.orderedPhase[precPhase]:
#             orderedPoints = calculate(self.db, self.elements, phases[0],
#                                         pdens=self.sampling_pDens, model=sub_models, output='OCM',
#                                         phase_records=self.phase_records, to_xarray=False, **str_cond)
#         self._points_cache[precPhase] = SampledPointsCache(temperature=T, samples=precPoints, ordered_samples=orderedPoints)

#     #For phases at fixed composition, there will only be 1 set of site fractions
#     #So we force composition and site fractions to be 2D
#     precComp = np.atleast_2d(np.squeeze(precPoints.X))
#     y = np.atleast_2d(np.squeeze(precPoints.Y))
#     gm = np.atleast_1d(np.squeeze(precPoints.GM))
#     mu = np.atleast_2d(matrix_chem_pot)

#     #Difference between the chemical potential hyperplane and the free energy of the samples
#     #   Chemical potential hyperplane is the free energy on the plane at the composition of the samples
#     #The max driving force is the same as when the chemical potentials of the two phases are parallel
#     mult = precComp * mu
#     diff = np.sum(mult, axis=1) - np.squeeze(gm)

#     #Find maximum driving force and corresponding composition -----------------------------------------------------------------------------------
#     #For phases with order/disorder transition, a filter is applied such that it will only use points that are below the disordered energy surface
#     if self.orderedPhase[precPhase]:
#         indices = np.squeeze(orderedPoints.OCM) < orderTol
#         diff = diff[indices]
#         y = y[indices]

#     dg = np.amax(diff)
#     idx = np.argmax(diff)

#     prec_cs = CompositionSet(self.phase_records[precPhase])
#     state_variables = np.array([state_cond[v.GE], state_cond[v.N], state_cond[v.P], state_cond[v.T]], dtype=np.float64)
#     y = np.array(y[idx][:prec_cs.phase_record.phase_dof], dtype=np.float64)
#     prec_cs.update(y, 1, state_variables)

#     return dg, prec_cs
