import copy
from typing import Protocol
from functools import wraps

import numpy as np
from tinydb import where

from pycalphad import Workspace, Model, Database, calculate, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.utils import extract_parameters
from pycalphad.core.solver import Solver, SolverResult

import kawin.thermo.mobility as mob_funcs

setattr(v, 'GE', v.IndependentPotential('GE'))

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

def local_equilibrium(
        dbf: Database, comps: list[str], phases: list[str], conds: dict[v.StateVariable, float],
        models: dict[str, Model], phase_record_factory: PhaseRecordFactory,
        composition_sets: list[CompositionSet]=None, pdens: int=10
        ) -> tuple[SolverResult, list[CompositionSet]]:
    '''
    Local equilibrium calculation

    Global eq samples all possible phase space to acheive a global minimization
    Local eq will minimize using the supplied composition sets as the starting points
      In addition, this function specifically forces 1 composition set per phase

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
    SolverResult, list of composition sets
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
                               pdens=pdens, model=models, phase_records=phase_record_factory, conditions=local_phase_conds)
            idx_p = np.argmin(calc_p.GM.values.squeeze())
            compset = CompositionSet(phase_record_factory[phases[0]])
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
                                pdens=pdens, model=models, phase_records=phase_record_factory)
                idx_p = np.argmin(calc_p.GM.values.squeeze())
                compset = CompositionSet(phase_record_factory[phase])
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
    matrix : list
        Matrix phases
    precipitates : list
        Precipitate phases
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
        ordered_phase = self.dbf.phases[new_phase].model_hints['ordered_phase']
        self.dbf.phases[ordered_phase].model_hints['disordered_phase'] = new_phase

        #Copy database parameters with new name
        param_query = where('phase_name') == phase
        params = self.dbf.search(param_query)
        for p in params:
            #We have to create a new dictionary since p is a TinyDB.Document
            new_p = {}
            for entry in p:
                new_p[entry] = p[entry]
            new_p['phase_name'] = new_phase
            self.dbf._parameters.insert(new_p)

def _get_phase(phases: list[str], phase: str=None) -> str:
    """
    Fallback function to grab first phase from list if not supplied
    """
    return phases[0] if phase is None else phase

def _process_conditions(conditions: dict[v.StateVariable, float | list[float]]) -> tuple[dict[v.StateVariable, list[float]], int]:
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

def _full_comp_from_conditions(conditions: dict[v.StateVariable, float], elements: list[str]) -> np.array:
    """
    Creates composition vector from pycalphad conditions. Composition vector will be in the same order as elements
    """
    vals = np.array([conditions.get(v.X(e), 0) for e in sorted(elements)], dtype=np.float64)
    vals[vals == 0] = np.sum(vals)
    return vals

def generate_model_subset(therm: Thermodynamics, phases: str | list[str]=None) -> tuple[list[str], dict[str, Model]]:
    """
    Creates a subset of phases and models and updates the phase records accordingly

    Parameters
    ----------
    therm: Thermodynamics
    phases: list

    Returns
    -------
    phases as a list, dictionary of models for supplied phases
    """
    if isinstance(phases, str):
        phases = [phases]
    sub_models = {p: therm.models[p] for p in phases}
    # pycalphad seems to use the phase record model list to generate composition sets
    # so we need to make sure that the phase record models align with the subset of phases
    therm.phase_record_factory.models = sub_models
    return phases, sub_models

class ThermodynamicFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
            *args, **kwargs) -> any:
        ...

def enumerate_conditions(squeeze_function = lambda x: x):
    """
    Decorator to loop through pycalphad conditions and return outputs as list

    Parameters
    ----------
    squeeze_func: callable
        Takes in a list of inputs and converts them to a pretty list
    """
    def decorator(func: ThermodynamicFunction):
        @wraps(func)
        def wrapper(therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]], *args, **kwargs):
            conds, num_conds = _process_conditions(conditions)
            results = [func(therm, {key: value[i] for key, value in conds.items()}, *args, **kwargs) for i in range(num_conds)]
            return squeeze_function(results)
        return wrapper
    return decorator

def get_eq(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
        phases: str | list[str]=None) -> Workspace:
    """
    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    phases: list

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

def get_local_eq(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        phases: str | list[str]=None, composition_sets: list[CompositionSet]=None) -> tuple[SolverResult, list[CompositionSet]]:
    """
    Calculates local equilibrium at specified x, T, gExtra

    Local equilibrium is defined as all phases have a single composition set
        Miscibility gaps are ignored
        In the case of a single phase with a miscibility gap, this returns the composition
        set at x

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    phases: list
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


def _get_cs_eq(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
        matrix_phase: str, prec_phase: str, cache: dict[str, any]) -> tuple[np.array, CompositionSet, CompositionSet]:
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
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

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
        #conditions = _process_conditions(conditions)
        phases, sub_models = generate_model_subset(therm, [matrix_phase, prec_phase])
        result, composition_sets = local_equilibrium(
            therm.dbf, therm.elements, phases,
            conditions, sub_models, therm.phase_record_factory, composition_sets,
            therm.local_pdens
            )
        cs_matrix, cs_precip, miscibility_gap = _process_composition_sets(composition_sets)
        return result.chemical_potentials, cs_matrix, cs_precip, miscibility_gap

    # If no cache exists, then compute global equilibrium, else, update cached composition sets
    cache_id = f'{matrix_phase}_{prec_phase}'
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