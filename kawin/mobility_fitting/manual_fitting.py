import numpy as np
from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from symengine import Piecewise, And, Symbol
from tinydb import where
from kawin.thermo.LocalEquilibrium import local_equilibrium
from espei.datasets import load_datasets, recursive_glob
from kawin.mobility_fitting.utils import _vname, find_last_variable, MobilityTerm

class EquilibriumSiteFractionGenerator:
    '''
    Grabs site fraction values from local equilibrium calculations
    '''
    def __init__(self, database : Database, phase : str):
        self.db = database
        self.phase = phase
        self.models = {}
        self.phase_records = {}
        self.constituents = {}

        self.full_constituents = [c for cons in self.db.phases[self.phase].constituents for c in sorted(list(cons))]
        self._conditions_override = {v.N: 1, v.GE: 0}

    def set_override_condition(self, variable, value):
        self._conditions_override[variable] = value

    def remove_override_condition(self, variable):
        self._conditions_override.pop(variable)

    def _generate_comps_key(self, components):
        comps = sorted(components)
        return frozenset(comps), comps

    def _generate_phase_records(self, components):
        # Caches models, phase_records and constituents based off active components
        active_comps, comps = self._generate_comps_key(components)
        if active_comps not in self.models:
            self.models[active_comps] = {self.phase: Model(self.db, comps, self.phase)}
            self.phase_records[active_comps] = PhaseRecordFactory(self.db, comps, {v.T, v.P, v.N, v.GE}, self.models[active_comps])
            self.constituents[active_comps] = [c for cons in self.models[active_comps][self.phase].constituents for c in sorted(list(cons))]

    def __call__(self, components, conditions : dict[v.StateVariable: float]) -> dict[v.Species: float]:
        # Get phase records from active components
        active_comps, comps = self._generate_comps_key(components)
        self._generate_phase_records(components)

        # Override any conditions
        for oc in self._conditions_override:
            conditions[oc] = self._conditions_override[oc]

        # Compute local equilibrium (to avoid miscibility gaps)
        # Store site fractions
        #    The first 4 items of CompositionSet.dof refers to v.GE, v.N, v.P and v.T
        #    The order of the site fractions in CompositionSet.dof should correspond to the order in the pycalphad model
        results, comp_sets = local_equilibrium(self.db, comps, [self.phase], conditions, self.models[active_comps], self.phase_records[active_comps])
        sfg = {c:val for c,val in zip(self.constituents[active_comps], comp_sets[0].dof[4:])}
        for c in self.full_constituents:
            sfg[c] = sfg.get(c,0)
        return sfg
    
class SiteFractionGenerator:
    def create_site_fractions(self, composition):
        return NotImplementedError()

    def __call__(self, components, conditions):
        comps_no_va = list(set(components) - set(['VA']))
        composition = {c:1 for c in comps_no_va}
        for key,val in conditions.items():
            if type(key) == v.MoleFraction:
                for c in composition:
                    composition[c] -= val
                composition[key.name[2:]] = val

        return self.create_site_fractions(composition)

def least_squares_fit(A, b, p=1):
    '''
    Given site fractions and function to generate Redlich-kister terms,
    compute RK coefficients and AICC criteria
    '''
    # Fit coefficients using least squares regression
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # AICC criteria
    k = len(A[0])
    n = len(A)
    b_pred = np.matmul(A, x)
    rss = np.sum((b_pred-b)**2)
    pk = p*k
    aic = 2*pk + n*np.log(rss/n)
    if pk >= n-1:
        correction = (2*pk**2 + 2*pk) / (-n + pk + 3)
    else:
        correction = (2*pk**2 + 2*pk) / (n - pk - 1)
    aicc = aic + correction
    return x, aicc

def fit(datasets, database, components, phase, diffusing_species, Q_test_models, D0_test_models, site_fraction_generator, p = 1):
    '''
    Fit mobility models to datasets

    Parameters
    ----------
    datasets : list[dict]
        Espei datasets
    species : list[v.Species]
        List of species in mobility_test_models
    components : list[str]
    phase : str
    diffusing_species : str
    mobility_test_models : list[list[MobilityTerm]]
    site_fraction_generator : function
        Takes in components and list of conditions and returns dictionary {v.SiteFraction : float}
    '''
    if type(datasets) == str:
        datasets = load_datasets(sorted(recursive_glob(datasets, '*.json')))

    components = list(set(components).union(set(['VA'])))

    fitted_mobility_model = []
    symbols = {}
    vIndex = find_last_variable(database)

    # Queries for activation energy (Q) and pre-factor (D0)
    q_query = (
        (where('components').test(lambda x: set(x).issubset(components))) & 
        (where('phases').test(lambda x: len(x) == 1 and x[0] == phase)) &
        (where('output').test(lambda x : 'TRACER_Q' in x and x.endswith(diffusing_species)))
    )
    d0_query = (
        (where('components').test(lambda x: set(x).issubset(components))) & 
        (where('phases').test(lambda x: len(x) == 1 and x[0] == phase)) &
        (where('output').test(lambda x : 'TRACER_D0' in x and x.endswith(diffusing_species)))
    )
    q_transform = lambda x : -x
    d0_transform = lambda x : 8.314*np.log(x)

    # Data_types will include:
    #    query - to search for data from datasets
    #    transform - transforms value to form to fit to (should lead to no extra multiplying factors in the database)
    #    multiplier - term to multiply by when storing in database
    #    test_models - list of models to test against
    data_types = {
        'Q': (q_query, q_transform, 1, Q_test_models),
        'D0': (d0_query, d0_transform, v.T, D0_test_models)
    }
    
    for data_key, data_val in data_types.items():
        query, transform, multiplier, test_models = data_val
        data = datasets.search(query)

        # Collect site fractions and output values from datasets
        site_fractions = []
        Y = []
        for d in data:
            conds_grid = []
            conds_key = []
            for c in d['conditions']:
                conds_grid.append(np.atleast_1d(d['conditions'][c]))
                conds_key.append(v.X(c[2:]) if c.startswith('X_') else getattr(v, c))
            conds_grid = np.meshgrid(*conds_grid)
            y_sub = transform(np.array(d['values']).flatten())

            conds_list = {key:val.flatten() for key,val in zip(conds_key, conds_grid)}

            # If non-equilibrium data, we could grab the site fractions directly
            if 'solver' in d:
                for sub_conf, sub_lat in zip(d['solver']['sublattice_configurations'], d['solver']['sublattice_occupancies']):
                    sub_index = 0
                    sf = {}
                    for species, occs in zip(sub_conf, sub_lat):
                        species, occs = np.atleast_1d(species), np.atleast_1d(occs)
                        for s, o in zip(species, occs):
                            y = v.SiteFraction(phase, sub_index, s)
                            sf[y] = o
                        sub_index += 1
                    site_fractions.append(sf)
            # If equilibrium data, we need a site_fraction_generator function to
            # convert composition to site fractions
            else:
                for i in range(len(y_sub)):
                    sf = site_fraction_generator(d['components'], {key:val[i] for key,val in conds_list.items()})
                    site_fractions.append(sf)

            Y = np.concatenate((Y, y_sub))

        # Fit models and evaluate AICC
        fitted_models = []
        aiccs = []
        for mob_model in test_models:
            X = [[mi.generate_multiplier(sf) for mi in mob_model] for sf in site_fractions]
            X = np.array(X)
            terms, aicc = least_squares_fit(X, Y, p)
            fitted_models.append(terms)
            aiccs.append(aicc)

        # Grab best model based off AICC
        index = np.argmin(aiccs)
        best_model = test_models[index]
        best_fit = fitted_models[index]

        # Store each coefficient in the best model to a list of MobilityTerms
        # MobilityTerms can be polled for equality, so if a term shows up
        # again, the coefficient will be added on rather than overriding it
        
        # We store the equations as VV00XX + VV00XX*T
        # and store the symbols VV00XX as piecewise functions separately
        for term, coef in zip(best_model, best_fit):
            if term not in fitted_mobility_model:
                fitted_mobility_model.append(MobilityTerm(term.constituent_array, term.order))
            
            combined_term = fitted_mobility_model[fitted_mobility_model.index(term)]

            var_name = _vname(vIndex)
            symbols[var_name] = Piecewise((coef, And(1.0 <= v.T, v.T < 10000)), (0, True))
            combined_term.expr += Symbol(var_name) * multiplier
            vIndex += 1

    # For each mobility term, convert to piecewise function to be compatible with database
    for f in fitted_mobility_model:
        f.expr = Piecewise((f.expr, And(1.0 <= v.T, v.T < 6000)), (0, True))

    return fitted_mobility_model, symbols




            




