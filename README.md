# kawin

Python implementation of the Kampmann-Wagner Numerical (KWN) model to predict precipitate nucleation and growth behavior. This package couples with pycalphad to perform thermodynamic and kinetic calculations.

0.6.0 Redesign
--------------
Goals
- From literature that cites kawin, it is clear that the way models are implemented in kawin make it difficult to modify behavior or implement new features/models. Thus, this redesign aims to make the current implemented models more modular such that physical contributions can work as a plug-and-play system. In addition, since each physical contribution to a model is separate, modifying the implement should not require major modifications to the model as a whole
- Objects will be split into either holding a state or being immutable. Main distinction would be the `State` and `Model`, where the `State` holds a state (duh) and the `Model` is stateless. "Stateless" is more or less interpretive in that it must be logically stateless (in that all functions should behave as if the object is stateless), but physical stateless-ness is not a requirement (for example, the object can hold a cache which will cause it to have a "state" but can still behave as if it was stateless since the cache will just be to avoid repeated calculations)
- Classes will be smaller, acting more as dataclasses than full-on singletons. Then any functions that compute a term will take in the class rather than being part of the class. Some functions will be intended not to change the class but use class data to generate new data (such as thermodynamics), other functions will be intended to change internal class data (such as the PBM)
- Caches for thermodynamic functions will also be separate from the class with the idea that all functions should be computable without a cache, but faster with one. The thermodynamic classes should be as close to immutable as possible, where the model can hold the cache as an option instead
- Work towards following PEP 8 guidelines. My coding style has changed since the past few years this project as started and now the codebase is a hodge-podge of different styles mixed together. Hopefully, the changes in this re-design effort will help create a more consistent style that's also compatible with the main upstream dependencies (i.e. pycalphad, ESPEI)
- Deprecate surrogate models. I have never needed to use this since it was created. Plus the use of caching composition sets are just as fast and more accurate
- Functions that are able to take in a scalar or vector input will use decorators to implement this behavior rather than splitting the function into two parts (for example, I had something like `compute_interdiffusivity` and `_compute_interdiffusivity_single` where the latter only computed for a scalar input, but the former would call the latter to take in scalar or vector inputs)
- Differential equation based models will not be forced to abide to the base model in how the state and derivatives should be implemented. Instead, the model is allowed to hold the state and derivatives in any format as long as it can define how the derivative can be applied to the state
- Models should no long be limited to differential equations (and for explicit schemes either). We should allow explicit and implicit implementations as well as arbitrary processes (i.e. stochastic processes). Coupling multiple models should allow for each model to have their own integration scheme.

Installation
------------
Installing through pip:

```
pip install kawin
```

Development version:

```
git clone https://github.com/materialsgenomefoundation/kawin
cd kawin
pip install -e .
```

Examples
--------
Examples on Jupyter notebooks can be found on [NBViewer](https://nbviewer.org/github/materialsgenomefoundation/kawin/tree/main/examples/).

Dependencies
------------
numpy, scipy, matplotlib, pycalphad
