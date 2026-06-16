# kawin

Python implementation of the Kampmann-Wagner Numerical (KWN) model to predict precipitate nucleation and growth behavior. This package couples with pycalphad to perform thermodynamic and kinetic calculations.

0.6.0 Redesign
--------------
Goals
- From literature that cites kawin, it is clear that the way models are implemented in kawin make it difficult to modify behavior or implement new features/models. Thus, this redesign aims to make the current implemented models more modular such that physical contributions can work as a plug-and-play system. In addition, since each physical contribution to a model is separate, modifying the implement should not require major modifications to the model as a whole
- Objects will be split into either holding a state or being immutable (though this can be subject to interpretation where behavior is immutable although internal states that do not change behavior such as caching may be allowed)
- Classes will be smaller, acting more as dataclasses than full-on singletons. Then any functions that compute a term will take in the class rather than being part of the class
- Work towards following PEP 8 guidelines. My coding style has changed since the past few years this project as started and now the codebase is a hodge-podge of different styles mixed together. Hopefully, the changes in this re-design effort will help create a more consistent style that's also compatible with the main upstream dependencies (i.e. pycalphad, ESPEI)

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
