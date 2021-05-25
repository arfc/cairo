# CAIRO
CAIRO : Computerized Artificially Intelligent Reactor Operator.  

### Requirements and Setup
1. Clone the ``cairo`` repository to your local machine.
2. Clone ``pyESN`` inside the ``cairo`` directory.
3. Install [Sphinx](https://www.sphinx-doc.org/en/master/) and the 
[Cloud](https://cloud-sptheme.readthedocs.io/en/latest/install.html) theme


With SSH
```bash
git clone git@github.com:arfc/cairo.git
cd cairo
git clone git@github.com:cknd/pyESN.git
```


### Contributing
See the `contributing.md` document for instructions on how to contribute to `CAIRO`.

### Continuous Integration
This repository uses continuous integration with CircleCI to automatically run tests.

### Building Documentation
To build, all you have to do is follow the requirements and set up instructions
and then run the commands
```
cd cairo/docs
make html
open _build/html/index.html
```
From there, you can navigate the documentation by interfacing with the html 
rendering.

### Testing
This repository contains various tests for each module, and makes use of 
[pytest](https://docs.pytest.org/en/stable/#) to run them. Upon downloading 
CAIRO, you should make sure that all of the tests pass. See the 
`contributing.md` document for instructions on how to run the tests.

Checkout pytest's page on 
[Usage and Invocations](https://docs.pytest.org/en/stable/usage.html#usage-and-invocations) 
for information on variations you can use in addition to the instructions in 
`contributing.md`.
