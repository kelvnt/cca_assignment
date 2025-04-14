# CCA Assignment

This notebook contains the steps to run the CCA Assignment solver.

## Dataset

There are 2 files that will be required for this to work.
* data/student_list.csv
* data/cca_vacancies.csv

### student_list
`student_list` is a table of student info consisting of 6 columns:
* student_id: unique identifier for the student, this can be the student's name if the names are all unique (no duplicates)
* date: date and time at which the student submitted his/her CCA choices (will be used to select the most recent submission and removed duplicated earlier submissions). Please ensure that the format are consistent, e.g. DD-MM-YYYY HH:MM
* choice_1: CCA choice 1, must correspond exactly to a CCA in the `cca_vacancies` file
* choice_2: CCA choice 2, must correspond exactly to a CCA in the `cca_vacancies` file
* choice_3: CCA choice 3, must correspond exactly to a CCA in the `cca_vacancies` file
* direct_assignment: special case for students that are directly assigned a CCA, must correspond exactly to a CCA in the `cca_vacancies` file


### cca_vacancies
CCA vacancies is a table of CCA details consisting of 4 columns:
* cca: CCA id, must correspond exactly to the choices that choices that are submitted by the students in `student_list`
* filled_spots: number of spots that are filled
* vacant_spots: number of vacant spots
* total_spots: total number of spots

## Project structure
```bash
.
├── config                      
├── data            
├── docs                            # documentation for your project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
```

## Set up the environment
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
```bash
brew install uv
uv python install 3.12
```
2. Set up the environment:
```bash
make activate
make install
```

## Install new packages
To install new PyPI packages, run:
```bash
uv add <package-name>
```