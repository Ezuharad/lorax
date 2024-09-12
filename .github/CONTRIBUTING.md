# LORAX Contributing Help
## Environment Setup
### Conda (preferred)
First install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) or another equivalent conda environment manager and add it to your PATH.

Then create and activate a conda environment:
```bash
conda create -p ./.conda
```
```bash
conda activate ./.conda
```

**This activation will be required every time you open a new shell session.**

Finally, use `environment.yml` to install the package list:
```bash
conda env update -f environment.yml
```

### Venv (supported)
First create and activate the venv:
```bash
python3 -m venv ./.venv
```
```bash
source .venv/bin/activate  # for POSIX compliant shells such as bash or zsh
.venv\Scripts\activate.bat  # for Windows cmd
.venv\Scripts\Activate.ps1  # for Windows powershell
```

**This activation will be required every time you open a new shell session.**

Finally, install the project packages with `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Local Installation (advanced)
First create and activate a new conda or venv virtual environment (see above).

Then install the package with `pip` from the repo root:
```bash
python3 -m pip install .
```

You should now be able to import `lorax` with the created virtual environment. Note this does not symlink the package to your environment, meaning any changes to the package will require a reinstall of `lorax` to your environment. This is not necessary unless testing release builds.

