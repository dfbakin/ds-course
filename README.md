# Data Science Course

Two lessons and one practice assignment, as Jupyter notebooks.

| | Topic | For a student who… |
|---|---|---|
| **lesson01** | From a table to a defensible decision | is learning the ML workflow and the statistics behind it |
| **lesson02** | Doing it by hand: NumPy | knows the theory but fumbles the implementation |
| **assignment01** | Edge gateway incident model | has done both and wants to practise unaided |

---

## Setup

Five steps. Takes about two minutes.

### 1. Clone the repository

```bash
git clone https://github.com/dfbakin/ds-course.git
cd ds-course
```

### 2. Check your Python

You need **Python 3.10 or newer** (developed on 3.12).

```bash
python3 --version
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate it

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

### 5. Install the packages and the notebook kernel

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ds-course --display-name "DS Course"
```

That last line registers the kernel the notebooks expect. Skip it and Jupyter
will open the notebooks with no kernel to run them.

### Check it worked

```bash
python -c "import numpy, pandas, sklearn, matplotlib; print('ok')"
jupyter kernelspec list        # 'ds-course' should be listed
```

---

## Run a notebook

```bash
jupyter lab
```

A browser tab opens. Then pick a file:

| Open this | To do this |
|---|---|
| `lesson01/lesson01.ipynb` | Lesson 1, fully worked |
| `lesson02/lesson02.ipynb` | Lesson 2, fully worked |
| `assignment01/assignment.ipynb` | **The assignment** — fill in the gaps |
| `assignment01/solution.ipynb` | The assignment, fully worked |

If the top-right corner says *No Kernel*, click it and choose **DS Course**.

Every notebook runs top to bottom with **Run ▸ Run All Cells** and reproduces
the numbers written in its own text. Lesson 1 takes about 50 seconds; the
others under 10.

> **Not using Jupyter Lab?** `jupyter notebook` works too, and VS Code opens
> these files directly — just select the **DS Course** kernel.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: jupyter` | The venv isn't active. Re-run step 4. |
| Notebook says *No Kernel* | Run step 5's `ipykernel install` line, then reload the page. |
| `ModuleNotFoundError: numpy` | Installed outside the venv. Activate it, then re-run step 5. |
| `FileNotFoundError: ...csv` | Launch `jupyter lab` from the repo root, and keep the folder layout intact. |
| Lesson 2 can't find the data | It reads `lesson01/data/`. Don't move or rename that folder. |
| `python3: command not found` (Windows) | Use `py -3` instead of `python3`. |
