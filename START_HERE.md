# Start Here: Open BO Forge On Your Computer

This guide takes you from a downloaded repository to the local browser interface.
You do not need Git or programming experience. Keep the downloaded folder: it
contains the example campaigns, and you will return to it to reopen the app.

## 1. Prepare Your Computer

You need an internet connection for installation, Python to run BO Forge, a
terminal to enter commands, and a browser to display the app. A terminal is a
window where you type a command and press Enter. Copy commands without the
surrounding code fences; run one line at a time and wait for it to finish.

Use **Python 3.12**. The repository CI targets Linux with Python 3.11/3.12 and
macOS with Python 3.12. This beginner walkthrough is runtime-checked on macOS
with Python 3.12; its Linux instructions are not separately runtime-verified
here. Windows is not in the tested platform matrix; the Windows commands below
are a best-effort starting point, not a claim of supported Windows workflows.

**macOS:** open Terminal from Applications > Utilities. Follow the
[official Python macOS installation instructions](https://docs.python.org/3.12/using/mac.html)
if Python 3.12 is missing. Do not replace Apple's system Python.

**Linux:** open your distribution's Terminal application. Follow the
[official Python Unix guidance](https://docs.python.org/3.12/using/unix.html) and
your distribution's package instructions for Python 3.12 and its `venv` module.

Check in either terminal:

```bash
python3.12 --version
```

Expected: `Python 3.12.x` (the last number may vary).

**Windows, not runtime-verified:** open **Command Prompt**, not PowerShell, for
the commands in this guide. Follow the
[official Python Windows installation instructions](https://docs.python.org/3.12/using/windows.html),
including the Python launcher, then check:

```bat
py -3.12 --version
```

## 2. Download BO Forge

Open the [BO Forge repository](https://github.com/angzeli/bo-forge). Click
**Code > Download ZIP**, then extract the ZIP using your file manager. Work in
the extracted folder, usually named `bo-forge-main`, not inside the ZIP viewer.
Move it somewhere you intend to keep before creating the environment.

Git is optional. If you already use it, this is an alternative to the ZIP:

```bash
git clone https://github.com/angzeli/bo-forge.git
```

The cloned folder is named `bo-forge`, rather than `bo-forge-main`.

## 3. Open A Terminal In That Folder

Locate the extracted folder in Finder or your file manager. `cd` means
"change directory". Replace the example path with your actual folder path.
Keep the quotes, especially when a path contains spaces.

**macOS/Linux** (a typical Downloads location):

```bash
cd "$HOME/Downloads/bo-forge-main"
ls pyproject.toml
```

On macOS you can also type `cd `, drag the extracted folder from Finder into
Terminal, and press Enter.

**Windows Command Prompt, not runtime-verified:**

```bat
cd /d "%USERPROFILE%\Downloads\bo-forge-main"
dir pyproject.toml
```

Expected: a file named `pyproject.toml`. If it is not found, stop and locate the
folder containing that file before installing. Some extractors create an extra
outer folder. All remaining commands assume you are in this project folder.

## 4. Create And Activate An Isolated Environment

An environment keeps BO Forge's dependencies separate from other projects.
These commands create a folder named `.venv`, then select its Python for this
terminal session. Create it once, but activate it in each new terminal.

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows Command Prompt, not runtime-verified:**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Your prompt will usually begin with `(.venv)`. Confirm which Python is selected:

```bash
python -c "import sys; print(sys.executable)"
```

Expected: a path inside this project's `.venv` folder. Do not continue with a
different project's or system Python.

## 5. Install This Download With Streamlit

Use the same commands on all three platforms, after activation:

```bash
python -m pip install ".[app]"
python -m pip check
python -m bo_forge --version
python -m bo_forge doctor
```

The dot in `".[app]"` means **this downloaded project**. The `app` extra includes
Streamlit, which displays the local interface. Installation can take several
minutes and downloads scientific libraries; wait for the terminal prompt to
return. Do not use a package-index BO Forge release in place of this command.

Expected: `No broken requirements found.`, `bo-forge 3.1.3` for this source
version, and `Status: OK` from the doctor. Stop and address errors before
starting experimental work. A different downloaded revision may have a
different version; check its README heading.

## 6. Open The Local Interface

```bash
python -m bo_forge_app
```

Keep this terminal open while using BO Forge. It runs the app. The launcher
prints the local address, normally **http://127.0.0.1:8501**. A browser may open
automatically; if it does not, type that address into your browser's address bar.
`bo-forge-app` is the equivalent launcher in the activated environment.

## 7. Confirm Success And Start A Campaign

On first launch, expect the **BO Forge** heading, **Load Existing** and
**Create Campaign** choices, and **Nothing loaded yet.** This is normal, not an
installation error. Those controls confirm that the interface has loaded, not
just that a web server is listening.

After you load or create a campaign, three work areas appear: **Campaign** for
inspecting campaign files, **Run** for suggestions and observations, and
**Analyze** for reports and plots. Loading and creation stay available above them.

Continue with the [app-created campaign tutorial](docs/09_APP_CREATED_CAMPAIGN_TUTORIAL.md).
New campaigns keep a YAML configuration, CSV experiment log, and provenance
manifest together. The [provenance reference](docs/PROVENANCE.md) explains this
history tracking; it is separate from these installation steps.

## 8. Stop And Reopen

In the terminal running BO Forge, press **Ctrl+C** to stop it. Closing only the
browser tab does not stop the app.

Next time, open a terminal and return to the same downloaded folder.

**macOS/Linux:**

```bash
cd "$HOME/Downloads/bo-forge-main"
source .venv/bin/activate
python -m bo_forge_app
```

**Windows Command Prompt, not runtime-verified:**

```bat
cd /d "%USERPROFILE%\Downloads\bo-forge-main"
.venv\Scripts\activate.bat
python -m bo_forge_app
```

Use your actual folder path again. Installation is normally a one-time step;
you do not need to reinstall on each launch. Keep your campaign files and
provenance archives when backing up or moving your work.

## 9. Troubleshoot

| Problem | What to do |
| --- | --- |
| Python command not found | Install Python 3.12 using step 1, reopen the terminal, and repeat the version check. |
| `pyproject.toml` not found or pip says this is not a project | Return to step 3; the terminal must be in the extracted folder containing `pyproject.toml`. |
| Wrong interpreter or `No module named bo_forge_app` | Activate this folder's environment and check `sys.executable` using step 4. |
| Streamlit is missing | In the activated environment and correct project folder, rerun `python -m pip install ".[app]"`, then `python -m pip check`. |
| Installation fails | Check Python 3.12, internet connectivity, disk space, and the first error. On Linux, a missing `venv`/`ensurepip` needs the distribution's Python venv package. Save the error for help; do not bypass certificate checks or system protections. |
| Browser does not open | Keep the terminal running and manually open the address printed by the launcher. |
| Port 8501 is occupied | Stop your other BO Forge terminal with Ctrl+C, or use the alternate-port command below. Do not terminate an unfamiliar process. |

```bash
python -m bo_forge_app --port 8502
```

Then open **http://127.0.0.1:8502**. Always install with `python -m pip` from the
selected environment; never use administrator-level pip installation. If you
need help, share the error and Python/BO Forge version, not private campaign data.

## 10. Keep Access Local

The browser interface runs on your computer, not on GitHub. The default address
is loopback-only: it is meant for your own computer. There is no built-in
authentication. Keep the default host; this beginner path does not expose the
app to a network.

For a separate advanced topic, read [deployment and safety guidance](docs/STREAMLIT_DEPLOYMENT.md).
For other installation choices, see [Installation](docs/INSTALLATION.md).
