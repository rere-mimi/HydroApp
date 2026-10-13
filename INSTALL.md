# Install HydroBridge

HydroBridge is a map in your browser. It stays on your computer. You do not need an account.

## Windows

1. Install [64-bit Python 3.12](https://www.python.org/downloads/release/python-31210/). On the first installer screen, tick **Add python.exe to PATH**.
2. Get a copy of this folder. Clone it, or unzip a copy. Do not copy someone else's `.venv` folder.
3. Double-click `run.bat`.
4. The first run installs the Python packages. That can take a few minutes and needs the internet.
5. A browser opens at http://127.0.0.1:5050. Leave the black window open while you use the app. Close that window to stop it.

Next time, double-click `run.bat` again.

If setup fails on NumPy or Matplotlib, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) and run `run.bat` again.

## Mac or Linux

Python 3.11 or 3.12 is required. In the project folder:

```bash
chmod +x run.sh
./run.sh
```

Then open http://127.0.0.1:5050. Leave the terminal open.

On Ubuntu, if the install cannot find GDAL, run this first:

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev libspatialindex-dev
```

## What you need while using it

An internet connection. Clip DEM and Run screening download the LINZ 1 m elevation sheets that cover the orange rectangle. Those files are saved in the project folder under `outputs` and reused next time.
