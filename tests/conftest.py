import pathlib
import sys

# Make `import medinova` work without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
