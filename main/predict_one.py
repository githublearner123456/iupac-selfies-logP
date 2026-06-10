from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logp_project.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        predict_iupac = input("Enter IUPAC name to predict: ").strip()
        if not predict_iupac:
            raise SystemExit("No IUPAC name provided. Prediction cancelled.")
        sys.argv.extend(["--predict_iupac", predict_iupac])
    main()
