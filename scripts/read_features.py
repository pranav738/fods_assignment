(uv run python - <<'PY'
          import pandas as pd
          from pathlib import Path
          file = Path('/home/pranav/projects/fods_assignment/bangalore 2022-01-01 to 2024-08-09 merged.csv')
          df = pd.read_csv(file, nrows=0)
          print('Columns:', ','.join(df.columns))
          PY, impact: medium)

