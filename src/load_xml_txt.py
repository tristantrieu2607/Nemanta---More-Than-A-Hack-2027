import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kloppy import metrica
from games import asset_paths

# load atd
paths = asset_paths()
dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
df = dataset.to_df()
print(df.shape)
print(df.columns[:10])

# load smart tagging
from kloppy import sportscode
events = sportscode.load(paths["smart_tagging"]).to_df()
print(events.head())