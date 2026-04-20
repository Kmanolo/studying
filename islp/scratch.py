from ISLP import load_data
from ISLP.models import ModelSpec as MS


college = load_data("College")
design = MS(college.columns.drop("Apps"))

X = design.fit_transform(college)
