import pandas as pd
import numpy as np
from featureexpand.feature_expander import FeatureExpander
import json

# Crear un DataFrame de ejemplo
data = {
    "mintermins": ["01", "01", "11", "00", "10", "01", "11", "00", "10", "01", "01", "10", "11", "00"],
    "Cluster": ["k0", "k0", "k1", "k1", "k0", "k0", "k1", "k1", "k0", "k0", "k0", "k0", "k1", "k1"]
}
df = pd.DataFrame(data)

# Crear una instancia de FeatureExpander
expander = FeatureExpander(n_variables=1)
resul = expander.fit(df)

values = [0, 1]
print(expander.transform(values))
print(resul)