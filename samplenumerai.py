import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from featureexpand.feature_expander import FeatureExpander

# Suprimir los warnings específicos de LightGBM
warnings.filterwarnings("ignore", category=UserWarning, message="No further splits with positive gain, best gain: -inf")

print("Comenzo aca")
num = 20000

# Definir la ruta del archivo temporal
temp_file_train_target = os.path.join(tempfile.gettempdir(), f"numerai_temp_train_target_{num}.csv")
temp_file_train_features = os.path.join(tempfile.gettempdir(), f"numerai_temp_train_features_{num}.csv")

# Verificar si el archivo temporal ya existe
if os.path.exists(temp_file_train_target):
    # Si existe, leer los datos desde el archivo temporal
    train_target = pd.read_csv(temp_file_train_target)
    train_features = pd.read_csv(temp_file_train_features)
    print("Datos cargados desde el archivo temporal.")
else:
    # Si no existe, leer los 200 registros de los datos originales
    train_target = pd.read_csv(f"numerai/train_target_{num}.csv").head(200)
    train_features = pd.read_csv(f"numerai/train_features_{num}.csv").head(200)
    
    # Guardar los datos en un archivo temporal
    train_target.to_csv(temp_file_train_target, index=False)
    train_features.to_csv(temp_file_train_features, index=False)
    print("Datos guardados en el archivo temporal.")

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
# Primero, dividimos en entrenamiento (80%) y prueba (20%)
# Normalizar los valores de train_features entre 0 y 1
scaler = MinMaxScaler()
train_features = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
# Primero, dividimos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

# Luego, dividimos el conjunto de entrenamiento en entrenamiento (60%) y validación (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

expander = FeatureExpander("Tp6uxDgDHf+meUtDirx0veUq7L59a6M7IsxjRqUJZlc=",enviroment="TEST")
#print({'target': [("x1" if yx else "x0") for yx in y_train ]})
y_train_numeric = y_train.astype(float)
yy = y_train_numeric.applymap(lambda yx: "x1" if yx > 0.5 else "x0")
yy.columns = ["Cluster"]
##expander.fit(X_train, yy, feacture_selection=["feature_able_deprived_nona","feature_ablest_inflexional_egeria","feature_absorbable_hyperalgesic_mode","feature_accoutered_revolute_vexillology","feature_acetose_crackerjack_needlecraft","feature_acheulian_conserving_output","feature_acronychal_bilobate_stevenage","feature_acrylic_gallic_wine"], deep=1, response="x1")
##expander.fit(X_train, yy, feacture_selection=["feature_able_deprived_nona","feature_ablest_inflexional_egeria","feature_absorbable_hyperalgesic_mode","feature_accoutered_revolute_vexillology","feature_acetose_crackerjack_needlecraft","feature_acheulian_conserving_output","feature_acronychal_bilobate_stevenage","feature_acrylic_gallic_wine"], deep=2, response="x1")
expander.fit(X_train, yy, feacture_selection=["feature_able_deprived_nona","feature_ablest_inflexional_egeria","feature_absorbable_hyperalgesic_mode","feature_accoutered_revolute_vexillology","feature_acetose_crackerjack_needlecraft","feature_acheulian_conserving_output","feature_acronychal_bilobate_stevenage","feature_acrylic_gallic_wine","feature_adminicular_shod_levant","feature_adorable_infernal_cartesianism","feature_adorable_unsuitable_cholecystectomy","feature_advertent_deferent_kaif","feature_aeneolithic_nineteenth_whipper","feature_afoul_drainable_cateran","feature_agamid_yuletide_physiology","feature_aired_temptable_murmansk","feature_alarming_forenamed_shearing","feature_aliunde_unhaunted_coacervate","feature_altern_packaged_presbyterian","feature_analgesic_pensionary_exterior","feature_analogical_obstructed_martingale","feature_anomic_isocyclic_absinth","feature_antediluvian_establishmentarian_zebra","feature_anthracoid_tamable_consulship","feature_antidepressant_rationalistic_adaptation","feature_antigenic_perforate_chickenpox","feature_antiphlogistic_unharming_rallier","feature_appraising_chasmogamic_picrate","feature_approbatory_labroid_coracle","feature_arizonian_orphan_accord"], deep=1, response="x1")

# Configurar el modelo
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5-1,
    colsample_bytree=0.1,
    verbosity=1
)

# Entrenar el modelo
print("Entrenando...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Configurar el modelo
model_extendido = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5-1,
    colsample_bytree=0.1,
    verbosity=1
)

X_train_expanded = expander.add_features(X_train)
X_test_expanded = expander.add_features(X_test)
X_val_expanded = expander.add_features(X_val)

print(X_train_expanded)

# Entrenar el modelo
print("Entrenando...Extendido")
##, eval_set=[(X_val_expanded, y_val)]
model_extendido.fit(X_train_expanded, y_train, eval_metric='rmse')
# Predecir en el conjunto de prueba
y_pred = model_extendido.predict(X_test_expanded)

# Evaluar el modelo
mse_extendido = mean_squared_error(y_test, y_pred)
r2_extendido = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse} vs Extendido {mse_extendido}")
print(f"Coeficiente de Determinación (R²): {r2} vs Extendido {r2_extendido}")