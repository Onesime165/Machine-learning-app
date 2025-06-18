import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import boxcox
import warnings

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="🛠️ ML Interactif",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark and Technological Theme from First Code ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;500&display=swap');

    /* Main app styling */
    body {
        font-family: 'Roboto', sans-serif;
        color: #e0e6ed !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
        border-right: 2px solid #333333;
        box-shadow: 5px 0 15px rgba(0, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #b0c4de !important;
        font-family: 'Roboto', sans-serif;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }

    /* Main title */
    h1 {
        color: #00ffff;
        text-align: center;
        font-family: 'Arial', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        padding-top: 1.5rem;
    }

    /* Sub-headers */
    h2, h3, h4 {
        color: #00ffff;
        font-family: 'Arial', sans-serif;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
    }

    /* Selectbox and multiselect styling */
    .stSelectbox, .stMultiSelect {
        background-color: rgba(15, 15, 35, 0.8) !important;
        border: 5px solid #333333;
        border-radius: 5px;
        padding: 10px;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    /* DataFrame styling */
    .stDataFrame {
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    /* Uploader styling in sidebar */
    [data-testid="stFileUploader"] {
        border: 2px dashed #333333;
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 0.5rem;
        background: rgba(15, 15, 35, 0.8);
        border: 1px solid #333333;
        color: #e0e6ed !important;
    }

    /* Uploaded file styling */
    .uploaded-file {
        color: #00ff88 !important;
        font-weight: bold;
    }

    /* Footer styling */
    .footer {
        font-size: 0.8rem;
        color: #b0c4de;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #333333;
        border-radius: 5px;
    }

    /* Author info box */
    .author-info {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 2px solid #333333;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.title("🛠️ Machine Learning Interactif")
st.markdown("<p style='text-align: center; color: #b0c4de; font-family: Roboto, sans-serif;'>Une plateforme interactive pour l'entraînement et l'évaluation de modèles de Machine Learning.</p>", unsafe_allow_html=True)

# Chargement du fichier CSV ou Excel
uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier CSV ou Excel",
    type=["csv", "xlsx"],
    help="Formats supportés : CSV, Excel (.xlsx)"
)

df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.sidebar.subheader("⚙️ Options CSV")
            header = st.sidebar.checkbox("Première ligne comme en-tête", value=True)
            sep = st.sidebar.selectbox("Séparateur de colonnes", [",", ";", "\t"], index=1)
            dec = st.sidebar.selectbox("Séparateur décimal", [".", ","], index=0)
            df = pd.read_csv(
                uploaded_file,
                sep=sep,
                decimal=dec,
                header=0 if header else None,
                engine='python'
            )
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"Fichier chargé: **{uploaded_file.name}**")
        st.sidebar.write(f"🔍 **{len(df)}** observations, **{len(df.columns)}** variables")
        st.sidebar.markdown(f"<p class='uploaded-file'>Fichier chargé: {uploaded_file.name}</p>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Erreur lors de la lecture : {e}")

# Si données chargées
if df is not None and not df.empty:
    # Sélection de la variable cible
    target_var = st.sidebar.selectbox("Sélectionnez la variable cible (Y)", options=df.columns.tolist())

    if not np.issubdtype(df[target_var].dtype, np.number):
        st.warning("La variable cible n'est pas numérique. Tentative de conversion...")
        df[target_var] = pd.to_numeric(df[target_var], errors='coerce')

    df = df.dropna(subset=[target_var])
    if len(df) == 0:
        st.error("Toutes les lignes ont été supprimées après suppression des NA dans la variable cible.")
        st.stop()

    # --- Transformation ---
    st.sidebar.subheader("🧮 Étape 1 : Transformation des données")
    transformations_options = ["Aucune", "BoxCox", "Centrer", "Réduire", "MinMaxScaler", "Standardisation", "Binarisation", "ACP"]
    transformations = st.sidebar.multiselect(
        "Choisissez jusqu'à 2 transformations (appliquées dans l'ordre de sélection si pertinent)",
        transformations_options,
        default=["Aucune"], max_selections=2
    )

    # Generic code snippets for transformations
    transformation_code_map = {
        "BoxCox": """
# Y: target variable (pd.Series from pandas)
# from scipy.stats import boxcox
if (y > 0).all(): # BoxCox requires positive values
    y_transformed, lambda_val = boxcox(y)
    y = pd.Series(y_transformed, index=y.index, name=y.name)
    # print(f"BoxCox appliqué à Y (lambda = {lambda_val:.2f})")
else:
    # print("BoxCox non appliqué : Y contient des valeurs non positives.")
    pass # Y non transformée
""",
        "Centrer": """
# X: features (pd.DataFrame from pandas)
# from sklearn.preprocessing import StandardScaler
scaler_center = StandardScaler(with_std=False) # Only subtracts mean
X_centered = scaler_center.fit_transform(X)
X = pd.DataFrame(X_centered, columns=X.columns, index=X.index)
# print("Centrage appliqué à X")
""",
        "Réduire": """
# X: features (pd.DataFrame from pandas)
# from sklearn.preprocessing import StandardScaler
scaler_reduce = StandardScaler(with_mean=False) # Only divides by std dev
X_reduced = scaler_reduce.fit_transform(X)
X = pd.DataFrame(X_reduced, columns=X.columns, index=X.index)
# print("Réduction appliquée à X (division par l'écart-type)")
""",
        "MinMaxScaler": """
# X: features (pd.DataFrame from pandas)
# from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler() # Scales to a range [0, 1]
X_minmax_scaled = scaler_minmax.fit_transform(X)
X = pd.DataFrame(X_minmax_scaled, columns=X.columns, index=X.index)
# print("MinMaxScaler appliqué à X")
""",
        "Standardisation": """
# X: features (pd.DataFrame from pandas)
# from sklearn.preprocessing import StandardScaler
scaler_standard = StandardScaler() # Centers to mean 0, scales to std dev 1
X_standardized = scaler_standard.fit_transform(X)
X = pd.DataFrame(X_standardized, columns=X.columns, index=X.index)
# print("Standardisation appliquée à X")
""",
        "Binarisation": """
# X: features (pd.DataFrame from pandas)
# Binarizes features based on their median value.
# Values > median become 1, otherwise 0.
median_values = X.median()
X_binarized = (X > median_values).astype(int)
X = X_binarized
# print("Binarisation appliquée à X (seuil = médiane de chaque colonne)")
""",
        "ACP": """
# X: features (pd.DataFrame from pandas)
# from sklearn.decomposition import PCA
if X.shape[1] >= 2: # PCA requires at least 2 features
    # Example: reduce to min of current features or 10 components
    n_components_pca = min(X.shape[1], 10) 
    pca_model = PCA(n_components=n_components_pca)
    X_pca_transformed = pca_model.fit_transform(X)
    X = pd.DataFrame(X_pca_transformed, columns=[f"PC{i+1}" for i in range(X_pca_transformed.shape[1])], index=X.index)
    # print(f"ACP appliquée à X ({X.shape[1]} composantes principales)")
else:
    # print("ACP non appliquée : moins de 2 variables numériques.")
    pass # X non transformée
"""
    }

    if st.sidebar.button("👁️ Voir le code des transformations", key="show_transfo_code_btn"):
        st.session_state.show_transfo_code_panel = not st.session_state.get("show_transfo_code_panel", False)

    # Initialize X and y
    X_original = df.drop(columns=[target_var]).select_dtypes(include=[np.number])
    y_original = df[target_var]

    # Work with copies for transformation
    X = X_original.copy()
    y = y_original.copy()
    
    transfo_summary = []
    selected_transformations_for_code_display = [t for t in transformations if t != "Aucune"]

    # Apply transformations in selected order
    for transfo_name in transformations:
        if transfo_name == "Aucune":
            if len(transformations) == 1: # Only if "Aucune" is the sole selection
                 transfo_summary.append("Aucune transformation appliquée.")
            continue

        if transfo_name == "BoxCox":
            if (y > 0).all():
                y_transf, lambda_box_val = boxcox(y)
                y = pd.Series(y_transf, name=target_var, index=y.index)
                transfo_summary.append(f"BoxCox appliqué à Y (lambda = {lambda_box_val:.2f})")
            else:
                transfo_summary.append("BoxCox échoué : Y contient des valeurs non positives. Y non transformée.")
        elif transfo_name == "Centrer":
            if not X.empty:
                scaler = StandardScaler(with_std=False)
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                transfo_summary.append("Centrage appliqué aux features X.")
            else:
                transfo_summary.append("Centrage non appliqué : x est vide.")
        elif transfo_name == "Réduire":
            if not X.empty:
                scaler = StandardScaler(with_mean=False)
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                transfo_summary.append("Réduction appliquée aux features X.")
            else:
                transfo_summary.append("Réduction non appliquée : x est vide.")
        elif transfo_name == "MinMaxScaler":
            if not X.empty:
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                transfo_summary.append("MinMaxScaler appliqué aux features X.")
            else:
                transfo_summary.append("MinMaxScaler non appliqué : x est vide.")
        elif transfo_name == "Standardisation":
            if not X.empty:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                transfo_summary.append("Standardisation appliquée aux features X.")
            else:
                transfo_summary.append("Standardisation non appliquée : x est vide.")
        elif transfo_name == "Binarisation":
            if not X.empty:
                median_vals = X.median()
                X = (X > median_vals).astype(int)
                transfo_summary.append("Binarisation appliquée aux features X (seuil = médiane).")
            else:
                transfo_summary.append("Binarisation non appliquée : x est vide.")
        elif transfo_name == "ACP":
            if not X.empty and X.shape[1] >= 2:
                pca_n_components = min(X.shape[1], 10) # Limit components for stability
                pca = PCA(n_components=pca_n_components)
                X_pca = pca.fit_transform(X)
                X = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])], index=X.index)
                transfo_summary.append(f"ACP appliquée aux features X ({X.shape[1]} composantes).")
            elif X.empty:
                transfo_summary.append("ACP non appliquée : x est vide.")
            else: # X.shape[1] < 2
                transfo_summary.append(f"ACP échouée : Moins de 2 features numériques pour l'ACP ({X.shape[1]} disponibles). X non transformée par ACP.")
    
    if not transfo_summary and "Aucune" in transformations : # Ensure "Aucune" message if it was the only choice
        if not any(t != "Aucune" for t in transformations):
             transfo_summary.append("Aucune transformation sélectionnée.")


    df_transformed = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    st.sidebar.write("✅ Transformations appliquées (résumé) :")
    if transfo_summary:
        for line in transfo_summary:
            st.sidebar.write(f"- {line}")
    else:
         st.sidebar.write("- Aucune transformation valide n'a été appliquée ou sélectionnée.")


    if st.session_state.get("show_transfo_code_panel", False):
        with st.sidebar.expander("Code des transformations sélectionnées", expanded=True):
            if not selected_transformations_for_code_display:
                st.info("Aucune transformation sélectionnée (hors 'Aucune') pour afficher le code.")
            else:
                displayed_code_count = 0
                for transfo_name_code in selected_transformations_for_code_display:
                    if transfo_name_code in transformation_code_map:
                        st.markdown(f"#### Code pour : {transfo_name_code}")
                        st.code(transformation_code_map[transfo_name_code], language="python")
                        st.markdown("---")
                        displayed_code_count +=1
                if displayed_code_count == 0:
                    st.info("Aucun code générique disponible pour les transformations sélectionnées.")
    
    if X.empty:
        st.error("Après transformations, il n'y a plus de variables explicatives (X). Vérifiez vos données et sélections de transformations.")
        st.stop()


    # --- Rééchantillonnage ---
    st.sidebar.subheader("🔁 Étape 2 : Méthode de Rééchantillonnage")
    resampling_method_options = [
        "Test/Train (70/30)", "Bootstrap", "K-fold", "Repeated K-fold", "Leave One Out"
    ]
    resampling_method = st.sidebar.selectbox("Méthode", resampling_method_options)

    # Button to show model training code
    if st.sidebar.button("👁️ Voir le code d'entraînement", key="show_model_code_btn"):
        st.session_state.show_model_code_panel = not st.session_state.get("show_model_code_panel", False)

    model_training_code_map = {
        "Test/Train (70/30)": """
# X, y: full dataset (features and target) after transformations
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# models = { "NomModèle": ModelClass(), ... } # Dictionnaire de modèles
# results_list = []
# predictions_on_test_set = {}

for model_name, model_instance in models.items():
    try:
        model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)
        
        rmse_val = np.sqrt(mean_squared_error(y_test, predictions))
        mae_val = mean_absolute_error(y_test, predictions)
        r2_val = r2_score(y_test, predictions)
        
        # results_list.append({"Modèle": model_name, "RMSE": rmse_val, ...})
        # predictions_on_test_set[model_name] = predictions
    except Exception as e:
        # print(f"Erreur avec {model_name}: {e}")
        # results_list.append({"Modèle": model_name, "RMSE": "Échec", ...})
        pass
""",
        "K-fold_base_template": """
# X, y: full dataset (features and target) after transformations
# from sklearn.model_selection import {KFold_CLASS} # e.g., KFold, RepeatedKFold, LeaveOneOut
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np

# models = { "NomModèle": ModelClass(), ... }
# results_list = []
# predictions_on_test_set = {} # For final display, usually on a hold-out set

# # Initial split for final evaluation/prediction display (optional but good practice)
# # X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.3, random_state=42)


for model_name, model_instance in models.items():
    try:
        fold_rmse_scores = []
        fold_mae_scores = []
        fold_r2_scores = []

        # cv_splitter = {KFold_CLASS_INITIALIZATION} 
        # # e.g., KFold(n_splits=5, shuffle=True, random_state=42) for Bootstrap-like CV
        # # e.g., KFold(n_splits=5) for K-Fold
        # # e.g., RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        # # e.g., LeaveOneOut()

        for train_indices, test_indices in cv_splitter.split(X):
            X_fold_train, X_fold_test = X.iloc[train_indices], X.iloc[test_indices]
            y_fold_train, y_fold_test = y.iloc[train_indices], y.iloc[test_indices]
            
            model_instance.fit(X_fold_train, y_fold_train)
            fold_predictions = model_instance.predict(X_fold_test)
            
            fold_rmse_scores.append(np.sqrt(mean_squared_error(y_fold_test, fold_predictions)))
            fold_mae_scores.append(mean_absolute_error(y_fold_test, fold_predictions))
            fold_r2_scores.append(r2_score(y_fold_test, fold_predictions))

        avg_rmse = np.mean(fold_rmse_scores)
        avg_mae = np.mean(fold_mae_scores)
        avg_r2 = np.mean(fold_r2_scores)
        # results_list.append({"Modèle": model_name, "RMSE": avg_rmse, ...})

        # For consistent prediction display (on X_test_final, if defined)
        # model_instance.fit(X_train_final, y_train_final) # Re-train on a larger portion or specific train set
        # final_predictions = model_instance.predict(X_test_final)
        # predictions_on_test_set[model_name] = final_predictions
        
    except Exception as e:
        # print(f"Erreur avec {model_name}: {e}")
        # results_list.append({"Modèle": model_name, "RMSE": "Échec", ...})
        pass
"""
    }
    model_training_code_map["Bootstrap"] = model_training_code_map["K-fold_base_template"].replace(
        "{KFold_CLASS}", "KFold"
    ).replace(
        "{KFold_CLASS_INITIALIZATION}", "KFold(n_splits=5, shuffle=True, random_state=42) # Simulates Bootstrap with KFold"
    )
    model_training_code_map["K-fold"] = model_training_code_map["K-fold_base_template"].replace(
        "{KFold_CLASS}", "KFold"
    ).replace(
        "{KFold_CLASS_INITIALIZATION}", "KFold(n_splits=5)"
    )
    model_training_code_map["Repeated K-fold"] = model_training_code_map["K-fold_base_template"].replace(
        "{KFold_CLASS}", "RepeatedKFold"
    ).replace(
        "{KFold_CLASS_INITIALIZATION}", "RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)"
    )
    model_training_code_map["Leave One Out"] = model_training_code_map["K-fold_base_template"].replace(
        "{KFold_CLASS}", "LeaveOneOut"
    ).replace(
        "{KFold_CLASS_INITIALIZATION}", "LeaveOneOut()"
    )

    if st.session_state.get("show_model_code_panel", False):
        with st.sidebar.expander("Code d'entraînement des modèles (structure générique)", expanded=True):
            selected_resampling_code_snippet = model_training_code_map.get(resampling_method, "Code non disponible pour cette méthode.")
            st.markdown(f"#### Code pour : {resampling_method}")
            st.code(selected_resampling_code_snippet, language="python")
            st.markdown(
                "**Note:** Ceci est un extrait de code générique illustrant la structure. "
                "Les modèles spécifiques (`LinearRegression`, `RandomForestRegressor`, etc.) sont définis dans un dictionnaire "
                "et itérés. Les variables `X`, `y` (et `X_train`, `y_train`, `X_test`, `y_test` pour la méthode Test/Train) "
                "sont supposées être définies et prétraitées (transformées)."
            )
            if resampling_method != "Test/Train (70/30)":
                 st.markdown(
                    "Pour l'affichage cohérent des prédictions finales dans l'interface, les modèles sont "
                    "ré-entraînés sur un ensemble d'entraînement fixe (`X_train`, `y_train` issu d'un partage initial 70/30) "
                    "et évalués sur `X_test` *après* l'étape de validation croisée."
                )

    # This initial split is always done for the "Test/Train (70/30)" method AND
    # for providing a consistent X_test for displaying predictions from models trained with CV methods.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Exécution des modèles ---
    if st.sidebar.button("Exécuter les modèles", key="run_models_btn") or "model_results" in st.session_state:
        models = {
            "Régression Linéaire": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso(), "ElasticNet": ElasticNet(),
            "PCR": PLSRegression(n_components=min(X_train.shape[1], 2 if X_train.shape[1] >=2 else 1)), # Ensure n_components <= n_features
            "PLS": PLSRegression(n_components=min(X_train.shape[1], 5 if X_train.shape[1] >=2 else 1)), # Ensure n_components <= n_features
            "KNN": KNeighborsRegressor(), "SVM": SVR(), "Arbre de Décision": DecisionTreeRegressor(),
            "Bagging": BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        if X_train.shape[1] == 0 :
             st.error("Impossible d'entraîner les modèles : Aucune variable explicative (X_train) disponible après transformations/split.")
             st.stop()
        if X_train.shape[1] == 1 : # Adjust for single feature case for PLS/PCR if necessary
            models["PCR"] = PLSRegression(n_components=1)
            models["PLS"] = PLSRegression(n_components=1)


        results = []
        all_predictions = {} # Stores predictions on the common X_test

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        total_models = len(models)

        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Entraînement du modèle : {name} ({i+1}/{total_models})")
            try:
                if resampling_method == "Test/Train (70/30)":
                    if X_train.empty:
                        raise ValueError("X_train est vide, impossible d'entraîner le modèle.")
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    mae = mean_absolute_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    all_predictions[name] = pred
                else: # Cross-validation methods
                    kf_cv = None
                    if resampling_method == "Bootstrap": # approximated with KFold shuffle=True
                        kf_cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42) if len(X) > 1 else None
                    elif resampling_method == "K-fold":
                        kf_cv = KFold(n_splits=min(5, len(X))) if len(X) > 1 else None
                    elif resampling_method == "Repeated K-fold":
                        kf_cv = RepeatedKFold(n_splits=min(5, len(X)), n_repeats=3, random_state=42) if len(X) > 1 else None
                    elif resampling_method == "Leave One Out":
                        kf_cv = LeaveOneOut() if len(X) > 1 else None
                    
                    if kf_cv is None or len(X) <= 1 : # Fallback for very small datasets or if kf_cv not init
                        # Fallback to Test/Train logic if CV is not possible
                        if X_train.empty:
                             raise ValueError(f"X_train est vide pour fallback sur {name}.")
                        model.fit(X_train, y_train)
                        pred_on_test = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, pred_on_test))
                        mae = mean_absolute_error(y_test, pred_on_test)
                        r2 = r2_score(y_test, pred_on_test)
                        all_predictions[name] = pred_on_test
                        if len(X) <=1 :
                            st.sidebar.warning(f"Peu de données ({len(X)} obs) pour {resampling_method}, fallback sur Test/Train pour {name}.")

                    else:
                        scores_rmse, scores_mae, scores_r2 = [], [], []
                        for train_idx, test_idx in kf_cv.split(X):
                            X_tr_fold, X_te_fold = X.iloc[train_idx], X.iloc[test_idx]
                            y_tr_fold, y_te_fold = y.iloc[train_idx], y.iloc[test_idx]
                            if X_tr_fold.empty: continue # Skip if fold results in empty train set

                            # Adjust n_components for PLS/PCR if it exceeds number of features in a fold
                            if name in ["PCR", "PLS"] and hasattr(model, 'n_components'):
                                current_n_components = model.get_params()['n_components']
                                if X_tr_fold.shape[1] < current_n_components :
                                    # Create a new model instance with adjusted n_components for this fold
                                    fold_model = model.__class__(n_components=max(1, X_tr_fold.shape[1]))
                                    fold_model.fit(X_tr_fold, y_tr_fold)
                                    pred_fold = fold_model.predict(X_te_fold)
                                else:
                                    model.fit(X_tr_fold, y_tr_fold)
                                    pred_fold = model.predict(X_te_fold)
                            else:
                                model.fit(X_tr_fold, y_tr_fold)
                                pred_fold = model.predict(X_te_fold)

                            scores_rmse.append(np.sqrt(mean_squared_error(y_te_fold, pred_fold)))
                            scores_mae.append(mean_absolute_error(y_te_fold, pred_fold))
                            scores_r2.append(r2_score(y_te_fold, pred_fold))
                        
                        rmse = np.mean(scores_rmse) if scores_rmse else np.nan
                        mae = np.mean(scores_mae) if scores_mae else np.nan
                        r2 = np.mean(scores_r2) if scores_r2 else np.nan

                        # For consistent prediction display, train on the initial X_train and predict on X_test
                        if X_train.empty:
                            raise ValueError(f"X_train est vide avant la prédiction finale pour {name}.")
                        model.fit(X_train, y_train) # Re-fit on the common X_train
                        all_predictions[name] = model.predict(X_test)


                results.append({"Modèle": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)})
            except Exception as e:
                results.append({"Modèle": name, "RMSE": "Échec", "MAE": "Échec", "R2": f"Échec: {str(e)[:50]}"})
                all_predictions[name] = [np.nan] * len(y_test) # Ensure correct length
            progress_bar.progress((i + 1) / total_models)
        
        status_text.text("Entraînement terminé !")
        st.session_state["model_results"] = pd.DataFrame(results)
        st.session_state["all_predictions"] = all_predictions
        progress_bar.empty() # Clear progress bar


    # --- Affichage résultats ---
    if "model_results" in st.session_state:
        st.header("📈 Performance des modèles")
        st.dataframe(st.session_state["model_results"].style.format(precision=4, na_rep="Échec"))

        st.header("🔮 Prédictions de tous les modèles (sur l'ensemble de test)")
        
        # Create the comparison DataFrame
        # y_test is from the initial split of (potentially transformed) X and y
        comparison_df_data = {"Valeur réelle": y_test.reset_index(drop=True)}
        
        if "all_predictions" in st.session_state:
            predictions_from_state = st.session_state["all_predictions"]
            for model_name_pred, preds_array in predictions_from_state.items():
                # Ensure preds_array is 1D and has correct length
                if isinstance(preds_array, np.ndarray) and preds_array.ndim > 1:
                    preds_array = preds_array.flatten() # Flatten if multi-dimensional
                
                # Pad with NaN if length mismatch (should not happen with robust error handling)
                if len(preds_array) != len(y_test):
                    padded_preds = np.full(len(y_test), np.nan)
                    common_len = min(len(preds_array), len(y_test))
                    padded_preds[:common_len] = preds_array[:common_len]
                    comparison_df_data[f"Prédiction ({model_name_pred})"] = pd.Series(padded_preds)
                else:
                     comparison_df_data[f"Prédiction ({model_name_pred})"] = pd.Series(preds_array)
            
            try:
                comparison_df = pd.DataFrame(comparison_df_data)
                st.dataframe(comparison_df.head(10).style.format(precision=3, na_rep="-"))

                # Interactive chart: Actual vs Predicted for a selected model
                # Use model names from results as they are guaranteed to exist
                model_names_for_plot = st.session_state["model_results"]["Modèle"].tolist()
                # Filter out failed models for selection
                successful_models = [m for m in model_names_for_plot if st.session_state["model_results"].set_index("Modèle").loc[m, "RMSE"] != "Échec" ]

                if successful_models:
                    selected_model_for_plot = st.selectbox(
                        "Sélectionnez un modèle pour visualiser ses prédictions vs valeurs réelles (50 premières lignes)",
                        successful_models
                    )
                    if selected_model_for_plot:
                        plot_df = comparison_df[["Valeur réelle", f"Prédiction ({selected_model_for_plot})"]].head(50)
                        st.line_chart(plot_df.set_index(plot_df.index)) # Plot against row index for clarity
                else:
                    st.info("Aucun modèle n'a réussi pour afficher le graphique des prédictions.")

            except Exception as e:
                st.error(f"Erreur lors de la création du DataFrame de comparaison ou du graphique : {e}")
                st.write("Données de prédiction :", predictions_from_state)


    else:
        st.info("Cliquez sur 'Exécuter les modèles' pour démarrer l'analyse après avoir configuré les étapes.")
else:
    st.info("Veuillez charger un fichier CSV ou Excel pour commencer.")

# --- Author Information from First Code ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="author-info">
    <h4>🧾 À propos de l'auteur</h4>
    <p><b>Nom:</b> N'dri</p>
    <p><b>Prénom:</b> Abo Onesime</p>
    <p><b>Rôle:</b> Data Analyst / Scientist</p>
    <p><b>Téléphone:</b> 07-68-05-98-87 / 01-01-75-11-81</p>
    <p><b>Email:</b> <a href="mailto:ndriablatie123@gmail.com" style="color:#00ff88;">ndriablatie123@gmail.com</a></p>
    <p><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/" target="_blank" style="color:#00ff88;">Profil LinkedIn</a></p>
    <p><b>GitHub:</b> <a href="https://github.com/Aboonesime" target="_blank" style="color:#00ff88;">Mon GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# --- Instructions ---
st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Uploader un fichier CSV ou Excel.
2. Sélectionner la variable cible (Y).
3. Choisir les transformations des données (jusqu'à 2).
4. Sélectionner une méthode de rééchantillonnage.
5. Exécuter les modèles et explorer les résultats.
""")

# --- Footer ---
st.markdown("""
<div class="footer">
    © 2025 Abo Onesime N'dri | Développé avec ❤️ en Python/Streamlit
</div>
""", unsafe_allow_html=True)