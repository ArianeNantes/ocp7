import sys
import os

sys.path.append(os.path.abspath("."))

import pandas as pd
import streamlit as st

import requests


import shap
import streamlit.components.v1 as components
import joblib


# from src_api.routes.prediction_routes import router as prediction_router
from src.dashboard.charts import (
    gauge_plt,
    shap_waterfall,
    shap_force,
    boxplot_by_categorical,
    histogram_by_categorical,
    plot_scatter,
    plotly_style_title,
    global_importance_barh,
    pie,
    histogram_univariate,
)
from src.dashboard.util import (
    st_filters_loop,
    client_from_response,
    pred_from_response,
    local_explanation_jargon,
    local_explanation_replacement,
    gauge_replacement,
    gauge_jargon,
    boxplot_bivariate_replacement,
)
from src.dashboard.util import highlight_selected_columns
from src.constantes import (
    MODEL_DIR,
    DATA_CLEAN_DIR,
    BEST_DTYPES_NAME,
    NEW_LOANS_NAME,
    EXPLAINER_NAME,
    SAMPLE_TRAIN_NAME,
    GLOBAL_IMPORTANCES_NAME,
)

# THRESHOLD_PROB = 0.48
DEFAULT_LOANER_ID = 0
DEFAULT_COLOR_LOANER = "#F4E921"
DEFAULT_COLOR_RISK = "#7200CA"
DEFAULT_COLOR_SAFE = "#A1DE92"


# Chargement des données et des ressources


@st.cache_data
def load_new_loan_ids():
    # On remonte de deux crans et on descend dans data/cleaned
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", DATA_CLEAN_DIR)

    # Chemin du fichier contenant les nouveaux emprunts (en provenance de application_test)
    df_path = os.path.join(data_dir, NEW_LOANS_NAME)
    df_path = os.path.abspath(df_path)
    if not os.path.exists(df_path):
        raise FileNotFoundError(
            f"Le fichier de données des nouveaux emprunts '{df_path}' est introuvable."
        )

    # Nous n'avons pas besoin de lire le fichier entier mais uniquement des index,
    # on lit la première colonne et on récupère les index
    first_column = pd.read_csv(df_path, usecols=[0], index_col=0).sort_index()
    ids = first_column.index.astype(int).tolist()
    return ids


@st.cache_data
def load_train():
    # On remonte de deux crans et on descend dans data/cleaned
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", DATA_CLEAN_DIR)

    # Chemin pour le fichier de données d'entraînement
    train_path = os.path.join(data_dir, SAMPLE_TRAIN_NAME)
    train_path = os.path.abspath(train_path)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Le fichier de données d'entrainement '{train_path}' est introuvable."
        )

    # Chemin pour le dictionnaire utilisé afin de caster les colonnes
    dtypes_path = os.path.join(data_dir, BEST_DTYPES_NAME)
    dtypes_path = os.path.abspath(dtypes_path)
    if not os.path.exists(dtypes_path):
        raise FileNotFoundError(
            f"Le fichier dictionnaire de features '{dtypes_path}' est introuvable."
        )

    dtypes = joblib.load(dtypes_path)

    train = pd.read_csv(train_path, dtype=dtypes)

    if "TARGET" in train.columns:
        train = train.drop(columns=["TARGET"])
    if "SK_ID_CURR" in train.columns:
        train = train.set_index("SK_ID_CURR")

    features = [
        f
        for f in train.columns.to_list()
        if f not in ["SK_ID_CURR", "PROBA", "PREDICTION"]
    ]

    return train, features


@st.cache_resource
def load_explainer():
    # On remonte de deux crans et on descend dans models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", MODEL_DIR)

    # Chemin pour l'explainer
    explainer_path = os.path.join(model_dir, EXPLAINER_NAME)
    explainer_path = os.path.abspath(explainer_path)
    if not os.path.exists(explainer_path):
        raise FileNotFoundError(
            f"Le fichier modèle '{explainer_path}' est introuvable."
        )
    explainer = joblib.load(explainer_path)

    return explainer


@st.cache_data
def load_global_importances():
    # On remonte de deux crans et on descend dans models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", MODEL_DIR)

    # Chemin pour le dataframe des importances globales
    path_importances = os.path.join(model_dir, GLOBAL_IMPORTANCES_NAME)
    path_importances = os.path.abspath(path_importances)
    if not os.path.exists(path_importances):
        raise FileNotFoundError(
            f"Le fichier des importances globales '{path_importances}' est introuvable."
        )
    df = pd.read_csv(path_importances)
    return df


# plots


# Pour afficher les graphiques SHAP dans streamlit en HTML
# Voir : https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/8
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# Valeurs par défaut / default values
n_features_default = 10
# color_default_safe = "#ACF39A"
color_default_safe = DEFAULT_COLOR_SAFE
color_default_risk = DEFAULT_COLOR_RISK
color_default_loaner = DEFAULT_COLOR_LOANER
decision_explanation = False


# Fonctions pour les DF
# === Fonction de coloration ligne par ligne
def color_rows(row, colors=[color_default_safe, color_default_risk]):
    color = colors[1] if row["contribution"] >= 0 else colors[0]
    return [f"background-color: {color}"] * len(row)


# Fonction de coloration cellule par cellule
def color_contribution(val):
    color = color_default_risk if val >= 0 else color_default_safe
    return f"background-color: {color}"


########################################################## Dashboard
# Dictionnaire des labels pour les features catégorielles.
# (REGION_RATING_CLIENT_W_CITY n'y figure pas car c'est 1, 2, 3, pas de labels dans lez données initiales)
dic_labels = {
    "CODE_GENDERS": {
        1: "Female",
        0: "Male",
        -1: "XNA",
    },
    "NAME_EDUCATION_TYPE_Highereducation": {
        0: "No",
        1: "Higher education",
    },
    "NAME_FAMILY_STATUS_Married": {
        0: "Not married",
        1: "Married",
    },
    "PREDICTION": {
        0: "Crédit Accordé",
        1: "Crédit Refusé",
    },
}


# Configuration de la page
st.set_page_config(
    page_title="Home Credit",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    #                    base='light'
)
st.title("Home Credit Dashboard")
st.markdown(
    "<style>div.block-container{padding-top:2rem}</style>", unsafe_allow_html=True
)


train, features = load_train()
new_loans_ids = load_new_loan_ids()

# Sidebar
with st.sidebar:

    st.sidebar.header(":material/settings: Préférences", divider="gray")

    # Couples de couleurs prédéfinis
    color_schemes = {
        "Rouge / Vert": ("#FF4B4B", "#28B463"),
        "Bleu / Orange": ("#3498DB", "#E67E22"),
        "Gris / Violet": ("#95A5A6", "#8E44AD"),
    }

    with st.sidebar.popover("Modifier"):
        # Au démarrage toutes les couleurs sont celles par défaut
        color_safe = color_default_safe
        color_risk = color_default_risk
        color_loaner = color_default_loaner

        # On expose des possibilités de changer les couleurs pas défaut
        color_safe = st.color_picker("Accord de crédit", value=color_default_safe)
        color_risk = st.color_picker("Refus de crédit", value=color_default_risk)
        color_loaner = st.color_picker("Emprunteur", value=color_default_loaner)

    # Affichage pour confirmation
    # Créer des options avec affichage HTML de carrés de couleur
    options = list(color_schemes.keys())
    st.sidebar.markdown(
        f"""        
        <div style='display: flex; gap: 10px; align-items: center; margin-top: 1px;'>
            <div style='width: 20px; height: 20px; background-color: {color_safe}; border: 1px solid black;'></div>
            <span>Accord de crédit</span>
        </div>
        <div style='display: flex; gap: 10px; align-items: center;'>
            <div style='width: 20px; height: 20px; background-color: {color_risk}; border: 1px solid black;'></div>
            <span>Refus de crédit</span>
        </div>
        <div style='display: flex; gap: 10px; align-items: center; margin-bottom: 10px;'>
            <div style='width: 20px; height: 20px; background-color: {color_loaner}; border: 1px solid black;'></div>
            <span>Client</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    explainer = load_explainer()

    # st.sidebar.header(":material/Shield_Question: Prédiction", divider="gray")
    st.sidebar.header(":material/Live_Help: Décision", divider="gray")
    # st.sidebar.header(":material/Query_Stats: Prédiction", divider="gray")

    # Crée un sélecteur basé sur l'index du df client
    client_id = int(st.selectbox("Sélectionnez un ID client :", new_loans_ids))

    # Crée un slider pour le nombre de features à afficher dans les graphs SHAP et le df correspondant
    n_features = st.slider(
        "Nb caractéristiques", min_value=n_features_default, max_value=40, step=10
    )

    # st.sidebar.header(":material/Search: Look Distibution", divider="gray")
    st.sidebar.header(":material/Query_Stats: Analyse", divider="gray")
    # Lien link material icon : https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded


# Probabilité du client par l'API
proba_client = None
prediction_client = None
response = None

url_get = f"https://credit-score2-ejgde3bqeaa7axgn.canadacentral-01.azurewebsites.net/features/{client_id}"
# url_get = f"http://127.0.0.1:8000/features/{client_id}"
response_get = requests.get(url_get)
# [DEBUG] st.write("Reponse GET", response_get.json())

# url_predict = "http://127.0.0.1:8000/predict"
url_predict = (
    "https://credit-score2-ejgde3bqeaa7axgn.canadacentral-01.azurewebsites.net/predict"
)
# input_json = {"client_id": client_id}
input_json = response_get.json()
response = requests.post(url_predict, json=input_json)

if response.status_code == 200:
    client = client_from_response(response, extract_results=True)
    X_client = client[features]
    proba_client, prediction_client, threshold = pred_from_response(response)
    proba, prediction, threshold = pred_from_response(response)
    # st.success(f"Risque de défaut : {proba_client:.2f}")
else:
    st.error(f"Erreur lors de la requête : {response.status_code} - {response.text}")

shap_values_client = None

##################################################### Prédiction et explication Locale SHAP
st.header("Décision et facteurs décisifs")
st.subheader(f"Pour le client ID {client_id}")
# On partage en trois colonnes pour la prédiction
col_gauge, col_local_explanation = st.columns([1.2, 2])

with col_gauge:
    if proba_client:
        decision_str = "Crédit "
        decision_str += "Accordé" if proba_client < threshold else "Refusé"
        plotly_style_title(f"{decision_str}")
        with st.expander("", expanded=True):
            fig_plt = gauge_plt(
                proba=proba_client,
                threshold_prob=threshold,
                title="",
                colors=[color_safe, color_risk],
            )
            st.pyplot(fig_plt)

        text_gauge_replacement = gauge_replacement(
            client_id=client_id, proba_client=proba_client, threshold=threshold
        )
        st.text(
            text_gauge_replacement,
        )

        with st.expander(f"Réponse à la requête POST pour le client {client_id}"):
            st.write(response.json())
        with st.expander(f"Données du client {client_id}"):

            st.write(client.T)
        with st.expander("Jargon Jauge"):
            st.text(
                gauge_jargon(
                    client_id=client_id, proba_client=proba, threshold=threshold
                ),
            )

# Colonne pour le dataframe shap values locales
with col_local_explanation:
    if proba_client:
        shap_values_client = explainer(X_client)
        fig_plt = shap_waterfall(
            shap_values_client,
            colors=[color_safe, color_risk],
            # Plus 1 car nombre de barres est le nombre de features plus
            # une barre qui représente la synthèse des autres features
            max_display=n_features + 1,
            # title=f"Pour le client ID = {selected_id}",
        )
        # components.html(fig_html, height=800)
        plotly_style_title(
            f"Top {n_features} des caratéristiques les plus influentes pour le client {client_id}"
        )
        with st.expander("", expanded=True):
            st.pyplot(fig_plt)

    # Calcul approximatif de la hauteur en fonction du nombre de features sélectionné par l'utilisateur
    margin_top = 35
    feature_height = 35
    height = margin_top + n_features * feature_height

    if shap_values_client is not None:
        # On crée le df du client trié par contribution des features
        shap_df_local = pd.DataFrame(
            {
                # "feature": X_client.columns,
                "feature": features,
                "valeur_client": X_client.values.flatten(),
                "Importance_locale": shap_values_client.values.flatten(),
            }
        )

        # On trie par contribution absolue décroissante (importance)
        shap_df_local["abs_contribution"] = shap_df_local["Importance_locale"].abs()
        shap_df_local = shap_df_local.sort_values(
            by="abs_contribution", ascending=False
        )

        text_local_jargon = local_explanation_jargon(
            base_value=explainer.expected_value,
            df_local_importance=shap_df_local,
            proba_client=proba_client,
        )

        top_local_features = shap_df_local.head(n_features)["feature"].to_list()

        text_local_explanation = local_explanation_replacement(
            base_value=explainer.expected_value,
            df_local_importance=shap_df_local,
            proba_client=proba_client,
            top_local_features=top_local_features,
        )
        with st.expander(
            "Aide visuelle (texte de remplacement des principaux éléments graphiques)"
        ):
            st.text(text_local_explanation)
        with st.expander(f"Aide technique/jargon"):
            st.text(text_local_jargon)

        # On copie le df des importances avant de l'afficher car l'affichage permet le tri par l'utilisateur.
        # Si l'utilisateur trie le df alors le df orrignal ne serait plus trié correctement.
        # 1. Sélectionner les colonnes avant de styler
        df_to_display = shap_df_local.copy()

        # === Application du style
        styled_df = df_to_display.style.apply(
            lambda row: highlight_selected_columns(
                row, ["Importance_locale"], color_risk, color_safe
            ),
            axis=1,
        )

        # On le montre le dataframe stylisé dans unexpander,
        # Ainsi, en cliquant de ssu on pourra zoomer
        with st.expander(f"Voir les importances locales du client {client_id}"):
            # st.table(
            #    styled_df,
            #    height=height,
            # )
            st.dataframe(
                styled_df,
                height=height,
                use_container_width=True,
            )


############################################### Explication globale Avec SHAP
st.divider()
st.subheader("Pour la population")

global_importances = load_global_importances()


plotly_style_title(
    f"Top {n_features} des caratéristiques les plus influentes globalement"
)
with st.expander("", expanded=True):
    fig = global_importance_barh(
        # shap_df_global, max_display=n_features, plot_others=True
        global_importances,
        max_display=n_features,
        plot_others=True,
    )
    st.plotly_chart(fig, use_container_with=True)


# === Application du style
# On copie le df des importances avant de l'afficher car l'affichage permet le tri par l'utilisateur.
# Si l'utilisateur trie le df alors le df orrignal ne serait plus trié correctement.

df_global_to_display = global_importances.copy()
styled_df_global = df_global_to_display.style.apply(
    lambda row: highlight_selected_columns(
        row, ["signed_mean"], color_risk, color_safe
    ),
    axis=1,
)

# Affichage
with st.expander("Voir les importances globales"):
    st.dataframe(styled_df_global)

st.divider()
################################################################ Filtres - Graph plotly
st.header(f"Analyse des caractéristiques")

categorical_features = []
numeric_features = []
for f in train.columns:
    if train[f].nunique() > 4:
        numeric_features.append(f)
    else:
        categorical_features.append(f)


########################## Filtres statiques
st.subheader("Univariée - Caractéristiques les plus influentes pour le client")

# Sur toute la largeur, on affiche le force-plot local
fig_html = shap_force(shap_values_client, X_client, colors=[color_safe, color_risk])
st_shap(fig_html)

# Dans le force plot, les features de risk (tirent vers le refus) sont affichées à gauche
col_risk, col_safe = st.columns(2, gap="large")

# features importance local qui tirent vers le refus du prêt
features_risk = (
    shap_df_local[shap_df_local["Importance_locale"] > 0]["feature"].squeeze().to_list()
)
# features qui tirent le client vers l'accord du prêt
features_safe = (
    shap_df_local[shap_df_local["Importance_locale"] <= 0]["feature"]
    .squeeze()
    .to_list()
)

client_serie = client.squeeze()
with col_safe:
    # On modifie le flux standard de calcul de streamlit, de façon à ne pas tout recalculer depuis le haut du dasboard jusqu'en bas
    # (concernant les autres parties), si une option choisie concernant uniquement la colonne 'Caractérisitiques diminuant le risque'.
    # Cela améliore les temps de réponse du dashboard.
    @st.fragment
    def plot_safe_univariate():
        selected_safe = st.selectbox(
            "Caractéristiques qui favorisent le prêt", features_safe, index=0
        )

        if selected_safe:
            if selected_safe in categorical_features:

                fig_pie_safe = pie(train, client_serie, feature=selected_safe)
                plotly_style_title(f"Répartition de {selected_safe}")
                st.plotly_chart(fig_pie_safe)
                text_to_display = f"Le client {client_serie.name} appartient à la catégorie {client_serie[selected_safe].astype(int)} de {selected_safe}\n"
                text_to_display += "Cela joue en sa faveur pour obtenir le prêt"
                st.text(text_to_display)
            else:
                fig_hist_safe, text_to_display = histogram_univariate(
                    train,
                    client=client.squeeze(),
                    numeric_feature=selected_safe,
                    color_client=color_loaner,
                )
                plotly_style_title(f"Histogramme de {selected_safe}")
                st.plotly_chart(fig_hist_safe)
                st.text(text_to_display)

    plot_safe_univariate()

with col_risk:
    # On modifie le flux standard de calcul de streamlit, de façon à ne pas tout recalculer depuis le haut du dasboard
    # (concernant les autres parties), si une option choisie concernant uniquement la colonne 'Caractérisitiques augmentant le risque'.
    # Cela améliore les temps de réponse du dashboard.
    @st.fragment
    def plot_risk_univariate():
        selected_risk = st.selectbox(
            "Caractéristiques qui défavorisent le prêt", features_risk, index=0
        )
        if selected_risk:
            if selected_risk in categorical_features:
                client_serie = client.squeeze()
                fig_pie_risk = pie(train, client_serie, feature=selected_risk)
                st.plotly_chart(fig_pie_risk)
                if client_serie[selected_risk] is not None:
                    text_to_display = f"Le client {client_serie.name} appartient à la catégorie {client_serie[selected_risk].astype(int)} de {selected_risk}\n"
                    text_to_display += f"Cela joue en sa défaveur pour obtenir le prêt"
                else:
                    text_to_display = f"La valeur de {selected_risk} est inconnue pour le client {client_serie.name}"
                st.text(text_to_display)
            else:
                fig_hist_risk, text_to_display = histogram_univariate(
                    train,
                    client=client.squeeze(),
                    numeric_feature=selected_risk,
                    color_client=color_loaner,
                )
                plotly_style_title(f"Histogramme de {selected_risk}")
                st.plotly_chart(fig_hist_risk)
                st.text(text_to_display)

    plot_risk_univariate()

with st.expander("Statistiques descriptives"):
    st.dataframe(train[numeric_features].describe())

st.divider()
################################################################ Bivariée - Graph plotly
st.subheader("Bivariée - Caractéristique numérique par catégories")


# On modifie le flux standard de calcul de streamlit, de façon à ne pas tout recalculer depuis le haut du dasboard
# (concernant les autres parties), si une option choisie concernant uniquement les boxplots est modifiée.
# Cela améliore énormément les temps de réponse du dashboard.
@st.fragment
def plot_bivariate_boxplot():
    # plotly_style_title("Sélection des caractéristiques")
    col_feature_num, col_feature_cat = st.columns(2)
    with col_feature_num:
        selected_numeric = st.selectbox(
            "Caractéristique numérique", numeric_features, index=0
        )
    with col_feature_cat:
        selected_category = st.selectbox(
            "Croisée par",
            categorical_features,
            index=categorical_features.index("PREDICTION"),
        )
    # Détection du cas où la variable catégorielle est binaire et doit avoir des couleurs fixées
    # Si on ne fait pas ça, les couleurs sont rangées non pas dans l'ordre 0 ou 1 de prédiction,
    # Si le premoier trouvé est 1 alors color_safe serait appliqué à la target 0 !
    if selected_category in ["PREDICTION", "TARGET"]:
        category_order = [0, 1]
        color_map = {0: color_safe, 1: color_risk}
    # Si la variable catégorielle choisie n'est pas la prédiction, on laisse Plotly choisir les couleurs
    else:
        category_order = sorted(train[selected_category].dropna().unique())
        color_map = None

    st.write(" ")
    col_box, col_hist = st.columns([1, 2], gap="large")

    # Boxplot
    with col_box:
        plotly_style_title(f"Boxplot {selected_numeric} par {selected_category}")
        # use_log = st.checkbox("Utiliser une échelle logarithmique sur l'axe Y", value=False)
        use_log = False
        # show_outliers = st.checkbox("Afficher les outliers", value=True)

        fig_boxplot = boxplot_by_categorical(
            df=train,
            client=client.squeeze(),
            numeric_feature=selected_numeric,
            category_feature=selected_category,
            colors=[color_safe, color_risk, color_loaner],
        )
        st.plotly_chart(
            fig_boxplot,
            use_container_width=True,
        )

    # Histogramme
    with col_hist:

        # with st.expander("Graph plotly", expanded=True):
        plotly_style_title(f"Histogramme {selected_numeric} par {selected_category}")
        fig_histo = histogram_by_categorical(
            train,
            client=client.squeeze(),
            numeric_feature=selected_numeric,
            category_feature=selected_category,
            colors=[color_safe, color_risk, color_loaner],
        )
        st.plotly_chart(fig_histo, theme="streamlit", use_container_width=True)

    text_bivariate_replacement = boxplot_bivariate_replacement(
        train,
        client.squeeze(),
        numeric_feature=selected_numeric,
        category_feature=selected_category,
    )
    st.text(text_bivariate_replacement)


plot_bivariate_boxplot()

#######################


# On modifie le flux standar de calcul de streamlit, de façon à ne pas tout recalculer depuis le haut du dasboard
# (concernant les autres parties), si une option choisie concernant uniquement le nuage de points est modifiée.
# Cela améliore énormément les temps de réponse du dashboard.
@st.fragment
def plot_bivariate_scatter():
    # st.subheader("Filtrer les données")
    # On construit la liste des features principales constitués des principales features SHAP globale et
    # des principales features SHAP locale
    main_global_features = global_importances["feature"].to_list()[:n_features]
    main_local_features = shap_df_local["feature"].to_list()[:n_features]
    main_features = (
        main_local_features
        + [f for f in main_global_features if f not in main_local_features]
        + ["PROBA", "PREDICTION"]
    )

    main_features.sort()
    features_left = main_features.copy()
    # st.write("debug df_all", train.head(1))
    df_main = train[main_features]

    st.subheader("Bivariée - Nuage de points coloré par le risque d'insolvabilité")

    ######################## Filtres dynamiques (nombre dynamique)
    # plotly_style_title("Filtrer les données")
    df_filtered, selected_filters = st_filters_loop(
        df=df_main,
        features=main_features,
        min_sample_size=50,
        show_output_size=True,
    )

    # df_filtered, selected_filters = st_filters_loop_button(
    # df=df_main,
    # features=main_features,
    # min_sample_size=50,
    # )
    # st.write("debug boucle nombre de filtres dynamique FINI")
    # st.write(df_filtered.head())

    df3 = df_filtered
    # plotly_style_title("Sélection des caractéristiques")
    col_num_y, col_num_x = st.columns(2)
    with col_num_y:

        feature_y = st.selectbox(
            "Caractéristique pour y",
            df3.select_dtypes(include="number").columns.to_list(),
            index=1,
        )
    with col_num_x:
        feature_x = st.selectbox(
            "En fonction de (carastéristique pour x)",
            df3.select_dtypes(include="number").columns.to_list(),
            index=0,
        )

    st.write(" ")

    colors = {
        "color_safe": color_safe,
        "color_risk": color_risk,
        "color_loaner": color_loaner,
        "threshold_prob": threshold,
    }

    plotly_style_title(f"{feature_y} en fonction de {feature_x}")
    points, text_to_display = plot_scatter(
        df3, feature_x, feature_y, client.squeeze(), colors=colors
    )
    st.plotly_chart(points)
    st.text(text_to_display)


plot_bivariate_scatter()
