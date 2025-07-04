import pandas as pd
import streamlit as st
import json


def pred_from_response(response):
    extract = ["proba", "prediction", "threshold"]

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        st.error("Erreur : la réponse n'est pas une réponse de requête HTTTP valide.")
        st.text(f"Détails : {e}")
        return

    # Vérifie que chaque clé attendue est bien présente
    missing_keys = [k for k in extract if k not in data]
    if missing_keys:
        st.error(f"Clé(s) manquante(s) dans la réponse : {missing_keys}")
        return

    # Extraction de la proba, de la prédiction et su seuil
    try:
        proba = data["proba"]
        prediction = data["prediction"]
        threshold = data["threshold"]

        return proba, prediction, threshold

    except Exception as e:
        st.error("Erreur inattendue lors l'extraction des résultats du client.")
        st.text(f"Détails : {e}")
        return


def client_from_response(response, extract_results=True):

    if extract_results:
        extract = ["client_id", "features", "proba", "prediction"]
    else:
        extract = ["client_id", "features"]

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        st.error("Erreur : la réponse n'est pas une réponse de requête HTTTP valide.")
        st.text(f"Détails : {e}")
        return

    # Vérifie que chaque clé attendue est bien présente
    missing_keys = [k for k in extract if k not in data]
    if missing_keys:
        st.error(f"Clé(s) manquante(s) dans la réponse : {missing_keys}")
        return

    # Extraction sécurisée
    try:
        features = data["features"]
        client_id = data["client_id"]

        # Vérifie que 'features' est bien un dict
        if not isinstance(features, dict):
            st.error("'features' devrait être un dictionnaire.")
            return

        df = pd.DataFrame([features], index=[client_id])
        df.index.name = "SK_ID_CURR"
        if extract_results:
            df["PROBA"] = data["proba"]
            df["PREDICTION"] = data["prediction"]
        return df

    except Exception as e:
        st.error("Erreur inattendue lors de la création du DataFrame Client.")
        st.text(f"Détails : {e}")
        return


def st_filters_loop(
    df,
    features,
    min_sample_size=50,
    show_output_size=True,
    prefix_key="titi",
):
    filtered_df = df.copy()
    used_features = []
    remaining_features = features.copy()
    num_filter = 1

    while filtered_df.shape[0] >= min_sample_size and remaining_features:
        filtered_df, selected_feature = st_new_filter(
            df=filtered_df,
            features=remaining_features,
            num_filter=num_filter,
            min_sample_size=min_sample_size,
            show_output_size=show_output_size,
        )

        if selected_feature:
            used_features.append(selected_feature)
            remaining_features.remove(selected_feature)
            num_filter += 1
        else:
            # Si l’utilisateur ne sélectionne pas de nouvelle feature
            break

    return filtered_df, used_features


def st_new_filter(
    df, features, num_filter=1, min_sample_size=50, show_output_size=True
):
    """# On vérifie que la taille de l'échantillon permet de proposer un filtre
    if df.shape[0] <= min_sample_size:
        st.text(
            f"Il ne reste plus assez d'observations pour proposer un nouveau filtre"
        )
        st.text(
            f"\n(taille de l'échantillon : {df.shape[0]}, minimum : {min_sample_size})"
        )
        selected_feature = ""
        return df, selected_feature"""

    # On distingue les features qui nécessiteront un filtre catégoriel de celles qui nécessiteront un filtre avec slider
    slider_features = [f for f in features if df[f].nunique() > 5]
    category_features = [f for f in features if f not in slider_features]

    # [TODO]
    # Si la feature est catégorielle et qu'il ne reste qu'une catégorie,
    # Voir ce qu'on fait (mise à jour de la liste de features ou contrôle des valeurs restantes)
    # Idem pour les features avec sliders, que faire si continue devient catégorielle car peu de valeurs ?

    if show_output_size:
        col1, col2, col3 = st.columns([1, 2, 1])
    else:
        col1, col2 = st.columns([1, 2.5])

    with col1:
        selected_feature = st.selectbox(
            f"Filtre N°{num_filter}",
            features,
            index=None,
            key=f"feature_filter_{num_filter}",
        )
    if selected_feature:
        with col2:
            if selected_feature in category_features:
                selected_categories = st.multiselect(
                    f"Catégories de {selected_feature}",
                    df[selected_feature].unique(),
                    key=f"categories_filter_{num_filter}",
                )
                filtered_df = df[df[selected_feature].isin(selected_categories)]
            else:
                min_val = float(df[selected_feature].min())
                max_val = float(df[selected_feature].max())
                selected_numbers = st.slider(
                    f"Valeurs de {selected_feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"slider_filter_{num_filter}",
                )
                filtered_df = df[
                    (df[selected_feature] >= selected_numbers[0])
                    & (df[selected_feature] <= selected_numbers[1])
                ]
    else:
        selected_categories = None
        selected_numbers = None
        filtered_df = df.copy()
        selected_feature = ""
    if col3:
        with col3:
            st.write("Nouvelle Taille échantillon", filtered_df.shape[0])
    return filtered_df, selected_feature


# === Colorie certaines colonnes dans un dataframe
def highlight_selected_columns(row, cols, color_pos, color_neg):
    styles = []
    for col in row.index:
        if col in cols:
            color = color_pos if row[col] >= 0 else color_neg
            styles.append(f"background-color: {color}")
        else:
            styles.append("")
    return styles


# Renvoie le texte expliquant pour un non staticien le graphique d'explication locale (SHAP Local Explanation)
def local_explanation_jargon(base_value, df_local_importance, proba_client):
    text = "Le graphique 'SHAP Waterfall plot' montre comment le modèle évalue le risque d'insolvabilité pour un client donné.\n"
    text += "Il indique quelles informations (appelées 'caractéristiques' ou 'features') ont eu le plus d'impact sur la prédiction du modèle.\n"
    text += "\t* Plus la barre est longue, plus l'influence de la caractéristique est forte,\n"
    text += f"\t* Chaque barre, par sa couleur et son orientation, indique si l'information augmente ou diminue le risque d'insolvabilité pour ce client.\n\n"

    text += "Comment lire le graphique plus en détail ?\n"
    text += f"1) En bas du graphique, on voit une valeur de base, notée 'E[f(X)] = {base_value:.3f}'.\n"
    text += "C'est le risque moyen d'insolvabilité pour un client 'type', lorsqu'on ne connaît aucune information personnelle sur lui.\n"
    proba_client = round(proba_client, 3)
    text += f"2) En haut, on trouve la prédiction du modèle pour ce client, notée 'f(x)={proba_client:g}'\n"
    text += f"C'est le risque calculé par le modèle en tenant compte des caractéristiques personnelles du client.\n"
    text += f"3) Entre ces deux étapes, les barres montrent l'impact des différentes caractéristiques sur le risque. Par exemple :\n"
    feature = df_local_importance.head(1)["feature"].item()
    value_client = df_local_importance.loc[
        df_local_importance["feature"] == feature, "valeur_client"
    ].item()
    if value_client:
        # On affiche la valeur avec au plus 3 décimales, le format g n'affichera pas les décimales non significatives
        value_client = round(value_client, 3)
        text += f"Dans les données du client choisi, on sait que {feature} a une valeur de {value_client:g}.\n"
    else:
        text += (
            f"Dans les données du client choisi, la valeur de {feature} est inconnue.\n"
        )
    # text += f"La caractéristique {feature} a une valeur de {df_local_importance.loc[df_local_importance['feature'] == feature, 'valeur_client'].item():.3f} pour ce client.\n"
    local_importance = df_local_importance.loc[
        df_local_importance["feature"] == feature, "Importance_locale"
    ].item()
    effect_str = ""
    influence_str = ""
    if local_importance > 0:
        effect_str = f"Cette information concernant le client a fait augmenter son risque de {local_importance:.2f} (notée dans la barre, le signe '+' indique une augmentation du risque).\n"
        influence_str = "La caractéristique ayant augmenté le risque, elle a influencé la prédiction du modèle vers un refus du prêt.\n"
    else:
        effect_str = f"Cette information concernant le client a fait diminuer le risque de {local_importance:.2f} (notée dans la barre, le signe '-' indique une diminution du risque).\n"
        influence_str = "La caractéristique ayant diminué le risque, elle a influencé la prédiction du modèle vers un accord du prêt.\n"

    text += effect_str
    text += influence_str
    text += "4 - Nota Bene) En additionnant la valeur de base et toutes les influences (avec leur signe d'augmentation ou de diminution du risque), on obtient le risque calculé par le modèle pour le client choisi."
    # text += "Les caractéristiques son classées de haut en bas selon la force de leur influence."
    return text


# Pour plus d'inclusivité, renvoie le texte de remplacement des éléments visuels 'imagés' contenus dans le graphique SHAP Local Explanation
def local_explanation_replacement(
    base_value, df_local_importance, proba_client, top_local_features
):
    text = f"Les {len(top_local_features)} caractéristiques qui impactent le plus la décision pour le client sont :\n"

    for f in top_local_features:
        mask = df_local_importance["feature"] == f
        detail = f"\tInfluence ({df_local_importance.loc[mask, 'Importance_locale'].item():.2f}) "
        if df_local_importance.loc[mask, "Importance_locale"].item() > 0:
            detail += "vers le REFUS du prêt"
        else:
            detail += "vers l'ACCORD du prêt"
        text += f"{f}{detail}\n"

    text += (
        f"\nLa valeur de base (sans connaissance du client) est de {base_value:.3f}\n"
    )
    text += f"La prédiction du risque d'insolvabilité pour ce client (connaissant ses informations) est de {proba_client:.3f}."
    return text


def gauge_replacement(client_id, proba_client, threshold):
    # Texte à afficher si le crédit est accordé
    text_gauge_safe = f"Le crédit est accordé pour le client {client_id}.\n"
    text_gauge_safe += (
        f"Le risque d'insolvabilité du client est de {proba_client:.2f}, "
    )
    text_gauge_safe += f"ce risque est tolérable pour la banque qui accepte jusqu'à {threshold:.2f} de risque."

    # Texte à afficher si le crédit est refusé
    text_gauge_risk = f"Le crédit est refusé pour le client {client_id}.\n"
    text_gauge_risk += (
        f"Le risque d'insolvabilité du client est de {proba_client:.2f}, "
    )
    text_gauge_risk += f"ce risque est supérieur au risque que la banque peut tolérer ({threshold:.2f})."

    if proba_client > threshold:

        text = text_gauge_risk

    else:
        text = text_gauge_safe
    return text


def gauge_jargon(client_id, proba_client, threshold):
    # Risk
    text_gauge = "Le risque est la probabilité (prédite par le modèle) que le client ne puisse pas rembourser l'emprunt.\n"
    text_gauge += (
        f"Une probabilité de {proba_client:.2f} peut s'interpréter en pourcentage :\n"
    )
    text_gauge += f" Le client a {proba_client:.0%} de 'chances' sur 100 d'avoir des difficultés de remboursement.\n"

    text_gauge += f"\nEnsuite, la banque a un seuil stratégique de décision, ici {threshold:.2f}.\n"
    text_gauge += "Si le risque est en dessous de seuil, elle accorde le prêt,"
    text_gauge += "si le risque est trop grand, elle le refuse.\n"
    text_gauge += "Ce seuil de risque tolérable est optimisé de manière à ce que la banque perde le moins d'argent possible. Il n'est pas forcément à 0.5.\n"
    text_gauge += f"Ici, le seuil de risque tolérable est {threshold:.2f}"

    """# Texte à afficher si le crédit est accordé
    text_gauge_safe = f"Le crédit est accordé pour le client {client_id}.\n"
    text_gauge_safe += (
        f"Le risque d'insolvabilité du client est de {proba_client:.2f}, "
    )
    text_gauge_safe += f"ce risque est tolérable pour la banque qui accepte jusqu'à {threshold:.2f} de risque."

    # Texte à afficher si le crédit est refusé
    text_gauge_risk = f"Le crédit est refusé pour le client {client_id}.\n"
    text_gauge_risk += (
        f"Le risque d'insolvabilité du client est de {proba_client:.2f}, "
    )
    text_gauge_risk += f"ce risque est supérieur au risque que la banque peut tolérer ({threshold:.2f})."

    if proba_client > threshold:

        text = text_gauge_risk

    else:
        text = text_gauge_safe"""
    return text_gauge


# REnvoie le texte de remplacement des éléments graphiques pour une boîte à moustaches croisée avec une feature catégorielle
def boxplot_bivariate_replacement(df, client, numeric_feature, category_feature):
    # Si toutes les données nécessaires sont connues pour le client
    if client[numeric_feature] is not None and client[category_feature] is not None:
        text = f"Le client {client.name} appartient à la catégorie {client[category_feature].astype(int)} de {category_feature}.\n"
        median_value = (
            df.loc[df[category_feature] == client[category_feature], numeric_feature]
            .dropna()
            .median()
        )
        text += f"Au sein de cette catégorie :\n"
        text += f"\tla médiane de {numeric_feature} est de {median_value:.2f}.\n"
        if client[numeric_feature] > median_value:
            text += f"\tLe client {client.name} se situe en dessus de la médiane ({numeric_feature} = {client[numeric_feature]:.2f}).\n"
        else:
            text += f"\tLe client {client.name} se situe au dessous de la médiane ({numeric_feature} = {client[numeric_feature]:.2f}).\n"

        q1 = (
            df.loc[df[category_feature] == client[category_feature], numeric_feature]
            .dropna()
            .quantile(0.25)
        )
        q3 = (
            df.loc[df[category_feature] == client[category_feature], numeric_feature]
            .dropna()
            .quantile(0.75)
        )
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if (
            client[numeric_feature] < lower_bound
            or client[numeric_feature] > upper_bound
        ):
            text += f"\tLe client {client.name} est atypique par rapport à la population pour la caractérisitique {numeric_feature}."
        else:
            text += f"\tLe client {client.name} n'est pas atypique par rapport à la population pour la caractérisitique {numeric_feature}."

    # Si on ne connait pas la catégorie mais on connait la feature numerique
    elif client[category_feature] is None and client[numeric_feature] is not None:
        text = f"La catégorie du client {client.name} est inconnue pour la caractéristique {category_feature}."
        text += f"La valeur de {numeric_feature} = {client[numeric_feature]:.2f}"
    elif client[category_feature] is not None and client[numeric_feature] is None:
        text = (
            f"La valeur de {numeric_feature} est inconnue pour le client {client.name}."
        )
    else:
        text = f"Les valeurs de {numeric_feature} et de {category_feature} sont inconnues pour le client {client.name}."
    return text
