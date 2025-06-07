import pandas as pd
import streamlit as st
import json
import requests


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


"""def st_dynamic_filters(
    df,
    features,
    min_sample_size=50,
    show_output_size=True,
    max_filters=10,
):
    df_filtered = df.copy()
    used_features = []
    available_features = features.copy()
    filter_history = [(df.copy(), None)]  # historique pour rollback

    is_filtered = False
    for i in range(max_filters):
        if df_filtered.shape[0] <= min_sample_size or not available_features:
            break

        st.markdown(f"### Filtre {i + 1}")
        df_filtered, selected_feature = st_new_filter(
            df=df_filtered,
            features=available_features,
            num_filter=i + 1,
            min_sample_size=min_sample_size,
            show_output_size=show_output_size,
        )

        if selected_feature:
            is_filtered = True
            used_features.append(selected_feature)
            available_features.remove(selected_feature)
            filter_history.append((df_filtered.copy(), selected_feature))
        else:
            break

    # Vérification de la taille finale
    if is_filtered:
        if df_filtered.shape[0] < min_sample_size:
            st.warning(
                f"⚠️ L'échantillon est trop petit ({df_filtered.shape[0]} observations)."
            )
            if st.button("❌ Supprimer le dernier filtre appliqué"):
                if len(filter_history) >= 2:
                    # On revient à l'état précédent
                    df_filtered, removed_feature = filter_history[-1]
                    used_features.pop()
                    st.info(f"Le filtre sur **{removed_feature}** a été retiré.")
                else:
                    st.info("Aucun filtre à retirer.")

    return df_filtered, used_features"""


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
