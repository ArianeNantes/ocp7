import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Wedge, Circle, FancyBboxPatch, Rectangle, FancyArrow
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import nbformat
import streamlit as st
import shap
import kaleido

print(nbformat.__version__)

DEFAULT_SHAP_COLOR_SAFE = (
    "#008bfb"  # Couleur par défaut de la catégorie 0 (rembousement OK pour SHAP)
)
DEFAULT_SHAP_COLOR_RISK = (
    "#ff0051"  # Couleur par défaut de la catégorie 1 (risque de défaut pour SHAP)
)

"""DEFAULT_COLOR_RISK = "#ca0020"
DEFAULT_COLOR_SAFE = "#92c5de"
DEFAULT_COLOR_LOANER = "#E0D01C"
"""
THRESHOLD_PROB = 0.48
DEFAULT_COLOR_LOANER = "#E0D01C"
DEFAULT_COLOR_RISK = "#7200CA"
DEFAULT_COLOR_SAFE = "#A1DE92"


####################################### STYLES
def plotly_style_title(
    title: str,
    level: int = 6,
    fontsize: int = 16,
    margin_bottom: float = 0.0,
    align="center",
):
    """
    Affiche un titre dans Streamlit avec un style visuellement similaire à celui de Plotly.
    La couleur est gérée par streamlit.

    Args:
        title (str): Le texte du titre à afficher.
        level (int): Le niveau de titre HTML (ex: 1 = h1, 2 = h2, 3 = h3...). Par défaut h3.
        fontsize (int): taille en px de la police utilisée pour le titre. Par défaut 16.
    """
    # [TODO] Pour une taille de police correcte gérée par streamlit -> niveau 6
    # Gérer la fontsize et mettre un niveau 3 ?

    st.markdown(
        # f"<h{level} style='font-family:Open Sans, sans-serif; font-size:{fontsize}px; "
        f"<h{level} style='font-family:Open Sans, sans-serif; "
        # Ne surtout pas mettre de couleur car on ne peut pas récupérer la couleur du fond de l'utilisateur.
        # En omettant la couleur, c'est streamlit qui choisit
        # f"font-weight:bold; color:{hex_color}; margin-bottom:{margin_bottom}em;'>"
        f"font-weight:bold; margin-bottom:{margin_bottom}em; "
        f"text-align:{align};'>"
        f"{title}"
        f"</h{level}>",
        unsafe_allow_html=True,
    )


####################################### COULEURS - CMAP
def darken_color(hex_color, blend_pct=0.7):
    """
    Fonce une couleur hexadécimale en la mélangeant avec du noir.
    blend_pct = 0.0 → couleur d'origine
    blend_pct = 1.0 → noir pur
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]

    r_blend = int(r * (1 - blend_pct))
    g_blend = int(g * (1 - blend_pct))
    b_blend = int(b * (1 - blend_pct))

    return f"#{r_blend:02x}{g_blend:02x}{b_blend:02x}"


def lighten_color(hex_color, blend_pct=0.7):
    """
    Eclaircit une couleur hexadécimale en la mélangeant avec du blanc.
    blend_pct = 0.0 → couleur d'origine
    blend_pct = 1.0 → blanc pur
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]

    r_blend = int(r + (255 - r) * blend_pct)
    g_blend = int(g + (255 - g) * blend_pct)
    b_blend = int(b + (255 - b) * blend_pct)

    return f"#{r_blend:02x}{g_blend:02x}{b_blend:02x}"


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


####################################### PLOTLY Univariés
def pie(
    df,
    client,
    feature,
    labels=[],
    # color_base=DEFAULT_COLOR_SAFE,
    # color_loaner=DEFAULT_COLOR_LOANER,
):

    # Comptage des catégories
    counts = df[feature].value_counts().reset_index()
    counts.columns = [feature, "count"]

    # Liste des labels et des valeurs
    labels = counts[feature].tolist()
    values = counts["count"].tolist()

    # Création des paramètres visuels
    colors = []
    pull = []
    line_colors = []
    line_widths = []
    customdata = []
    hovertemplates = []

    i = 0
    for category, count in zip(labels, values):
        if category == client[feature]:
            # colors.append(color_loaner)
            pull.append(0.1)
            line_colors.append("black")
            line_widths.append(1)
            customdata.append(client.name)
            hovertemplates.append(
                f"<b>{category}</b><br>%{{percent}}<br>👤 Client : {client.name}<extra></extra>"
            )
        else:

            # colors.append(lighten_color(color_base, blend_pct=0.2 * i))
            i += 1
            pull.append(0)
            line_colors.append("white")
            line_widths.append(2)
            customdata.append("")
            hovertemplates.append("<b>%{label}</b><br>%{percent}<extra></extra>")

    # Création du graphique
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            pull=pull,
            marker=dict(line=dict(color=line_colors, width=line_widths)),
            sort=False,
            textinfo="label+percent",
            customdata=customdata,
            hovertemplate=hovertemplates,
        )
    )

    fig.update_layout(
        # title="",
        height=500,
        margin=dict(t=50, b=50),
    )

    return fig


def histogram_univariate(
    df,
    client,
    numeric_feature,
    color_client=DEFAULT_COLOR_LOANER,
):
    # Donnée de l'emprunteur
    client_value = client[numeric_feature]

    fig = px.histogram(
        df,
        x=numeric_feature,
        nbins=50,
        histnorm="percent",
        opacity=0.6,
        # title=f"Distribution de {numeric_feature} par {category_feature}",
    )

    # Forcer l'apparition de la 'population' (première trace à apparaître dans la légende)
    fig.data[0].name = "Population"
    # fig.data[0].marker.color = "lightblue"  # ou autre couleur
    fig.data[0].showlegend = True

    # Ajouter une ligne verticale pour la moyenne
    mean_value = df[numeric_feature].mean()
    fig.add_shape(
        type="line",
        x0=mean_value,
        y0=0,
        x1=mean_value,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="darkgrey", width=2, dash="dash"),
        name="Moyenne",
    )
    # Ajouter une trace invisible pour avoir une légende "Moyenne"
    fig.add_trace(
        go.Scatter(
            x=[mean_value],
            y=[0],
            mode="lines",
            line=dict(color="darkgrey", width=2, dash="dash"),
            name="Moyenne",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    client_id = client.name
    text_to_display = ""
    # --- Ajouter le client sur la figure sous forme d'un point ---
    # Ici on place le point sur l'axe des X, à la base de l'histogramme.
    if client[numeric_feature] is not None:
        fig.add_trace(
            go.Scatter(
                x=[client[numeric_feature]],
                y=[0],  # placer à la base de l'histogramme
                # marker=dict(color=colors[2], size=14, symbol="circle"),
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=color_client,
                    symbol="circle",
                    # color="yellow",
                    line=dict(
                        color="black",
                        # color="darkgrey",
                        width=2,  # Contour noir pur le rendre visible sur fond clair
                    ),
                ),
                # marker=dict(size=14, symbol="circle"),
                name=f"Client {client_id}",
                text=[f"Client {client_id}"],
                # L'aspect de l'umprunteur n'a aucune transparence et 'écrase' les formes de derrière'
                # Fonctionne sur la forme et sur le texte qui l'accompagne
                opacity=1,
                textposition="top center",
                hovertemplate=(
                    f"<b>Client {client_id}</b><br>"
                    + f"{numeric_feature} = {client[numeric_feature]:.2f}<extra></extra>"
                ),
            )
        )

        if client[numeric_feature] <= mean_value:
            text_to_display = f"La valeur de {numeric_feature} pour le client {client_id} ({client[numeric_feature]:.2f}) est inférieure à la moyenne ({mean_value:.2f})."
        else:
            text_to_display = f"La valeur de {numeric_feature} pour le client {client_id} ({client[numeric_feature]:.2f}) est supérieure à la moyenne ({mean_value:.2f})."
    else:
        text_to_display = (
            f"La valeur de {numeric_feature} pour le client {client_id} est inconnue."
        )
    fig.update_layout(
        # title=f"Histogram de {numeric_feature} by {category_feature}",
        xaxis_title=numeric_feature,
        margin=dict(t=10),
    )

    return fig, text_to_display


def pie1(df, client, feature, labels=[], color_loaner=DEFAULT_COLOR_LOANER):
    # Exemple de DataFrame

    # Client spécifique
    client_gender = "M"  # ou 'F'

    # Comptage des catégories
    counts = df[feature].value_counts().reset_index()
    counts.columns = [feature, "count"]

    # Liste des labels et des valeurs
    labels = counts[feature]
    values = counts["count"]

    # Préparer les couleurs et l'effet "explosé"
    colors = []
    pull = []

    for category in labels:
        if category == client[feature]:
            colors.append(color_loaner)  # Couleur spécifique
            pull.append(0.15)  # Décalage de la portion

        else:
            colors.append("lightgray")
            pull.append(0)  # Pas de décalage

    # Création du graphique
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            pull=pull,
            sort=False,
            textinfo="label+percent",
        )
    )

    return fig


####################################### PLOTLY Bivariés


def plot_scatter(
    df,
    feature_x,
    feature_y,
    client,
    bar_position="top",
    colors={
        "color_safe": DEFAULT_COLOR_SAFE,
        "color_risk": DEFAULT_COLOR_RISK,
        "color_loaner": DEFAULT_COLOR_LOANER,
        "threshold_prob": THRESHOLD_PROB,
    },
):
    # fig = px.scatter(df, x=feature_x, y=feature_y)
    # fig["layout"].update(title=f"Relation entre {feature_x} et {feature_y}")

    color_neutral = "#ffffff"  # (blanc couleur neutre de la cmap au seuil)
    """colorscale = [
        [0.0, DEFAULT_COLOR_SAFE],
        # [0.5, DEFAULT_COLOR_SAFE],  # jusqu'au seuil, c'est encore la couleur safe
        [
            THRESHOLD_PROB,
            color_neutral,
        ],  # à partir du seuil, ça devient la couleur risque
        [1.0, DEFAULT_COLOR_RISK],
    ]"""

    colorscale = [
        [0.0, colors["color_safe"]],
        [colors["threshold_prob"], lighten_color(colors["color_safe"])],
        [colors["threshold_prob"], lighten_color(colors["color_risk"])],
        [1.0, colors["color_risk"]],
    ]

    # Pour mettre une colorbar à droite ou à gauche, il faut effectuer un décalage
    # car sinon elle se retrouve au dessus de la légende.
    # Or ces décalages dépendent de la DPI, donc préférer l'affichage horizontal (au dessus)
    if bar_position == "top":
        colorbar = dict(orientation="h")
    elif bar_position == "right":
        colorbar = (
            dict(
                title="PROBA",
                x=1.15,  # décale à droite hors de la légende
            ),
        )
    else:
        colorbar = (dict(x=0.1, xref="container", title=dict(text="PROBA")),)

    # On supprime les lignes contenant au moins une valeur manquante
    df_filtered = df[[feature_x, feature_y, "PROBA"]].dropna()
    fig = go.Figure(
        go.Scatter(
            x=df_filtered[feature_x],
            y=df_filtered[feature_y],
            mode="markers+text",
            name=f"Population",
            marker=dict(
                color=df["PROBA"],
                colorscale=colorscale,
                cmin=0,  # min possible
                cmax=1,  # max possible
                colorbar=dict(orientation="h"),
                showscale=True,
            ),
            # hovertemplate=(
            #    f"<b>Loaner {df.index}</b><br>"
            #    + f"{feature_x} = {df[feature_x].iloc[df.index]}<br>"
            #    + f"{feature_y} = {df[feature_y].iloc[df.index]}<extra></extra>"
            # ),
            customdata=df.index.to_numpy().reshape(-1, 1),
            hovertemplate=(
                # Pour personnaliser : Double accolade pour échapper les {} dans la fstring
                # car si on dit f"{feature_x} = %{x:.2f}",
                # le formattage %{x:.2f} se fait au moment de l’exécution Python,
                # ce qui déclenche une erreur car x n’est pas une variable Python, mais une variable Plotly (JavaScript)
                "<b>Client %{customdata[0]}</b><br>"
                + f"{feature_x} = %{{x:.2f}}<br>"
                + f"{feature_y} = %{{y:.2f}}<br>"
                + "PROBA = %{marker.color:.2f}<extra></extra>"
            ),
        )
    )

    # if loaner_id in df.index:
    client_id = client.name
    # loaner_x = df[feature_x].iloc[client_id]
    # loaner_y = df[feature_y].iloc[client_id]

    """if loaner_target == 1:
        color_loaner = colors["color_risk"]
    else:
        color_loaner = colors["color_safe"]"""

    # On ne trace le point représentant le client que si ses données sont connues
    text_to_display = ""
    if client[feature_x] is not None and client[feature_y] is not None:
        color_loaner = colors["color_loaner"]
        fig.add_trace(
            go.Scatter(
                x=[client[feature_x]],
                y=[client[feature_y]],
                mode="markers+text",
                # marker=dict(color=colors[2], size=14, symbol="circle"),
                marker=dict(
                    size=14,
                    symbol="circle",
                    # color="yellow",
                    color=color_loaner,
                    line=dict(
                        color="black",
                        # color="darkgrey",
                        width=2,  # Contour noir pur le rendre visible sur fond clair
                    ),
                ),
                name=f"Client {client_id}",
                # text= ne permet pas de spécifier une couleur de fond du texte, et il devient illisible sur fonc clair,
                # L'annotation ne convient pas non plus. On supprime le texte au dessus du point.
                # text=[f"Loaner {loaner_id}"],
                # L'aspect de l'umprunteur n'a aucune transparence et 'écrase' les formes de derrière'
                # Fonctionne sur la forme et sur le texte qui l'accompagne
                opacity=1,
                textposition="top center",
                hovertemplate=(
                    f"<b>Client {client_id}</b><br>"
                    + f"{feature_x} = {client[feature_x]:.2f}<br>"
                    + f"{feature_y} = {client[feature_y]:.2f}<extra></extra>"
                ),
            )
        )
        text_to_display = f"Pour le client {client_id} :\n"
        text_to_display += f"\t{feature_x} = {client[feature_x]}\n\t{feature_y} = {client[feature_y]}\n\tPROBA = {client['PROBA']:.2f}"
    # Si au moins l'une des données (x, y ou proba n'est pas connue pour le client)
    elif client[feature_x] is None and client[feature_y] is not None:
        text_to_display = (
            f"\tLa valeur de {feature_x} est inconnue pour le client {client_id}."
        )
    elif client[feature_x] is not None and client[feature_y] is None:
        text_to_display = (
            f"\tLa valeur de {feature_y} est inconnue pour le client {client_id}."
        )
    # Si les deux valeurs sont inconnues
    elif client[feature_x] is None and client[feature_y] is None:
        text_to_display = f"Les valeurs de {feature_x}  et de {feature_y} sont inconnues pour le client {client_id}."

    if bar_position == "top":
        margin_size = 0
    else:
        margin_size = 10
    fig.update_layout(
        # title=f"{feature_y} en fonction de {feature_x}",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        margin=dict(t=margin_size),
    )

    return fig, text_to_display


def histo(
    df_all,
    numeric_feature,
    loaner_id=0,
    category_feature="PREDICTION",
    colors=[DEFAULT_COLOR_SAFE, DEFAULT_COLOR_RISK],
):

    if category_feature in ["PREDICTION", "TARGET"]:
        category_order = [0, 1]
        color_map = {0: colors[0], 1: colors[1]}

    # Si la feature catégorielle n'est pas la prédiction, elle n'est pas représentative du risque de défaut et on laisse choisir plotly
    else:
        category_order = sorted(df_all[category_feature].dropna().unique())
        color_map = None

    # On récupère les données de l'emprunteur pour le comparer aau reste de la population
    # loaner_x = df_all.loc[loaner_id, category_feature]
    loaner_value = df_all.loc[loaner_id, numeric_feature]

    # --- Histogramme avec Plotly Express ---
    fig = px.histogram(
        df_all,
        x=numeric_feature,
        color=category_feature,
        color_discrete_map=color_map if color_map else {},
        category_orders={category_feature: category_order},
        nbins=50,
        histnorm="percent",
        barmode="group",
        opacity=0.6,
        title=f"Distribution de {numeric_feature} par {category_feature}",
    )

    # --- Ajouter une ligne verticale avec Scatter pour la légende ---
    # On n'a pas accès comme dans matplolib aux patches, ni à ax.get_ylim() ni à...
    # => soit on ajoute une ligne verticale simple et elle ne figure pas dans la légende,
    # soit on trace un scatter mais il faut calculer les valeurs min-max sans quoi pas de dessin
    # Donc il faut calculer le y_max (grossièrement au moins)
    # --- Déterminer une hauteur raisonnable pour la ligne verticale ---
    # Cela récupère le maximum actuel de l'axe Y pour tracer une ligne entière
    fig.update_traces()  # assure la mise à jour des traces
    fig.update_layout(bargap=0.05)
    y_max = 0
    for trace in fig.data:
        if isinstance(trace, go.Histogram):
            y_max = max(y_max, max(trace.y) if trace.y is not None else 0)

    """y_max = max(
        max(trace.y)
        for trace in fig.dataHistogram
        if isinstance(trace, go.Histogram) and trace.y is not None
        tout est à None
    )"""

    # On force le calcul complet des axes et traces, puis on récupère y_max (nécessite l'install de kaleido et ne fonctionne pas mieux)
    """fig = fig.full_figure_for_development(warn=False)
    # Récupérer ymax parmi toutes les traces d’histogramme
    y_max = max(
        max(t.y) if t.y is not None else 0
        for t in fig.data
        if isinstance(t, go.Histogram)
    )"""

    # calcul de y_max très arbitraire :
    y_max = df_all[numeric_feature].shape[0] * 0.1
    # st.write(trace.type, trace.name, trace.y)
    st.write("y_max", y_max)

    fig.add_trace(
        go.Scatter(
            x=[loaner_value, loaner_value],
            # y=[0, df_all[numeric_feature].value_counts().max()],
            y=[0, y_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name=f"Loaner ({loaner_value:.2f})",
            showlegend=True,
        )
    )

    # --- (Facultatif) Annotation textuelle --- au lieu d'une ligne verticale, met une flèeche
    """fig.add_annotation(
        x=loaner_value,
        y=0,
        text=f"Loaner: {loaner_value:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
    )"""

    # --- Mettre à jour les titres ---
    fig.update_layout(
        title=f"Histogramme de {numeric_feature} selon {category_feature}",
        xaxis_title=numeric_feature,
        yaxis_title="Nombre de clients",
    )
    return fig


def histogram_by_categorical(
    df,
    client,
    numeric_feature,
    category_feature="PREDICTION",
    # colors=[DEFAULT_COLOR_SAFE, DEFAULT_COLOR_RISK, DEFAULT_COLOR_LOANER],
    colors=[DEFAULT_COLOR_SAFE, DEFAULT_COLOR_RISK, DEFAULT_COLOR_LOANER],
):
    # On récupère les données de l'emprunteur pour le comparer aau reste de la population
    # loaner_category = df_all.loc[loaner_id, category_feature]
    # loaner_numeric = df_all.loc[loaner_id, numeric_feature]
    client_category = client[category_feature]
    client_numeric = client[numeric_feature]
    """if df_all["PREDICTION"].iloc[loaner_id] == 0:
        color_loaner = DEFAULT_COLOR_SAFE
    else:
        color_loaner = DEFAULT_COLOR_RISK"""

    color_client = colors[2]

    if category_feature in ["PREDICTION", "TARGET"]:
        category_order = [0, 1]
        color_map = {0: colors[0], 1: colors[1]}

    # Si la feature catégorielle n'est pas la prédiction, elle n'est pas représentative du risque de défaut et on laisse choisir plotly
    else:
        category_order = sorted(df[category_feature].dropna().unique())
        color_map = None

    # On récupère les données de l'emprunteur pour le comparer aau reste de la population
    # loaner_x = df_all.loc[loaner_id, category_feature]
    # loaner_value = df_all.loc[loaner_id, numeric_feature]

    fig = px.histogram(
        df,
        x=numeric_feature,
        color=category_feature,
        color_discrete_map=color_map if color_map else {},
        category_orders={category_feature: category_order},
        nbins=50,
        # barmode="overlay",
        barmode="group",
        histnorm="percent",
        opacity=0.6,
        # title=f"Distribution de {numeric_feature} par {category_feature}",
    )

    client_id = client.name
    # --- Ajouter le client sur la figure sous forme d'un point ---
    # Ici on place le point sur l'axe des X, à la base de l'histogramme.
    if client[numeric_feature] is not None:
        fig.add_trace(
            go.Scatter(
                x=[client[numeric_feature]],
                y=[0],  # placer à la base de l'histogramme
                # marker=dict(color=colors[2], size=14, symbol="circle"),
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=color_client,
                    symbol="circle",
                    # color="yellow",
                    line=dict(
                        color="black",
                        # color="darkgrey",
                        width=2,  # Contour noir pur le rendre visible sur fond clair
                    ),
                ),
                # marker=dict(size=14, symbol="circle"),
                name=f"Client {client_id}",
                text=[f"Client {client_id}"],
                # L'aspect de l'umprunteur n'a aucune transparence et 'écrase' les formes de derrière'
                # Fonctionne sur la forme et sur le texte qui l'accompagne
                opacity=1,
                textposition="top center",
                hovertemplate=(
                    f"<b>Client {client_id}</b><br>"
                    + f"{category_feature} = {client[category_feature]}<br>"
                    + f"{numeric_feature} = {client[numeric_feature]:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        # title=f"Histogram de {numeric_feature} by {category_feature}",
        xaxis_title=numeric_feature,
        margin=dict(t=10),
    )

    return fig


def boxplot_by_categorical(
    df,
    client,
    numeric_feature,
    category_feature="PREDICTION",
    colors=[DEFAULT_COLOR_SAFE, DEFAULT_COLOR_RISK, DEFAULT_COLOR_LOANER],
):

    # Les boxplots sont croisées par une feature catégorielle, par défaut il s'agit de la TARGET prédite (0 ou 1)
    # La catégorie 0 est la catégorie des clients fiables (le client pourra rembourser),
    # La catégorie 1 est la catégorie des clients risqués (le client va faire défaut)
    # Si la feature choisie est la prédiction, il nous faut reprendre les couleurs choisies pour représenter
    # le risque de défaut (colors_safe et color_risk)
    # [TODO] Changer prediction -> TARGET
    if category_feature in ["PREDICTION", "TARGET"]:
        category_order = [0, 1]
        color_map = {0: colors[0], 1: colors[1]}

    # Si la feature catégorielle n'est pas la prédiction, elle n'est pas représentative du risque de défaut et on laisse choisir plotly
    else:
        category_order = sorted(df[category_feature].dropna().unique())
        color_map = None

    # On récupère les données de l'emprunteur pour le comparer aau reste de la population
    # loaner_x = df_all.loc[loaner_id, category_feature]
    # loaner_y = df_all.loc[loaner_id, numeric_feature]
    """color_loaner = (
        DEFAULT_COLOR_SAFE
        if df_all.loc[loaner_id, "PREDICTION"] == 0
        else DEFAULT_COLOR_RISK
    )"""
    color_client = colors[2]

    # On montre toujours les outliers
    show_outliers = True

    # On crèe la figuere avec les boxplots sur l'ensemble de la population
    fig = px.box(
        df,
        x=category_feature,
        y=numeric_feature,
        color=category_feature,
        color_discrete_map=color_map if color_map else {},
        category_orders={category_feature: category_order},
        points="outliers" if show_outliers else False,
        title=None,
    )

    # A l'origine, client est 1 DataFrame d'une ligne et il est indexé par l'ID
    # On squeeze, du coup l'index est le name de la série résultante
    client_id = client.name

    # On ajoute le l'emprunteur particulier sur les boxplots initiales
    if client[category_feature] is not None and client[numeric_feature] is not None:
        fig.add_scatter(
            x=[client[category_feature]],
            y=[client[numeric_feature]],
            mode="markers+text",
            # marker=dict(color="yellow", size=12),
            # marker=dict(size=12, color=color_loaner, ),
            marker=dict(
                size=14,
                symbol="circle",
                # color="yellow",
                color=color_client,
                line=dict(
                    color="black",
                    # color="darkgrey",
                    width=2,  # Contour noir pur le rendre visible sur fond clair
                ),
            ),
            text=[f"Client {client_id}"],
            # L'aspect de l'umprunteur n'a aucune transparence et 'écrase' les formes de derrière'
            # Fonctionne sur la forme et sur le texte qui l'accompagne
            opacity=1,
            textposition="top center",
            name=f"Client {client_id}",
            # Il est nécessaire de redéfinir le on_hoover pour qu'à l'approche du pointeur on ait
            # les coordonnées correctes de l'emprunteur
            hovertemplate=(
                f"<b>Client {client_id}</b><br>"
                + f"{category_feature} = {client[category_feature]}<br>"
                + f"{numeric_feature} = {client[numeric_feature]:.2f}<extra></extra>"
            ),
        )

    # On met à jour l'affichage de la figure pour prendre en compte les personalisations
    fig.update_layout(
        # title=f"Boxplot de {numeric_feature} selon {category_feature}",
        yaxis_title=numeric_feature,
        xaxis_title=category_feature,
        margin=dict(t=0),
    )
    return fig


def global_importance_barh(df_importance, max_display=10, plot_others=False, title=""):
    df_sorted = df_importance.sort_values("importance", ascending=False)

    # Données principales
    top_features = df_sorted["feature"][:max_display].tolist()
    top_importances = df_sorted["importance"][:max_display].tolist()

    # Ajout de la barre "Autres"
    if plot_others:
        other_importance = df_sorted["importance"][max_display:].sum()
        top_features.append("Somme des Autres caractéristiques")
        top_importances.append(other_importance)
        colors = ["steelblue"] * max_display + ["lightgray"]
    else:
        colors = ["steelblue"] * max_display

    # Création du graphique Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_importances[::-1],
            y=top_features[::-1],
            orientation="h",
            marker=dict(color=colors[::-1]),
            text=[f"{imp:.4f}" for imp in top_importances[::-1]],
            textposition="auto",
        )
    )
    if not title:
        title = f"Top {max_display} des variables les plus importantes dans le calcul du risque de défaut"
    fig.update_layout(
        height=400 + 20 * len(top_features),
        # title=title,
        xaxis_title="Importance moyenne (|SHAP value|)",
        yaxis_title="",
        margin=dict(l=100, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        visible=True,
        showline=True,
        linewidth=1,
        showticklabels=True,
        ticklen=6,
        tickwidth=2,
    )
    fig.update_yaxes(visible=True, showline=False)

    return fig


def global_importance_barh2(df_importance, max_display=10, plot_others=False, title=""):
    df_plot = df_importance.sort_values("importance", ascending=False).head(max_display)

    # Crée le graphique
    fig = px.bar(
        df_plot,
        x="importance",
        y="feature",
        orientation="h",
        color="impact_direction",
        color_discrete_map={
            "Tire vers défaut": "crimson",
            "Tire vers remboursement": "royalblue",
        },
        labels={
            "importance_abs": "Importance moyenne (|SHAP value|)",
            "feature": "Variable",
            "impact_direction": "Effet moyen",
        },
        title="Top 20 des variables par importance globale SHAP\n(couleur = direction de l'effet moyen)",
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showline=True, linecolor="black", ticks="outside"),
        bargap=0.3,
    )

    return fig


####################################### SHAP Plots
def shap_summary_dot(
    shap_values_global, colors=["#008bfb", "#ff0051"], max_display=10, title=""
):
    # On crée une cmap à partir des deux couleurs :
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [colors[0], colors[1]]
    )

    fig, ax = plt.subplots()
    # Personnalisation du titre
    if title:
        ax.set_title(title)

    # On trace le plot shap sans le montrer
    shap.summary_plot(
        shap_values_global,
        plot_type="dot",
        max_display=max_display,
        show=False,
    )

    # On retrace avec matplotlib le graphique en changeant la cmap
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(custom_cmap)

    # On montre le graphique une fois les corrections effectuées
    plt.show()
    return fig


def make_color_list(start_color, end_color, n=10):
    cmap = LinearSegmentedColormap.from_list("custom", [start_color, end_color])
    return [cmap(i) for i in np.linspace(0, 1, n)]


def shap_force(shap_values, X_client, colors=["#008bfb", "#ff0051"]):

    force_plot = shap.force_plot(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values[0],
        features=X_client,
        # Attention des features qui tirent vers le refus sont affichées à gauche,
        # C'est l'inverse de la jauge et du waterfall
        plot_cmap=[colors[1], colors[0]],
    )

    # Pour l'affichage, on convertira en HTML avec la fonction st_shap
    return force_plot


def shap_waterfall(
    shap_values,
    colors=["#008bfb", "#ff0051"],
    max_display=10,
    title="",
):

    # Default SHAP colors
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    # Custom colors
    positive_color = colors[1]
    negative_color = colors[0]

    fig, ax = plt.subplots()
    # Personnalisation du titre
    if title:
        ax.set_title(title)
        # fig.suptitle(title)

    # Trace le plot waterfall avec les couleurs par défaut sans le montrer
    shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)

    # Change les couleurs des formes dessinées par SHAP avec les couleurs par défaut
    # Pour un Waterfall, les formes ('artists') sont des FancyArrow
    # On récupère donc toutes les formes et on leur applique les nouvelles couleurs
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if isinstance(fcc, FancyArrow):
                if cm.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                    fcc.set_facecolor(positive_color)
                elif cm.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                    fcc.set_color(negative_color)
            elif isinstance(fcc, plt.Text):
                if cm.colors.to_hex(fcc.get_color()) == default_pos_color:
                    fcc.set_color(positive_color)
                elif cm.colors.to_hex(fcc.get_color()) == default_neg_color:
                    fcc.set_color(negative_color)

    return fig


####################################### GAUGE Plots
def gauge_default(proba, threshold_prob=0.6):

    # Jauge
    # Probabilité prédite (à ajuster dynamiquement)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba,
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "darkgray"},
                "bar": {"color": "black", "thickness": 0.35},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, threshold_prob], "color": "lightgreen"},
                    {"range": [threshold_prob, 1], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": proba,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
    return fig


# Trace une jauge par matplotlib en demi-cercle avec une aiguille au milieu
def gauge_plt(
    proba,
    threshold_prob=0.6,
    title="",
    colors=["#008bfb", "#ff0051"],
):
    size = 1
    thickness = 0.4  # Epaisseur du donut
    # color_needle = "darkgrey"
    color_needle = "black"

    if proba > threshold_prob:
        decision = "Refusé"
    else:
        decision = "Accordé"

    # Création de la figure
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.015, 1.1)  # Limité en bas à 0 = demi-cercle parfaitement horizontal
    ax.axis("off")

    # Fond vert (gauche)
    wedge_left = Wedge(
        center=(0, 0),
        r=size,
        theta1=180 * (1 - threshold_prob),
        theta2=0,
        width=thickness,
        facecolor=colors[0],
        edgecolor="darkgrey",
    )
    ax.add_patch(wedge_left)

    # Fond rouge (droite)
    wedge_right = Wedge(
        center=(0, 0),
        r=size,
        theta1=190,
        theta2=180 * (1 - threshold_prob),
        width=thickness,
        facecolor=colors[1],
        edgecolor="darkgrey",
    )
    ax.add_patch(wedge_right)

    # Aiguille
    angle = 180 * (1 - proba)
    angle_rad = np.radians(angle)
    needle_length = size - thickness / 3
    x = needle_length * np.cos(angle_rad)
    y = needle_length * np.sin(angle_rad)
    # ax.plot([0, x], [0, y], color="black", linewidth=3)

    boxstyle = "circle"
    needle_anotation = f"{proba:.2f}"

    plt.annotate(
        needle_anotation,
        xytext=(0, 0.02),
        xy=(x, y),
        arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
        bbox=dict(
            boxstyle=boxstyle,
            facecolor=color_needle,
            linewidth=2.0,
        ),
        fontsize=20,
        color="white",
        ha="center",
    )

    # Centre de l’aiguille
    # ax.add_patch(Circle((0, 0), 0.03, color="black", zorder=5))

    ## Ticks
    # Major ticks (tous les 0.2) avec labels
    major_ticks = np.arange(0, 1.01, 0.2)
    minor_ticks = np.arange(0, 1.01, 0.1)

    # Paramètres des ticks
    tick_length_major = 0.03
    tick_length_minor = 0.015
    tick_start_r = size  # à partir du bord extérieur
    label_r = size - 0.09  # position des labels (plus proche du centre)

    # Minor ticks (sans label)
    for val in minor_ticks:
        angle_deg = 180 * (1 - val)
        angle_rad = np.radians(angle_deg)

        x_start = tick_start_r * np.cos(angle_rad)
        y_start = tick_start_r * np.sin(angle_rad)
        x_end = (tick_start_r - tick_length_minor) * np.cos(angle_rad)
        y_end = (tick_start_r - tick_length_minor) * np.sin(angle_rad)

        ax.plot([x_start, x_end], [y_start, y_end], color="black", lw=1)

    # Major ticks (avec label)
    for val in major_ticks:
        angle_deg = 180 * (1 - val)
        angle_rad = np.radians(angle_deg)

        x_start = tick_start_r * np.cos(angle_rad)
        y_start = tick_start_r * np.sin(angle_rad)
        x_end = (tick_start_r - tick_length_major) * np.cos(angle_rad)
        y_end = (tick_start_r - tick_length_major) * np.sin(angle_rad)

        ax.plot([x_start, x_end], [y_start, y_end], color="black", lw=2)

        # Label légèrement à l’intérieur
        x_label = label_r * np.cos(angle_rad)
        y_label = label_r * np.sin(angle_rad)
        ax.text(x_label, y_label, f"{val:.1f}", ha="center", va="center", fontsize=7)

    if title:
        # if not text_in_needle:
        #    ax.set_title(f"Crédit {decision}", fontsize=20)
        ax.set_title(title, fontsize=20)
    return fig


import math


def create_needle_path(score, center_x=0.5, center_y=0.5, length=0.3, base_width=0.02):
    """
    Crée un path SVG pour dessiner une aiguille sur une jauge semi-circulaire.

    - score : entre 0 et 1
    - center_x, center_y : centre de la jauge (normalisé entre 0 et 1)
    - length : longueur de l'aiguille
    - base_width : largeur de la base de l'aiguille (en coord paper)
    """

    # Convertit le score en angle (demi-cercle : pi radians)
    angle = (1 - score) * math.pi  # 1 = gauche (π), 0 = droite (0)

    # Pointe de l’aiguille
    x_tip = center_x + length * math.cos(angle)
    y_tip = center_y + length * math.sin(angle)

    # Base gauche et droite
    x_base_left = center_x - base_width / 2
    x_base_right = center_x + base_width / 2
    y_base = center_y

    # Crée le chemin SVG
    path = f"M {x_base_left} {y_base} L {x_tip} {y_tip} L {x_base_right} {y_base} Z"
    return path


def gauge_plotly(proba, threshold_prob=0.6):
    base_chart = {
        "values": [40, 10, 10, 10, 10, 10, 10],
        "labels": ["-", "0", "20", "40", "60", "80", "100"],
        # "domain": {"x": [0, 0.48]},
        "domain": {"x": [0, 1], "y": [0, 1]},
        "marker": {
            "colors": [
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
            ],
            "line": {"width": 1},
        },
        "name": "Gauge",
        "hole": 0.4,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 108,
        "showlegend": False,
        "hoverinfo": "none",
        "textinfo": "label",
        "textposition": "outside",
    }

    # Now we will superimpose our semi-circular meter on top of this.
    # For that, we will also use 6 sections, but one of them will be invisible to form the
    # lower half (colored same as the background).
    meter_chart = {
        "values": [50, 50 * (threshold_prob), 50 * (1 - threshold_prob)],
        "labels": ["", "Accordé", "Refusé"],
        "marker": {
            "colors": [
                "rgb(255, 255, 255)",
                "lightgreen",
                "lightcoral",
            ]
        },
        # "domain": {"x": [0, 0.48]},
        "domain": {"x": [0, 1], "y": [0, 1]},
        "name": "Gauge",
        "hole": 0.4,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 90,
        "showlegend": False,
        "textinfo": "label",
        "textposition": "inside",
        "hoverinfo": "none",
    }

    path = create_needle_path(proba)

    needle_shape = {
        "type": "path",
        "path": path,
        "fillcolor": "rgba(44, 160, 101, 0.7)",
        "line": {"width": 1},
        "xref": "paper",
        "yref": "paper",
    }

    layout = {
        "xaxis": {
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis": {
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        "shapes": [needle_shape],
        "annotations": [
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.23,
                "y": 0.45,
                "text": "50",
                "showarrow": False,
            }
        ],
    }

    layout["shapes"] = [needle_shape]

    # we don't want the boundary now
    base_chart["marker"]["line"]["width"] = 0

    fig = {"data": [base_chart, meter_chart], "layout": layout}
    layout["height"] = 800  # ou la hauteur souhaitée
    layout["width"] = 700  # optionnel

    final_fig = go.Figure({"data": [base_chart, meter_chart], "layout": layout})
    # final_fig.update_layout(height=300)

    # final_fig.show(renderer="browser")

    return fig


def gauge_plt2(proba, threshold_prob=0.6):
    size = 1
    thickness = 0.4  # Epaisseur du donut

    # Création de la figure
    """fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.015, 1.1)  # Limité en bas à 0 = demi-cercle parfaitement horizontal
    ax.axis("off")"""

    colors = [
        "#4dab6d",
        "#72c66e",
        "#c1da64",
        "#f6ee54",
        "#fabd57",
        "#f36d54",
        "#ee4d55",
    ]
    values = [100, 80, 60, 40, 20, 0]
    x_axis_vals = [0, 0.44, 0.88, 1.32, 1.76]

    fig = plt.figure(figsize=(18, 18))

    ax = fig.add_subplot(projection="polar")

    ax.bar(
        x=[0, 0.44],
        width=0.5,
        height=0.5,
        bottom=2,
        linewidth=3,
        edgecolor="white",
        color=colors[:2],
        align="edge",
    )

    plt.annotate(
        "Accordé",
        xy=(0.5, 2.1),
        rotation=-75,
        color="white",
        fontweight="bold",
    )

    plt.annotate(
        "Foundational", xy=(2.1, 2.25), rotation=20, color="white", fontweight="bold"
    )

    for loc, val in zip([0, 0.44, 0.88, 1.32, 1.76, 3.14], values):
        plt.annotate(val, xy=(loc, 2.5), ha="right" if val <= 20 else "left")

    plt.annotate(
        "50",
        xytext=(0, 0),
        xy=(1.1, 2.0),
        arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
        bbox=dict(
            boxstyle="circle",
            facecolor="black",
            linewidth=2.0,
        ),
        fontsize=45,
        color="white",
        ha="center",
    )

    plt.title(
        "Performance Gauge Chart", loc="center", pad=20, fontsize=35, fontweight="bold"
    )

    ax.set_axis_off()

    return fig
