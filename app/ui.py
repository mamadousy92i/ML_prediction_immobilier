"""
Gradio UI — Immo Predictor Interface.
Provides a user-friendly web interface for regression and classification predictions,
with integrated charts and visualizations.
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging

from app.predictor import load_models, predict_regression, predict_classification, get_store

logger = logging.getLogger(__name__)

# Ensure models are loaded
load_models()

# Neighborhood & HouseStyle lists 
NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
    "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
    "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes",
    "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer",
    "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]

HOUSE_STYLES = [
    "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin",
    "2.5Unf", "SFoyer", "SLvl",
]

# Chart helpers ─

def create_regression_comparison_chart(data: dict) -> plt.Figure:
    """Compare predictions from both regression models."""
    try:
        price_rf = predict_regression(data, model_name="random_forest")
        price_dt = predict_regression(data, model_name="decision_tree")
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Random Forest", "Decision Tree"]
    prices = [price_rf, price_dt]
    colors = ["#2563eb", "#f59e0b"]

    bars = ax.bar(models, prices, color=colors, width=0.5, edgecolor="white", linewidth=1.2,
                  label=[f"Random Forest: ${price_rf:,.0f}", f"Decision Tree: ${price_dt:,.0f}"])
    for bar, price in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"${price:,.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_xlabel("Modele")
    ax.set_ylabel("Prix predit (USD)")
    ax.set_title("Comparaison des predictions par modele")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    # Add top margin so price labels are not cut off
    y_max = max(prices) * 1.15
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def create_feature_importance_chart(data: dict) -> plt.Figure:
    """Show a horizontal bar chart of the input feature values (normalized)."""
    numeric_keys = [k for k, v in data.items() if isinstance(v, (int, float))]
    values = np.array([float(data[k]) for k in numeric_keys])

    # Normalize to 0-1
    max_val = values.max() if values.max() != 0 else 1
    normalized = values / max_val

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(numeric_keys))
    colors = plt.cm.Blues(0.3 + 0.7 * normalized)

    ax.barh(y_pos, normalized, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(numeric_keys, fontsize=10)
    ax.set_xlabel("Valeur normalisee (0 a 1)")
    ax.set_ylabel("Caracteristique")
    ax.set_title("Profil du bien immobilier")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Add value labels on bars
    for i, (val, norm) in enumerate(zip(values, normalized)):
        ax.text(norm + 0.02, i, f"{val:,.0f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def create_classification_comparison_chart(data: dict) -> plt.Figure:
    """Show prediction from both classification models as a grouped display."""
    try:
        label_rf, enc_rf = predict_classification(data, model_name="random_forest")
        label_svm, enc_svm = predict_classification(data, model_name="svm")
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    models = ["Random Forest", "SVM"]
    labels = [label_rf, label_svm]
    encoded = [enc_rf, enc_svm]
    colors = ["#2563eb", "#10b981"]

    bars = ax.bar(models, [1, 1], color=colors, width=0.4, edgecolor="white", linewidth=1.2,
                  label=[f"Random Forest: {label_rf}", f"SVM: {label_svm}"])
    for bar, label, enc in zip(bars, labels, encoded):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5,
                f"{label}\n(code: {enc})", ha="center", va="center",
                fontweight="bold", fontsize=13, color="white")

    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Modele")
    ax.set_title("Comparaison des classifications par modele")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# Regression Prediction

def regression_ui(
    GrLivArea, TotalBsmtSF, LotArea, BedroomAbvGr, FullBath,
    TotRmsAbvGrd, OverallQual, OverallCond, YearBuilt, YearRemodAdd,
    Neighborhood, GarageCars, GarageArea, PoolArea, Fireplaces, model_name
):
    try:
        data = {
            "GrLivArea": float(GrLivArea),
            "TotalBsmtSF": float(TotalBsmtSF),
            "LotArea": float(LotArea),
            "BedroomAbvGr": int(BedroomAbvGr),
            "FullBath": int(FullBath),
            "TotRmsAbvGrd": int(TotRmsAbvGrd),
            "OverallQual": int(OverallQual),
            "OverallCond": int(OverallCond),
            "YearBuilt": int(YearBuilt),
            "YearRemodAdd": int(YearRemodAdd),
            "Neighborhood": Neighborhood,
            "GarageCars": int(GarageCars),
            "GarageArea": float(GarageArea),
            "PoolArea": float(PoolArea),
            "Fireplaces": int(Fireplaces),
        }
        price = predict_regression(data, model_name=model_name)
        result_text = f"Prix predit : ${price:,.2f} USD\nModele utilise : {model_name}"

        comparison_chart = create_regression_comparison_chart(data)
        feature_chart = create_feature_importance_chart(data)

        return result_text, comparison_chart, feature_chart
    except Exception as e:
        return f"Erreur : {str(e)}", None, None


# Classification Prediction 
def classification_ui(
    GrLivArea, TotRmsAbvGrd, OverallQual, YearBuilt,
    GarageCars, Neighborhood, HouseStyle, model_name
):
    try:
        data = {
            "GrLivArea": float(GrLivArea),
            "TotRmsAbvGrd": int(TotRmsAbvGrd),
            "OverallQual": int(OverallQual),
            "YearBuilt": int(YearBuilt),
            "GarageCars": int(GarageCars),
            "Neighborhood": Neighborhood,
            "HouseStyle": HouseStyle,
        }
        label, encoded = predict_classification(data, model_name=model_name)
        result_text = f"Type predit : {label}\nCode : {encoded}\nModele utilise : {model_name}"

        comparison_chart = create_classification_comparison_chart(data)
        feature_chart = create_feature_importance_chart(data)

        return result_text, comparison_chart, feature_chart
    except Exception as e:
        return f"Erreur : {str(e)}", None, None


# Build Gradio Interface 
CUSTOM_CSS = """
.gradio-container > footer {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: auto !important;
    z-index: 1000 !important;
    background: var(--background-fill-primary) !important;
    border-bottom: 1px solid var(--border-color-primary) !important;
    padding: 6px 16px !important;
}
.gradio-container > .main {
    margin-top: 40px !important;
}
"""

with gr.Blocks(
    title="Immo Predictor",
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
        # Immo Predictor
        ### Prediction immobiliere par Machine Learning
        Realise par **Mamadou Sy** | [GitHub](https://github.com/mamadousy92i) | [API Documentation](/docs)

        Predisez le **prix de vente** ou classifiez le **type de batiment** a partir des caracteristiques du bien.
        """
    )

    with gr.Tabs():
        # ── Tab 1: Regression ──
        with gr.TabItem("Prediction de Prix (Regression)"):
            gr.Markdown("### Entrez les caracteristiques du bien immobilier")

            with gr.Row():
                with gr.Column():
                    reg_GrLivArea = gr.Number(label="Surface habitable (sq ft)", value=1500, minimum=1)
                    reg_TotalBsmtSF = gr.Number(label="Surface sous-sol (sq ft)", value=800, minimum=0)
                    reg_LotArea = gr.Number(label="Surface du terrain (sq ft)", value=9000, minimum=1)
                    reg_BedroomAbvGr = gr.Slider(label="Chambres", minimum=0, maximum=10, step=1, value=3)
                    reg_FullBath = gr.Slider(label="Salles de bain", minimum=0, maximum=5, step=1, value=2)

                with gr.Column():
                    reg_TotRmsAbvGrd = gr.Slider(label="Total pieces (hors sous-sol)", minimum=1, maximum=15, step=1, value=7)
                    reg_OverallQual = gr.Slider(label="Qualite generale (1-10)", minimum=1, maximum=10, step=1, value=7)
                    reg_OverallCond = gr.Slider(label="Condition generale (1-10)", minimum=1, maximum=10, step=1, value=5)
                    reg_YearBuilt = gr.Number(label="Annee de construction", value=2000, minimum=1800, maximum=2030)
                    reg_YearRemodAdd = gr.Number(label="Annee de renovation", value=2005, minimum=1800, maximum=2030)

                with gr.Column():
                    reg_Neighborhood = gr.Dropdown(label="Quartier", choices=NEIGHBORHOODS, value="NAmes")
                    reg_GarageCars = gr.Slider(label="Places de garage", minimum=0, maximum=5, step=1, value=2)
                    reg_GarageArea = gr.Number(label="Surface garage (sq ft)", value=500, minimum=0)
                    reg_PoolArea = gr.Number(label="Surface piscine (sq ft)", value=0, minimum=0)
                    reg_Fireplaces = gr.Slider(label="Cheminees", minimum=0, maximum=5, step=1, value=1)

            reg_model = gr.Radio(
                label="Modele de regression",
                choices=["random_forest", "decision_tree"],
                value="random_forest",
            )

            reg_btn = gr.Button("Predire le prix", variant="primary", size="lg")
            reg_output = gr.Textbox(label="Resultat", lines=3, interactive=False)

            reg_comparison_plot = gr.Plot(label="Comparaison des modeles")
            reg_feature_plot = gr.Plot(label="Profil du bien")

            reg_btn.click(
                fn=regression_ui,
                inputs=[
                    reg_GrLivArea, reg_TotalBsmtSF, reg_LotArea, reg_BedroomAbvGr,
                    reg_FullBath, reg_TotRmsAbvGrd, reg_OverallQual, reg_OverallCond,
                    reg_YearBuilt, reg_YearRemodAdd, reg_Neighborhood, reg_GarageCars,
                    reg_GarageArea, reg_PoolArea, reg_Fireplaces, reg_model,
                ],
                outputs=[reg_output, reg_comparison_plot, reg_feature_plot],
            )

        # ── Tab 2: Classification ──
        with gr.TabItem("Classification du Type de Bien"):
            gr.Markdown("### Entrez les caracteristiques du bien immobilier")

            with gr.Row():
                with gr.Column():
                    clf_GrLivArea = gr.Number(label="Surface habitable (sq ft)", value=1500, minimum=1)
                    clf_TotRmsAbvGrd = gr.Slider(label="Total pieces (hors sous-sol)", minimum=1, maximum=15, step=1, value=7)
                    clf_OverallQual = gr.Slider(label="Qualite generale (1-10)", minimum=1, maximum=10, step=1, value=7)
                    clf_YearBuilt = gr.Number(label="Annee de construction", value=2000, minimum=1800, maximum=2030)

                with gr.Column():
                    clf_GarageCars = gr.Slider(label="Places de garage", minimum=0, maximum=5, step=1, value=2)
                    clf_Neighborhood = gr.Dropdown(label="Quartier", choices=NEIGHBORHOODS, value="NAmes")
                    clf_HouseStyle = gr.Dropdown(label="Style de la maison", choices=HOUSE_STYLES, value="1Story")

            clf_model = gr.Radio(
                label="Modele de classification",
                choices=["random_forest", "svm"],
                value="random_forest",
            )

            clf_btn = gr.Button("Classifier le type", variant="primary", size="lg")
            clf_output = gr.Textbox(label="Resultat", lines=3, interactive=False)

            clf_comparison_plot = gr.Plot(label="Comparaison des modeles")
            gr.Markdown(
                """
                **Legende BldgType** : 1Fam (Maison individuelle) | 2FmCon (Conversion 2 familles) | Duplx (Duplex) | TwnhsE (Maison de ville) | TwnhsI (Maison de ville interieure)
                """
            )
            clf_feature_plot = gr.Plot(label="Profil du bien")

            clf_btn.click(
                fn=classification_ui,
                inputs=[
                    clf_GrLivArea, clf_TotRmsAbvGrd, clf_OverallQual, clf_YearBuilt,
                    clf_GarageCars, clf_Neighborhood, clf_HouseStyle, clf_model,
                ],
                outputs=[clf_output, clf_comparison_plot, clf_feature_plot],
            )



if __name__ == "__main__":
    demo.launch(server_port=7861)
