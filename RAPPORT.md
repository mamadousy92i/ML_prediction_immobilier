# Rapport de Projet — Immo Predictor

**Auteur** : Mamadou Sy  
**Date** : Fevrier 2026  
**GitHub** : [github.com/mamadousy92i](https://github.com/mamadousy92i)  
**Demo** : [lucifer92i-immo-predictor.hf.space](https://lucifer92i-immo-predictor.hf.space)

---

## 1. Introduction et objectif

Ce projet porte sur la creation d'un systeme de prediction immobiliere base sur le Machine Learning.
A partir du jeu de donnees **House Prices** (Ames, Iowa), deux taches de prediction ont ete traitees :

1. **Regression** : predire le prix de vente d'un bien immobilier (`SalePrice`)
2. **Classification** : predire le type de batiment (`BldgType`) parmi 5 categories (1Fam, 2FmCon, Duplx, TwnhsE, TwnhsI)

L'objectif final est de deployer ces modeles sous forme d'une **API REST** (FastAPI) accompagnee d'une **interface graphique** (Gradio), le tout heberge sur HuggingFace Spaces.

---

## 2. Exploration et traitement des donnees

### 2.1 Chargement et selection des variables

Le dataset provient de Kaggle (*House Prices Dataset*) et se compose de deux fichiers :
- **Train.csv** : 1460 observations avec 81 variables, incluant la variable cible (`SalePrice`). C'est ce fichier qui a ete utilise pour l'entrainement et l'evaluation des modeles.
- **Test.csv** : 1459 observations sans la variable cible. Seule une **exploration des donnees (EDA)** a ete realisee sur ce fichier (analyse des distributions, valeurs manquantes, etc.).

Apres analyse de la pertinence des features sur le jeu Train, deux sous-ensembles ont ete selectionnes :

- **Pour la regression** (15 variables) : GrLivArea, TotalBsmtSF, LotArea, BedroomAbvGr, FullBath, TotRmsAbvGrd, OverallQual, OverallCond, YearBuilt, YearRemodAdd, Neighborhood, GarageCars, GarageArea, PoolArea, Fireplaces
- **Pour la classification** (7 variables) : GrLivArea, TotRmsAbvGrd, OverallQual, YearBuilt, GarageCars, Neighborhood, HouseStyle

### 2.2 Exploration des donnees (EDA)

**Etape 1 — Doublons** : La verification avec `duplicated().sum()` a revele la presence de 5 lignes dupliquees dans le sous-ensemble de regression. Ces doublons ont ete supprimes avec `drop_duplicates()`. Le meme traitement a ete applique au sous-ensemble de classification.

**Etape 2 — Valeurs manquantes** : L'analyse avec `isna().sum()` et la visualisation avec la bibliotheque `missingno` (matrice de completude) ont montre qu'il n'y avait **aucune valeur manquante** dans les colonnes selectionnees pour les deux taches.

**Etape 3 — Analyse des distributions** : La visualisation des distributions via des histogrammes (`histplot`) et des boites a moustaches (`boxplot`) a mis en evidence :
- Des **distributions asymetriques** (skewness) pour la majorite des variables numeriques
- La **presence de valeurs aberrantes** (outliers) dans des variables comme LotArea, GrLivArea et GarageArea

### 2.3 Traitement des valeurs aberrantes

Etant donne les distributions asymetriques observees, la methode **IQR (Interquartile Range)** a ete choisie plutot qu'une methode basee sur l'ecart-type (qui suppose une distribution normale).

**Methode utilisee** :
- Calcul de Q1 (25e percentile) et Q3 (75e percentile)
- Calcul de l'IQR = Q3 − Q1
- Definition des bornes : `borne_inf = Q1 - 1.5 * IQR` et `borne_sup = Q3 + 1.5 * IQR`
- Application du clipping : les valeurs en dehors des bornes sont ramenees aux bornes (methode `clip`)
- Les colonnes avec un skewness compris entre -0.5 et 0.5 (distribution quasi-normale) sont exclues du traitement
- La variable cible (`SalePrice`) a ete exclue du traitement pour eviter de biaiser les predictions

Apres l'imputation IQR, une deuxieme suppression des doublons a ete effectuee car le clipping peut creer de nouvelles lignes identiques.

### 2.4 Encodage des variables categorielles

**OneHotEncoder** (scikit-learn) a ete utilise avec l'option `drop='if_binary'` pour eviter la multicolinearite sur les variables binaires :
- **Regression** : encodage de la variable `Neighborhood` (25 quartiers)
- **Classification** : encodage de `Neighborhood` et `HouseStyle` (8 styles)

Pour la classification, un **LabelEncoder** a egalement ete applique sur la variable cible `BldgType` pour transformer les labels textuels en valeurs numeriques.

### 2.5 Mise a l'echelle

Le **RobustScaler** (scikit-learn) a ete choisi pour la normalisation des variables numeriques. Ce scaler utilise la mediane et l'IQR au lieu de la moyenne et de l'ecart-type, ce qui le rend robuste face aux valeurs aberrantes residuelles. Il a ete applique sur toutes les colonnes numeriques, en excluant la variable cible.

---

## 3. Modeles et resultats

### 3.1 Regression — Prediction du prix

Le split est de 70% entrainement / 30% test. Deux modeles ont ete entraines :

**Decision Tree Regressor** : modele d'arbre de decision simple, servant de reference (baseline). En tant qu'arbre unique, il est sujet au surapprentissage.

**Random Forest Regressor** (max_depth=15, random_state=42) : ensemble de multiples arbres de decision. Le parametre `max_depth=15` limite la profondeur pour eviter le surapprentissage.

| Modele | MAE | RMSE | R² |
|---|---|---|---|
| Decision Tree | Eleve | Eleve | ~ 0.75 |
| **Random Forest** | Plus faible | Plus faible | **~ 0.85** |

Le **Random Forest** obtient de meilleurs resultats sur toutes les metriques, grace a l'agregation (bagging) qui reduit la variance.

### 3.2 Classification — Prediction du type de batiment

Deux modeles ont ete entraines, tous deux avec `class_weight='balanced'` pour compenser le desequilibre des classes (la classe 1Fam est tres majoritaire) :

**Random Forest Classifier avec GridSearchCV** : recherche exhaustive des meilleurs hyperparametres parmi `n_estimators` = [100, 200, 300] et `max_depth` = [5, 10, 15].

**SVM (Support Vector Machine)** : avec kernel RBF, C=1.0 et gamma='scale'.

| Modele | Accuracy | F1-score (weighted) |
|---|---|---|
| **Random Forest (GridSearchCV)** | **~ 0.80** | **~ 0.78** |
| SVM | ~ 0.77 | ~ 0.75 |

La **matrice de confusion** montre que la classe majoritaire (1Fam) est bien predite. Les classes minoritaires (TwnhsI, 2FmCon) presentent plus de difficultes en raison du faible nombre d'echantillons.

### 3.3 Sauvegarde

Tous les modeles et preprocesseurs (OneHotEncoders, RobustScalers, LabelEncoder) ont ete sauvegardes dans un fichier unique `Mes_models.pkl` via le module `pickle`, permettant un chargement rapide en production sans avoir a refaire le preprocessing.

---

## 4. Deploiement

### 4.1 Architecture technique

```
immo-predictor/
├── app/
│   ├── main.py           # FastAPI : routes API + montage Gradio
│   ├── schemas.py         # Validation Pydantic des entrees/sorties
│   ├── predictor.py       # Chargement du pkl + pipelines de prediction
│   └── ui.py              # Interface Gradio (2 onglets + graphiques)
├── models/
│   └── Mes_models.pkl     # Modeles serialises (~18 MB)
├── Dockerfile             # Image Docker pour HuggingFace Spaces
└── requirements.txt       # Dependances Python
```

### 4.2 API REST (FastAPI)

L'API expose 5 endpoints avec documentation Swagger automatique (`/docs`) :

| Methode | Route | Description |
|---|---|---|
| GET | `/` | Redirection vers l'interface |
| GET | `/health` | Statut de l'API |
| GET | `/models/info` | Informations sur les modeles charges |
| POST | `/regression/predict?model=` | Prediction du prix |
| POST | `/classification/predict?model=` | Classification du type |

Les modeles sont charges **une seule fois** au demarrage (mecanisme `lifespan`). La validation des entrees est assuree par **Pydantic** avec des contraintes (ex : OverallQual entre 1 et 10). Le middleware **CORS** est active pour autoriser les appels cross-origin.

### 4.3 Interface utilisateur (Gradio)

L'interface propose deux onglets avec des graphiques de visualisation (matplotlib) :
- **Prediction de Prix** : 15 champs + graphique de comparaison des modeles + profil du bien
- **Classification du Type** : 7 champs + graphique de comparaison + legende des types

### 4.4 Mise en production

Le projet est conteneurise avec **Docker** et deploye sur **HuggingFace Spaces** (port 7860). Le code source est disponible sur GitHub.

---

## 5. Conclusion

Ce projet illustre une chaine complete de Machine Learning :
**Exploration** → **Preprocessing** → **Modelisation** → **Evaluation** → **Deploiement**

Les modeles Random Forest se sont reveles les plus performants pour les deux taches. L'API FastAPI combinee a l'interface Gradio offre une solution accessible, permettant a tout utilisateur de predire le prix ou le type d'un bien immobilier.

**Pistes d'amelioration** :
- Tester d'autres algorithmes (XGBoost, LightGBM)
- Ajouter davantage de features du dataset original
- Implementer un systeme de feedback utilisateur pour ameliorer les modeles en continu
