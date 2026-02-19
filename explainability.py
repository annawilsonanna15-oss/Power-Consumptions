
import shap
import matplotlib.pyplot as plt
import io, base64
from sklearn.ensemble import RandomForestRegressor

def shap_plot(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = RandomForestRegressor()
    model.fit(X,y)
    explainer = shap.Explainer(model,X)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
