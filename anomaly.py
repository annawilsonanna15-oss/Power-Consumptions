
from sklearn.ensemble import IsolationForest

def detect_anomaly(df, column):
    iso = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly"] = iso.fit_predict(df[[column]])
    return df
