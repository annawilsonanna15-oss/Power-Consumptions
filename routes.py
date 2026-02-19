from flask import Blueprint, render_template, request, session, redirect, url_for
from flask_login import login_required, current_user
import pandas as pd
import os, uuid
#import uuid
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import io, base64
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from flask import send_file
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from flask import session

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
import shap
import lime
import lime.lime_tabular
import datetime


UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

main = Blueprint("main", __name__)


@main.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")


@main.route("/seasonal")
@login_required
def seasonal():
    return render_template("seasonal.html")



@main.route("/upload", methods=["GET", "POST"])
@login_required
def upload():

    if request.method == "POST":

        file = request.files.get("file")

        if not file:
            return render_template("upload.html", error="No file selected")

        # Save file
        filename = file.filename
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # Store filename only
        session["uploaded_file"] = filename

        # Read dataset for preview
        df = pd.read_csv(filepath)

        # Clean column names
        df.columns = df.columns.str.strip()

        columns = df.columns.tolist()
        preview = df.head().to_html(
            classes="data-table",
            index=False
        )

        return render_template(
            "upload.html",
            success=True,
            columns=columns,
            preview=preview
        )

    return render_template("upload.html")





@main.route("/eda")
@login_required
def eda_page():

    # -----------------------------------
    # 1️⃣ Check Uploaded File
    # -----------------------------------
    filename = session.get("uploaded_file")

    if not filename:
        return redirect(url_for("main.upload"))

    filepath = os.path.join("uploads", filename)

    if not os.path.exists(filepath):
        return redirect(url_for("main.upload"))

    # -----------------------------------
    # 2️⃣ Load Dataset
    # -----------------------------------
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # -----------------------------------
    # 3️⃣ Datetime Handling
    # -----------------------------------
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            dayfirst=True,
            errors="coerce"
        )

    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Datetime"],
            errors="coerce"
        )

    else:
        return render_template(
            "eda.html",
            error="❌ No Date / Time / Datetime column found"
        )

    df.dropna(subset=["Datetime"], inplace=True)
    df.set_index("Datetime", inplace=True)

    # -----------------------------------
    # 4️⃣ Auto Detect Energy Column
    # -----------------------------------
    energy_col = None

    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["power", "active", "energy"]):
            energy_col = col
            break

    if not energy_col:
        return render_template(
            "eda.html",
            error="❌ Energy column not found"
        )

    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    df.dropna(subset=[energy_col], inplace=True)

    # -----------------------------------
    # 5️⃣ Daily Resampling
    # -----------------------------------
    daily = df[energy_col].resample("D").mean()

    # Handle missing values safely
    daily = daily.interpolate(method="linear")

    # -----------------------------------
    # 6️⃣ Line Chart (Interactive)
    # -----------------------------------
    line_fig = px.line(
        daily,
        title="📈 Daily Energy Consumption",
        labels={"value": "Energy Consumption", "Datetime": "Date"},
    )

    line_graph = line_fig.to_html(full_html=False)

    # -----------------------------------
    # 7️⃣ Seasonal Decomposition
    # -----------------------------------
    try:
        decomposition = seasonal_decompose(
            daily,
            model="additive",
            period=30
        )

        decomp_fig = go.Figure()

        decomp_fig.add_trace(
            go.Scatter(
                y=decomposition.trend,
                name="Trend",
                line=dict(color="gold")
            )
        )

        decomp_fig.add_trace(
            go.Scatter(
                y=decomposition.seasonal,
                name="Seasonality",
                line=dict(color="green")
            )
        )

        decomp_fig.add_trace(
            go.Scatter(
                y=decomposition.resid,
                name="Residual",
                line=dict(color="red")
            )
        )

        decomp_fig.update_layout(
            title="📊 Seasonal Decomposition",
            height=500
        )

        decomp_graph = decomp_fig.to_html(full_html=False)

    except Exception:
        decomp_graph = None

    # -----------------------------------
    # 8️⃣ Render Page
    # -----------------------------------
    return render_template(
        "eda.html",
        energy_col=energy_col,
        line_graph=line_graph,
        decomp_graph=decomp_graph
    )



@main.route("/forecasting")
@login_required
def forecasting():

    filename = session.get("uploaded_file")

    if not filename:
        return redirect(url_for("main.upload"))

    filepath = os.path.join("uploads", filename)

    if not os.path.exists(filepath):
        return render_template("forecasting.html",
                               error="Uploaded file not found")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # -----------------------------
    # Datetime Handling
    # -----------------------------
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            dayfirst=True,
            errors="coerce"
        )
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    else:
        return render_template("forecasting.html",
                               error="No Datetime column found")

    df.dropna(subset=["Datetime"], inplace=True)
    df.set_index("Datetime", inplace=True)

    # -----------------------------
    # Auto Detect Energy Column
    # -----------------------------
    energy_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["power", "energy", "active"]):
            energy_col = col
            break

    if not energy_col:
        return render_template("forecasting.html",
                               error="Energy column not found")

    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    df.dropna(subset=[energy_col], inplace=True)

    # -----------------------------
    # Daily Resampling
    # -----------------------------
    daily = df[energy_col].resample("D").mean()
    daily = daily.interpolate()

    # -----------------------------
    # Train/Test Split (80/20)
    # -----------------------------
    split = int(len(daily) * 0.8)
    train = daily[:split]
    test = daily[split:]

    # =============================
    # 1️⃣ ARIMA MODEL
    # =============================
    arima_model = ARIMA(train, order=(2,1,2))
    arima_fit = arima_model.fit()

    arima_forecast = arima_fit.get_forecast(steps=len(test))
    arima_pred = arima_forecast.predicted_mean
    conf_int = arima_forecast.conf_int()

    arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
    arima_mae = mean_absolute_error(test, arima_pred)

    # Save metrics for comparison page
    session["arima_metrics"] = {
        "rmse": float(arima_rmse),
        "mae": float(arima_mae)
    }

    # ARIMA Plot
    arima_fig = go.Figure()
    arima_fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
    arima_fig.add_trace(go.Scatter(x=test.index, y=test, name="Actual"))
    arima_fig.add_trace(go.Scatter(x=test.index, y=arima_pred, name="ARIMA Forecast"))

    arima_fig.add_trace(go.Scatter(
        x=test.index,
        y=conf_int.iloc[:, 0],
        line=dict(width=0),
        showlegend=False
    ))

    arima_fig.add_trace(go.Scatter(
        x=test.index,
        y=conf_int.iloc[:, 1],
        fill='tonexty',
        name="Confidence Interval",
        line=dict(width=0)
    ))

    arima_graph = arima_fig.to_html(full_html=False)

    # =============================
    # 2️⃣ LSTM MODEL
    # =============================
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily.values.reshape(-1,1))

    def create_sequences(data, step=30):
        X, y = [], []
        for i in range(len(data)-step):
            X.append(data[i:i+step])
            y.append(data[i+step])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled)

    split_seq = int(len(X)*0.8)
    X_train, X_test = X[:split_seq], X[split_seq:]
    y_train, y_test = y[:split_seq], y[split_seq:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[es],
        verbose=0
    )

    lstm_pred = model.predict(X_test)

    lstm_pred = scaler.inverse_transform(lstm_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

    lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred))
    lstm_mae = mean_absolute_error(y_test_inv, lstm_pred)

    # Save metrics for comparison
    session["lstm_metrics"] = {
        "rmse": float(lstm_rmse),
        "mae": float(lstm_mae)
    }

    # LSTM Plot
    lstm_fig = go.Figure()
    lstm_fig.add_trace(go.Scatter(y=y_test_inv.flatten(), name="Actual"))
    lstm_fig.add_trace(go.Scatter(y=lstm_pred.flatten(), name="LSTM Forecast"))

    lstm_graph = lstm_fig.to_html(full_html=False)

    # =============================
    # Download CSV
    # =============================
   
# Align lengths safely
    min_len = min(len(test), len(arima_pred), len(lstm_pred))

    forecast_df = pd.DataFrame({
        "Date": test.index[:min_len],
        "Actual": test.values[:min_len],
        "ARIMA_Predicted": arima_pred.values[:min_len],
        "LSTM_Predicted": lstm_pred.flatten()[:min_len]
    })

    forecast_path = os.path.join("uploads", "forecast_output.csv")
    forecast_df.to_csv(forecast_path, index=False)

    os.makedirs("uploads", exist_ok=True)

    return render_template(
        "forecasting.html",
        arima_graph=arima_graph,
        lstm_graph=lstm_graph,
        arima_rmse=round(arima_rmse,3),
        arima_mae=round(arima_mae,3),
        lstm_rmse=round(lstm_rmse,3),
        lstm_mae=round(lstm_mae,3),
        download_file="forecast_output.csv"
    )



@main.route("/anomaly")
@login_required
def anomaly():

    filename = session.get("uploaded_file")
    if not filename:
        return redirect(url_for("main.upload"))

    filepath = os.path.join("uploads", filename)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # --------------------------
    # Datetime Handling
    # --------------------------
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            dayfirst=True,
            errors="coerce"
        )
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    else:
        return render_template("anomaly.html",
                               error="No Datetime column found")

    df.dropna(subset=["Datetime"], inplace=True)
    df.set_index("Datetime", inplace=True)

    # --------------------------
    # Energy Column Detection
    # --------------------------
    energy_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["power", "energy", "active"]):
            energy_col = col
            break

    if not energy_col:
        return render_template("anomaly.html",
                               error="Energy column not found")

    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    df.dropna(subset=[energy_col], inplace=True)

    daily = df[energy_col].resample("D").mean()
    daily = daily.interpolate()

    data = daily.values.reshape(-1,1)

    # ==========================
    # 1️⃣ Isolation Forest
    # ==========================
    iso = IsolationForest(contamination=0.02,
                          random_state=42)
    iso_pred = iso.fit_predict(data)

    # ==========================
    # 2️⃣ One-Class SVM
    # ==========================
    svm = OneClassSVM(nu=0.02)
    svm_pred = svm.fit_predict(data)

    # Convert to 0 (normal) / 1 (anomaly)
    iso_anomaly = np.where(iso_pred == -1, 1, 0)
    svm_anomaly = np.where(svm_pred == -1, 1, 0)

    # ==========================
    # Confusion Matrix
    # ==========================
    cm = confusion_matrix(iso_anomaly, svm_anomaly)

    tn, fp, fn, tp = cm.ravel()

    # ==========================
    # Plot
    # ==========================
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily.index,
        y=daily.values,
        name="Energy"
    ))

    anomaly_points = daily[iso_anomaly == 1]

    fig.add_trace(go.Scatter(
        x=anomaly_points.index,
        y=anomaly_points.values,
        mode="markers",
        marker=dict(color="red", size=8),
        name="Isolation Forest Anomaly"
    ))

    anomaly_graph = fig.to_html(full_html=False)

    return render_template(
        "anomaly.html",
        anomaly_graph=anomaly_graph,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp
    )


@main.route("/comparison")
@login_required
def comparison():

    forecast_path = os.path.join("uploads", "forecast_output.csv")

    # If forecast not generated
    if not os.path.exists(forecast_path):
        return render_template(
            "comparison.html",
            error="Run forecasting first.",
            best_model=None
        )

    # Read forecast results
    df = pd.read_csv(forecast_path)

    if "Actual" not in df.columns or "ARIMA_Predicted" not in df.columns:
        return render_template(
            "comparison.html",
            error="Invalid forecast file format."
        )

    # =============================
    # Calculate ARIMA metrics
    # =============================
    arima_rmse = np.sqrt(
        mean_squared_error(df["Actual"], df["ARIMA_Predicted"])
    )

    arima_mae = mean_absolute_error(
        df["Actual"], df["ARIMA_Predicted"]
    )

    # =============================
    # LSTM (Optional if stored separately)
    # =============================
    lstm_rmse = None
    lstm_mae = None

    if "LSTM_Predicted" in df.columns:
        lstm_rmse = np.sqrt(
            mean_squared_error(df["Actual"], df["LSTM_Predicted"])
        )

        lstm_mae = mean_absolute_error(
            df["Actual"], df["LSTM_Predicted"]
        )

    # =============================
    # Best Model Logic
    # =============================
    best_model = "ARIMA"
    best_reason = ""

    if lstm_rmse and lstm_rmse < arima_rmse:
        best_model = "LSTM"

    improvement = 0
    if lstm_rmse:
        improvement = abs(arima_rmse - lstm_rmse) / arima_rmse * 100
        best_reason = f"{best_model} reduces RMSE by {round(improvement,2)}%"

    # =============================
    # Ranking Score (Lower RMSE = Better Score)
    # =============================
    arima_score = round(100 / (1 + arima_rmse), 2)

    lstm_score = None
    if lstm_rmse:
        lstm_score = round(100 / (1 + lstm_rmse), 2)

    return render_template(
        "comparison.html",
        arima_rmse=round(arima_rmse, 4),
        arima_mae=round(arima_mae, 4),
        lstm_rmse=round(lstm_rmse, 4) if lstm_rmse else None,
        lstm_mae=round(lstm_mae, 4) if lstm_mae else None,
        arima_score=arima_score,
        lstm_score=lstm_score,
        best_model=best_model,
        best_reason=best_reason,
        improvement=round(improvement, 2)
    )

@main.route("/explain")
@login_required
def explain():
    try:
        filename = session.get("uploaded_file")
        if not filename:
            return redirect(url_for("main.upload"))

        filepath = os.path.join("uploads", filename)
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        # -----------------------------
        # Remove Date & Time safely
        # -----------------------------
        df = df.drop(columns=["Date", "Time"], errors="ignore")

        # Convert to numeric safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_df = df.select_dtypes(include=["float64", "int64"])
        numeric_df = numeric_df.dropna()

        if numeric_df.shape[0] < 20 or numeric_df.shape[1] < 2:
            return render_template(
                "explain.html",
                error="Not enough numeric data for explainability."
            )

        # ---------------------------------
        # Reduce size (VERY IMPORTANT)
        # ---------------------------------
        numeric_df = numeric_df.sample(min(500, len(numeric_df)), random_state=42)

        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]

        # ---------------------------------
        # Train Model
        # ---------------------------------
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # ---------------------------------
        # SHAP
        # ---------------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For regression shap_values is array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # ---------------- SHAP BAR PLOT ----------------
        plt.figure(figsize=(6, 4))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False
        )

        buf_bar = io.BytesIO()
        plt.savefig(buf_bar, format="png", bbox_inches="tight")
        buf_bar.seek(0)
        shap_bar = base64.b64encode(buf_bar.getvalue()).decode("utf-8")
        plt.close()

        # ---------------- SHAP FORCE PLOT ----------------
        plt.figure(figsize=(10, 3))
        shap.plots._force.force(
            explainer.expected_value,
            shap_values[0],
            X.iloc[0],
            matplotlib=True,
            show=False
        )

        buf_force = io.BytesIO()
        plt.savefig(buf_force, format="png", bbox_inches="tight")
        buf_force.seek(0)
        shap_force = base64.b64encode(buf_force.getvalue()).decode("utf-8")
        plt.close()

        # ---------------------------------
        # Top 5 Important Features
        # ---------------------------------
        importance = np.abs(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        top5 = feature_importance.head(5)

        return render_template(
            "explain.html",
            shap_bar=shap_bar,
            shap_force=shap_force,
            top5=top5.to_dict(orient="records")
        )

    except Exception as e:
        return render_template(
            "explain.html",
            error=str(e)
        )



@main.route("/monitor")
@login_required
def monitor():

    forecast_path = os.path.join("uploads", "forecast_output.csv")

    if not os.path.exists(forecast_path):
        return render_template(
            "monitor.html",
            error="Run forecasting first."
        )

    df = pd.read_csv(forecast_path)

    if "Actual" not in df.columns:
        return render_template(
            "monitor.html",
            error="Invalid forecast file."
        )

    # -----------------------------
    # ARIMA Metrics
    # -----------------------------
    arima_rmse = None
    if "ARIMA_Predicted" in df.columns:
        arima_rmse = np.sqrt(
            mean_squared_error(df["Actual"],
                               df["ARIMA_Predicted"])
        )

    # -----------------------------
    # LSTM Metrics
    # -----------------------------
    lstm_rmse = None
    if "LSTM_Predicted" in df.columns:
        lstm_rmse = np.sqrt(
            mean_squared_error(df["Actual"],
                               df["LSTM_Predicted"])
        )

    # -----------------------------
    # Last Training Time
    # -----------------------------
    last_training = datetime.datetime.fromtimestamp(
        os.path.getmtime(forecast_path)
    ).strftime("%Y-%m-%d %H:%M")

    return render_template(
        "monitor.html",
        arima_rmse=round(arima_rmse, 4) if arima_rmse else None,
        lstm_rmse=round(lstm_rmse, 4) if lstm_rmse else None,
        last_training=last_training
    )


