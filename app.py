from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/Sziszkó/Downloads/digital_marketing_campaign_dataset.csv")

# Define features & target
feature_cols = [
    "Age", "Income", "AdSpend", "ClickThroughRate", "ConversionRate",
    "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares",
    "EmailOpens", "EmailClicks", "PreviousPurchases", "LoyaltyPoints"
]
X = df[feature_cols]
y = df["Conversion"]

# Fit XGBoost model
model = XGBClassifier(n_estimators=400, max_depth=12, random_state=45, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Predict high-converting customers
df["PredictedConversion"] = model.predict_proba(X)[:, 1] > 0.8
potential_customers = df[df["PredictedConversion"]]

# Feature importance
importance_values = model.feature_importances_
feature_importance = pd.Series(importance_values, index=feature_cols).sort_values()

# Shiny UI
app_ui = ui.page_fluid(
    ui.tags.div(
        "Campaign Targeting Dashboard",
        style="background-color:#f0f0f0; padding:20px; font-size:24px; font-weight:bold; text-align:center; border-radius:10px; margin-bottom:20px;"
    ),
    ui.navset_tab(
        ui.nav_panel("🎯 Potential Customers", ui.output_table("customer_table")),
        ui.nav_panel("📊 Feature Importance", ui.output_plot("feature_plot")),
        ui.nav_panel("📣 Campaign Channel Breakdown", ui.output_plot("channel_plot")),
        ui.nav_panel("📣 Campaign Type Breakdown", ui.output_plot("type_plot"))
    )
)

# Shiny server
def server(input, output, session):
    @output
    @render.table
    def customer_table():
        return potential_customers[["CustomerID", "CampaignType", "CampaignChannel"]]

    @output
    @render.plot
    def feature_plot():
        fig, ax = plt.subplots()
        feature_importance.plot(kind="barh", ax=ax)
        ax.set_title("Most Influential Features for Conversion")
        ax.invert_yaxis()
        return fig

    @reactive.Calc
    def channel_counts():
        return potential_customers["CampaignChannel"].value_counts()

    @reactive.Calc
    def type_counts():
        return potential_customers["CampaignType"].value_counts()

    @output
    @render.plot
    def channel_plot():
        fig, ax = plt.subplots()
        channel_counts().plot(kind="bar", ax=ax)
        ax.set_title("Potential Customers by Campaign Channel")
        ax.set_ylabel("Count")
        return fig

    @output
    @render.plot
    def type_plot():
        fig, ax = plt.subplots()
        type_counts().plot(kind="bar", ax=ax)
        ax.set_title("Potential Customers by Campaign Type")
        ax.set_ylabel("Count")
        return fig

app = App(app_ui, server)
