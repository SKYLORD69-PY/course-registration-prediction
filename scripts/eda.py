import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.db_connection import get_engine
from project_config import PLOTS_FOLDER


def load_data():

    engine = get_engine()

    query = "SELECT * FROM terms_enrollment"

    df = pd.read_sql(query, engine)

    return df


def enrollment_by_school(df):

    plt.figure(figsize=(10,10))

    sns.barplot(
        data=df,
        x="school",
        y="enrollment_this_term",
        estimator="mean"
    )

    plt.title("Average Enrollment by School")

    plt.savefig(os.path.join(PLOTS_FOLDER, "enrollment_by_school.png"))

    plt.close()


def enrollment_by_term(df):

    plt.figure(figsize=(10,10))

    sns.barplot(
        data=df,
        x="term_label",
        y="enrollment_this_term",
        estimator="mean"
    )

    plt.title("Average Enrollment by Term")

    plt.savefig(os.path.join(PLOTS_FOLDER, "enrollment_by_term.png"))

    plt.close()


def enrollment_trend(df):

    df = df.sort_values("term_start_date")

    plt.figure(figsize=(30,10))

    sns.lineplot(
        data=df,
        x="term_start_date",
        y="enrollment_this_term",
        hue="school"
    )

    plt.title("Enrollment Trend Over Time")

    plt.savefig(os.path.join(PLOTS_FOLDER, "enrollment_trend.png"))

    plt.close()


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(15,15))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=False
    )

    plt.title("Feature Correlation Heatmap")

    plt.savefig(os.path.join(PLOTS_FOLDER, "correlation_heatmap.png"))

    plt.close()


def run_eda():

    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    df = load_data()

    print("Running EDA...")

    enrollment_by_school(df)

    enrollment_by_term(df)

    enrollment_trend(df)

    correlation_heatmap(df)

    print("EDA completed. Plots saved in artifacts/plots/")


if __name__ == "__main__":

    run_eda()