import seaborn as sns
import matplotlib.pyplot as plt

def histogram(df, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df, 
        x=feature_name, 
        kde=True, 
        ax=ax,
        color='#4c72b0',
        edgecolor='white'
    )
    ax.set_title(f"Distribution of {feature_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    sns.despine()
    return fig

def scatter_plot(df, x_feature, y_feature):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax)
    ax.set_title(f"{x_feature} vs {y_feature}")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    return fig

def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def box_plot(df, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df[feature_name], ax=ax)
    ax.set_title(f"Box Plot of {feature_name}")
    ax.set_xlabel(feature_name)
    return fig 

def violin_plot(df, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x=df[feature_name], ax=ax)
    ax.set_title(f"Violin Plot of {feature_name}")
    ax.set_xlabel(feature_name)
    return fig

def bar_plot(df, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    df[feature_name].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Bar Plot of {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frequency")
    return fig

def pie_chart(df, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    df[feature_name].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title(f"Pie Chart of {feature_name}")
    return fig

def line_plot(df, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    df[feature_name].plot(kind='line', ax=ax)
    ax.set_title(f"Line Plot of {feature_name}")
    ax.set_xlabel("Index")
    ax.set_ylabel(feature_name)
    return fig
