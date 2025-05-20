# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("🤖 Fulfil Product Order Exploratory Data Analysis")
st.markdown("**By Jinnson Lim**")

# loading in the datasets, merging, and creating some new features
@st.cache_data
def load_data():
    product = pd.read_csv("./data/product.csv")
    purchase_header = pd.read_csv("./data/purchase_header.csv")
    purchase_lines = pd.read_csv("./data/purchase_lines.csv")
    df = pd.merge(pd.merge(purchase_lines, product, how='left'), purchase_header, how='left')
    df["PURCHASE_DATE_TIME"] = pd.to_datetime(df["PURCHASE_DATE_TIME"])
    df["VOLUME"] = df["HEIGHT_INCHES"] * df["WIDTH_INCHES"] * df["DEPTH_INCHES"]
    day_name_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df["DAY"] = df["PURCHASE_DATE_TIME"].dt.dayofweek.map(day_name_map)
    df["HOUR"] = df["PURCHASE_DATE_TIME"].dt.hour
    df["DATE"] = df["PURCHASE_DATE_TIME"].dt.date
    return df

df = load_data()

# main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Orders", f"{df['PURCHASE_ID'].nunique():,}")
with col2:
    st.metric("Unique Products", f"{df['PRODUCT_ID'].nunique():,}")
with col3:
    st.metric("Missing Volume %", f"{df['VOLUME'].isna().mean()*100:.1f}%", delta="+5%")
with col4:
    st.metric("Avg Order Size", f"{df.groupby('PURCHASE_ID')['QUANTITY'].sum().mean():.1f} Items")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric("Top Department", df["DEPARTMENT_NAME"].value_counts().idxmax())
with col6:
    st.metric("Top Product", df.groupby("PURCHASE_ID")["QUANTITY"].sum().idxmax())
with col7:
    orders_by_hour = df.groupby(["DATE", "HOUR"])["PURCHASE_ID"].nunique()
    st.metric("Orders per Hour", f"{orders_by_hour.mean():.1f}")
with col8:
    peak = orders_by_hour.reset_index()
    peak_hour = peak.loc[peak["PURCHASE_ID"].idxmax(), "HOUR"]
    peak_orders = peak["PURCHASE_ID"].max()
    st.metric("Peak Hour", f"{int(peak_hour)}:00", delta=f"{int(peak_orders)} orders")

st.markdown("---")

# dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(100))
st.markdown("**Insights**: The datasets (originally comprised of three seperate tables) were merged together into one central dataframe. Looking into the features, we can see it is an overview of purchased products and various features such as department and dimensions. I did perform some additional feature engineering to extract more data such as VOLUME, and DAY/HOURS.")

# graphing null value ratios1
st.subheader("Missing Data Overview")
missing = df.isnull().mean().reset_index().sort_values(by=0, ascending=False)
missing.columns = ['Column', 'Missing Ratio']
missing['Percent'] = (missing['Missing Ratio'] * 100).round(1).astype(str) + '%'
fig_missing = px.bar(missing, 
                     x='Column',
                     y='Missing Ratio', 
                     text='Percent',
                     color="Missing Ratio",
                     color_continuous_scale='tealgrn',
                     title='Missing Data Ratio by Column',
                     labels={'Missing Ratio': 'Missing Ratio'},
                     hover_data=['Missing Ratio'])
fig_missing.update_traces(textposition='outside')
fig_missing.update_layout(yaxis=dict(range=[0, .8]))
st.plotly_chart(fig_missing, use_container_width=True)
st.markdown("**Insights**: One of the main issues with the dataset is the prevalence of NULL values. At the highest ratio of 70%, it severly limits the potential of the EDA but to move forward with the analysis, I opted to ignore entries that have NULLs for their respective functions.")

# quick summary of dimension and weight statistics
st.subheader("Summary Statistics")
numeric_cols = ['WEIGHT_GRAMS', 'HEIGHT_INCHES', 'WIDTH_INCHES', 'DEPTH_INCHES', 'VOLUME']
dim_stats = df[numeric_cols].agg(['min', 'max', 'mean', 'std']).T.round(2)
dim_stats.columns = ['Min', 'Max', 'Mean', 'Std Dev']
st.dataframe(dim_stats)
st.markdown("**Insights**: Looking at the basic summary statistics of the features (dropping NULLs), we observe some interesting and concerning values within the table. Specifically a negative weight for one of the products. This may be due to human input error, but can cause large problems if not removed for modeling/analysis. In addition, we do see some great outliers within the data. Take example the Max weight of 118000g which equates to ~260 pounds. Being aware of these outliers is vitaly important for the functionality and quality of order fulfillment, given the restrictions of robots.")

# orders by department
st.subheader("Orders by Departments")
orders_by_dept = df.groupby('DEPARTMENT_NAME')['QUANTITY'].sum().reset_index().sort_values(by='QUANTITY', ascending=False)
fig_dept = px.bar(
    orders_by_dept,
    x='DEPARTMENT_NAME',
    y='QUANTITY',
    color="QUANTITY",
    color_continuous_scale='tealgrn',
    title='Total Orders by Department',
    labels={'QUANTITY': 'Total Quantity', 'DEPARTMENT_NAME': 'Department'},
    text='QUANTITY'
)
fig_dept.update_traces(textposition='auto')
st.plotly_chart(fig_dept, use_container_width=True)
st.markdown("**Insights**: Most popular departments are food related, with less emphasis on less volatile goods. Reveals which areas to target for customer demand.")

# department specific analysis
st.markdown("### Choose Department to Analyze")
st.markdown('Pick a department below using the dropdown menu. From there, we will perform various data analyses on product dimensions, order rates, and the top selling products.')
dept_col = st.columns([1, 6, 1])[1]
with dept_col:
    department = st.selectbox("Select a department to explore:", df['DEPARTMENT_NAME'].dropna().unique())

df_dept = df[df['DEPARTMENT_NAME'] == department]
st.markdown(f"### Exploring the `{department}` Department")

fig_weight = px.histogram(df_dept, x='WEIGHT_GRAMS', nbins=50, title='Weight Distribution', color_discrete_sequence=px.colors.sequential.Tealgrn)
fig_weight.update_layout(xaxis_title="Weight (grams)", yaxis_title="Count")

fig_volume = px.box(df_dept, y='VOLUME', title='Volume Range', color_discrete_sequence=px.colors.sequential.Tealgrn)
fig_volume.update_layout(yaxis_title="Volume (in³)")

fig_hour = px.line(df_dept.groupby('HOUR')['QUANTITY'].sum().reset_index(),
                   x='HOUR', y='QUANTITY',
                   title='Hourly Order Quantity',
                   color_discrete_sequence=px.colors.sequential.Tealgrn,
                   markers=True)
fig_hour.update_layout(xaxis_title="Hour of Day", yaxis_title="Quantity Ordered")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_weight, use_container_width=True)
with col2:
    st.plotly_chart(fig_volume, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig_hour, use_container_width=True)
with col4:
    st.markdown("**Top Products by Quantity**")
    st.dataframe(df_dept.groupby('PRODUCT_ID')['QUANTITY'].sum().nlargest(10).reset_index().rename(columns={'PRODUCT_ID': 'Product ID', 'QUANTITY': 'Total Quantity'}))

# orders over time
st.subheader("Orders Over Time")
orders_pivot = df.groupby(["DAY", "HOUR"])["QUANTITY"].sum().reset_index().pivot(index="HOUR", columns="DAY", values="QUANTITY").reset_index()
fig_time = px.line(orders_pivot, x="HOUR", y=orders_pivot.columns[1:], title="Orders Over Time by Hour and Day")
st.plotly_chart(fig_time, use_container_width=True)
st.markdown("**Insights**: We can see that the general trend of orders through out the days of the week follows a general bell curve. Though, there does seem to be more fluxations with weekends having more variable order high and lows.")

# product complexity
df['COMPLEXITY'] = np.log(df['QUANTITY'] * df['VOLUME'] * df['WEIGHT_GRAMS'])
complex_products = df.groupby('PRODUCT_ID')['COMPLEXITY'].sum().nlargest(10).reset_index()
st.subheader("Most Complex Products to Fulfill")
st.markdown("This graph shows the *complexity* of fulfilling a given order. *complexity*  was calculated factoring in the VOLUME, QUANTITY, and WEIGHT of a given order, log scaled for visability.")
st.dataframe(complex_products.rename(columns={"PRODUCT_ID": "Product ID", "COMPLEXITY": "Fulfillment Complexity Score"}))

# robot load estimator
st.subheader("Robot Load Estimator")
st.markdown("Being aware of how many robots are needed at a given time (especially during peak hours) is key to make sure orders are getting fulfilled and operations are not backed up. Below, is an interactive graph that visualizes the number of robots needed at a given hour based on how much one can carry.")
capacity = st.slider("Robot capacity (g)", 1000, 10000, 5000, step=500)
hourly_avg = df.groupby(["DATE", "HOUR"])["WEIGHT_GRAMS"].sum().reset_index()
hourly_avg = hourly_avg.rename(columns={"WEIGHT_GRAMS":"ROBOTS_NEEDED"})
avg_weight_per_hour = hourly_avg.groupby("HOUR")["ROBOTS_NEEDED"].mean()
robots_needed = np.ceil(avg_weight_per_hour / capacity)
st.metric("Peak Robots Needed in an Hour", f"{int(robots_needed.max())} robots")
st.bar_chart(robots_needed, use_container_width=True)


# feature correlation heatmap
st.subheader("Correlation Heatmap of Physical Features")
st.markdown("Performing some feature correlation heatmaps to find features that could be used in predective modeling.")
corr = df[numeric_cols[:-1]].corr().round(3)
fig_corr = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(), colorscale='Viridis', showscale=True)
st.plotly_chart(fig_corr, use_container_width=True)


# k-means modeling with product dimensions
st.subheader("Product Clustering by Physical Dimensions with K-Means")
st.markdown("Knowing the similarity between products is highly important to streamline order fulfillment and reduce packing times. Below we use K-means clustering to find products that have similar characteristics to maximize robot operations. Running the model did slow down the dashboard when first loading, so I delayed the clustering until users decide to view it. **Press the button below to see the 3D viz!**")
filter_outliers = st.checkbox("Filter outliers?", value=True)
features = df[['HEIGHT_INCHES', 'WIDTH_INCHES', 'DEPTH_INCHES', 'WEIGHT_GRAMS']].dropna()

if filter_outliers:
    for col in features.columns:
        upper = features[col].quantile(0.95)
        lower = features[col].quantile(0.05)
        features = features[(features[col] >= lower) & (features[col] <= upper)]

if st.button("Run K-Means Clustering"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X_scaled)
    features['CLUSTER'] = kmeans.labels_

    fig_cluster = px.scatter_3d(
        features.reset_index(drop=True),
        x='HEIGHT_INCHES',
        y='WIDTH_INCHES',
        z='DEPTH_INCHES',
        color='CLUSTER',
        title='Product Clusters by Physical Dimensions',
        color_continuous_scale='Tealgrn',
        width=1200,
        height=700
    )
    fig_cluster.update_layout(scene=dict(
        xaxis_title='Height (in)',
        yaxis_title='Width (in)',
        zaxis_title='Depth (in)'
    ), margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_cluster, use_container_width=True)
else:
    st.info("Click the button above to run clustering analysis.")