# importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,silhouette_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import plotly.express as px
from streamlit_lottie import st_lottie
import requests


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_book = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json")
st_lottie(lottie_book, speed=1, height=500, key="initial")




hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    .centered-title {display: flex;justify-content: center;}
    .styled-image {width:20%}
    #MainMenu {visibility: hidden;}
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)



#st.title(':sparkles: MLee :sparkles:')
upload_icon = "📁"
exploration_icon = "🔍"
preprocessing_icon = "🛠️"
visualization_icon = "📊"
splitting_icon = "🔀"  # Change this icon to your preference
modeling_icon = "🤖"




st.sidebar.subheader(f"{upload_icon} Uploading Data")
upload_file = st.sidebar.file_uploader("Choose a file ", type=["csv", "xls", "xlsx"])


# Add subheaders with icons to the sidebar

st.sidebar.subheader(f"{exploration_icon} Data Exploration")
st.sidebar.subheader(f"{preprocessing_icon} Data Pre-Processing")
st.sidebar.subheader(f"{visualization_icon} Data Visualization")
st.sidebar.subheader(f"{splitting_icon} Data Splitting")
st.sidebar.subheader(f"{modeling_icon} Modeling and Evaluation")


def get_dataset():
        df = None
        if upload_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or upload_file.type == "application/vnd.ms-excel":
            st.write("Reading Excel file...")
            df = pd.read_excel(upload_file)
            st.write("Excel file read successfully!")
        elif upload_file.type == "text/csv":
            use_custom_sep = st.checkbox("Use custom separator (sep)")
            if use_custom_sep:
                custom_sep = st.text_input("Enter custom separator:",value=',')
                st.write(f"Reading CSV file with custom separator: '{custom_sep}'...")
                df = pd.read_csv(upload_file, sep=custom_sep)

            else:
                st.write("Reading CSV file...")
                df = pd.read_csv(upload_file)
        return df

if upload_file:

    df = get_dataset()
    
    st.success("Data read successfully")

    if st.checkbox('Show data'):
        st.write(df)





    st.markdown("<h1 style='font-family: Arial, sans-serif;'>Data Exploration</h1>", unsafe_allow_html=True)
    if st.checkbox('shape of dataset'):
        st.write(df.shape)
    if st.checkbox('Missing values'):
        missing_columns = df.columns[df.isnull().any()].tolist()
        if missing_columns:
            st.write('Columns with Missing Values:')
            for col in missing_columns:
                st.write(f"- {col}")
        else:
            st.write('No columns have missing values.')
    if st.checkbox('Sum of missing values'):
        st.write(df.isnull().sum())
    if st.checkbox('data description'):
        st.write(df.describe()) 
    if st.checkbox('data types'):
        st.write(df.dtypes)
        

   
    st.markdown("<h1 style='font-family: Arial, sans-serif;'>Data Pre-Processing</h1>", unsafe_allow_html=True)

    st.subheader("Data Cleaning")
    if st.checkbox('any Duplicates'):
        st.write(df.duplicated().any())
    if st.checkbox('drop Duplicates'):
        df.drop_duplicates(inplace=True)


    st.subheader("Handling Missing Values")
    missing_cols=[]
    for column in df.columns:
        if df[column].isnull().any():
            missing_cols.append(column)
    try:
     for index, miss in enumerate(missing_cols):
        if st.checkbox(miss, key=f"checkbox_{index}"):
            missing_strategy = st.selectbox(f'Select the strategy for missing values ({miss})',
                                            ['mean', 'median', 'mode', 'constant', 'drop_row'],
                                            key=f"selectbox_{index}"
                                           )

            if missing_strategy == 'mean' and miss in df.select_dtypes(include=[np.number]):
                df[miss] = df[miss].fillna(df[miss].mean())
            elif missing_strategy == 'median' and miss in df.select_dtypes(include=[np.number]):
                df[miss] = df[miss].fillna(df[miss].median())
            elif missing_strategy == 'mode':
                df[miss] = df[miss].fillna(df[miss].mode()[0])
            elif missing_strategy == 'constant':
                c = st.text_input("Constant:", key=f"text_input_{index}")
                df[miss] = df[miss].fillna(c)
            elif missing_strategy == 'drop_row':
                df = df.dropna(subset=[miss])

    except Exception as e:
     print(e)
     
    
# Create a custom CSS style to highlight missing values with a yellow background
    highlighted_css = """
    <style>
        .highlight-missing {
            background-color: lightblue !important;
        }
    </style>
"""

# Highlight missing values with a yellow background
    df_missing_highlighted = df.style.applymap(lambda x: 'background-color: lightblue' if pd.isnull(x) else '', subset=missing_cols)

# Display the DataFrame with missing values highlighted
    st.markdown(highlighted_css, unsafe_allow_html=True)
    st.write("Input features after handling the missing data:", df_missing_highlighted)

   

    # Display a section for data visualization

    st.markdown("<h1 style='font-family: Arial, sans-serif;'>Data Visualization</h1>", unsafe_allow_html=True)

    # Choose a visualization type using a selectbox
    visualization_type = st.selectbox("Select Visualization", ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Pie Chart", "3D Scatter Plot", "Heatmap", "Area Chart", "Choropleth Map"])
    
    enable_hue = st.checkbox("Enable Hue")

    if visualization_type == "Bar Chart":
        st.subheader("Bar Chart")
        x_column = st.selectbox("Select Column for X-Axis", df.columns)
    
        if enable_hue:
            hue_column = st.selectbox("Select Hue Column", df.columns)
        
            # Group and aggregate the data to get the count of categories for each combination
            grouped_df = df.groupby([x_column, hue_column]).size().unstack(fill_value=0)
        
            # Create the stacked bar chart using Plotly Express
            bar_chart = px.bar(grouped_df, x=grouped_df.index, y=grouped_df.columns, color=hue_column, barmode='stack')
        else:
            bar_chart = px.bar(df, x=x_column)
    
        st.plotly_chart(bar_chart)

    elif visualization_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_column = st.selectbox("Select Column for X-Axis", df.columns)
        y_column = st.selectbox("Select Column for Y-Axis", df.columns)
    
        if enable_hue:
            hue_column = st.selectbox("Select Hue Column", df.columns)
            scatter_chart = px.scatter(df, x=x_column, y=y_column, color=hue_column)
        else:
            scatter_chart = px.scatter(df, x=x_column, y=y_column)
    
        st.plotly_chart(scatter_chart)

    elif visualization_type == "Box Plot":
        st.subheader("Box Plot")
        x_column = st.selectbox("Select Column for X-Axis", df.columns)
        y_column = st.selectbox("Select Column for Y-Axis", df.columns)
    
        if enable_hue:
            hue_column = st.selectbox("Select Hue Column", df.columns)
            box_plot = px.box(df, x=x_column, y=y_column, color=hue_column)
        else:
            box_plot = px.box(df, x=x_column, y=y_column)
    
        st.plotly_chart(box_plot)
    
    elif visualization_type == "Line Chart":
        st.subheader("Line Chart")
        x_column = st.selectbox("Select Column for X-Axis", df.columns)
        y_column = st.selectbox("Select Column for Y-Axis", df.columns)
        line_chart = px.line(df, x=x_column, y=y_column)
        st.plotly_chart(line_chart)
    elif visualization_type == "Histogram":
        st.subheader("Histogram")
        hist_column = st.selectbox("Select Column for Histogram", df.columns)
        histogram = px.histogram(df, x=hist_column)
        st.plotly_chart(histogram) 
    
    elif visualization_type == "Pie Chart":
        st.subheader("Pie Chart")
        pie_column = st.selectbox("Select Column for Pie Chart", df.columns)
        pie_chart = px.pie(df, names=pie_column)
        st.plotly_chart(pie_chart)

    elif visualization_type == "3D Scatter Plot":
        st.subheader("3D Scatter Plot")
        x_column = st.selectbox("Select Column for X-Axis", df.columns)
        y_column = st.selectbox("Select Column for Y-Axis", df.columns)
        z_column = st.selectbox("Select Column for Z-Axis", df.columns)
        scatter_3d = px.scatter_3d(df, x=x_column, y=y_column, z=z_column)
        st.plotly_chart(scatter_3d)    
    elif visualization_type == "Heatmap":
        st.subheader("Heatmap")
        heatmap_columns = st.multiselect("Select Columns for Heatmap", df.columns)
        heatmap = px.imshow(df[heatmap_columns].corr())
        st.plotly_chart(heatmap)

    elif visualization_type == "Area Chart":
        st.subheader("Area Chart")
        x_column = st.selectbox("Select Column for X-Axis", df.columns)
        y_column = st.selectbox("Select Column for Y-Axis", df.columns)
        area_chart = px.area(df, x=x_column, y=y_column)
        st.plotly_chart(area_chart)

    elif visualization_type == "Choropleth Map":
        st.subheader("Choropleth Map")
        choropleth_column = st.selectbox("Select Column for Choropleth", df.columns)
        choropleth_map = px.choropleth(df, locations=choropleth_column, locationmode="country names")
        st.plotly_chart(choropleth_map)

    
    st.subheader("Data Transformation")
    cat_cols = []
    for column in df.select_dtypes(include=[np.object_]).columns:
        cat_cols.append(column)
    checkbox_states = {}
    try:
     for cat in cat_cols:
        checkbox_states[cat] = st.checkbox(cat)

     for cat in cat_cols:
        if checkbox_states[cat]:
            cat_strategy = st.selectbox(f'Select the strategy for Categorical values ({cat})',
                                        ['Label Encoder', 'One hot-encoding'],
                                        key=f"selectbox_{cat}"
                                       )

            if cat_strategy == 'Label Encoder':
                encoder = LabelEncoder()
                df[cat] = encoder.fit_transform(df[cat])
            elif cat_strategy == 'One hot-encoding':
                  df = pd.get_dummies(df, columns=[cat], prefix=[cat], dtype=int)
            

    except Exception as e:
      print(e)

    all_categorical_cols = [col for col in df.columns if col in cat_cols]

# Define a custom CSS style to highlight categorical features with a blue background
    highlighted_css = """
    <style>
        .highlight-cat {
            background-color: lightblue !important;
        }
    </style>
"""

    # Apply the custom style to the DataFrame for all categorical columns
    df_styled = df.style.applymap(lambda x: 'background-color: lightblue', subset=all_categorical_cols)

    # Display the DataFrame with the custom style
    st.markdown(highlighted_css, unsafe_allow_html=True)
    st.write("Input features after transforming the Categorical data:", df_styled) 

    clust_df = df.copy() 
    #Feature Selection
    st.subheader("Feature Selection")
    features_X = list(df.columns)
    features_Y = list(df.columns)
    
    cols_X = st.multiselect('Select the input features(X)',
                            features_X,
                           )

    for f in cols_X:
        features_Y.remove(f)

    cols_Y = st.multiselect('Select the output features(Y)',
                            features_Y,
                           )

    X = df[cols_X]
    y = df[cols_Y]

    st.write("Input features:", X, "Output features:", y)

     # normalising the data

    st.subheader("Standarizing the data")


    try:
     # User input for data transformation
      transformation_type = st.selectbox('Select the data transformation type', ['Standardization', 'Normalization'])
      transform_cols = st.multiselect('Select the columns for transformation', cols_X)
    
      if transformation_type == 'Standardization':
        # Standardization
        sc = StandardScaler()
        X[transform_cols] = sc.fit_transform(X[transform_cols])
        y[transform_cols] = sc.transform(y[transform_cols])
      elif transformation_type == 'Normalization':
        # Normalization
        transformation_tech = st.selectbox('Select the data transformation technique', ['MinMaxScaler', 'MaxAbsScaler','Normalizer'])
        if transformation_tech == 'MinMaxScaler':
         min_max_scaler = MinMaxScaler()
         X[transform_cols] = min_max_scaler.fit_transform(X[transform_cols])
         y[transform_cols] = min_max_scaler.transform(y[transform_cols])
        elif transformation_tech == 'MaxAbsScaler':
        # Scaling using MaxAbsScaler
         max_abs_scaler = MaxAbsScaler()
         X[transform_cols] = max_abs_scaler.fit_transform(X[transform_cols])
         y[transform_cols] = max_abs_scaler.transform(y[transform_cols])
        elif transformation_tech == 'Normalizer':
        # Scaling using MaxAbsScaler
         max_abs_scaler = MaxAbsScaler()
         X[transform_cols] = max_abs_scaler.fit_transform(X[transform_cols])
         y[transform_cols] = max_abs_scaler.transform(y[transform_cols])
    

    except Exception as e:
      print(e)
 
    st.write("After Data Transformation: ", X, y)

    clust_X = X.copy()

    st.header("Splitting the data into training and test sets")

    try: 

        train_test_ratio = st.number_input('Enter the test_size', min_value = 0.1, max_value = 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = train_test_ratio, random_state = 2)
        st.write('Rows is train set =', len(y_train))
        st.write('Rows is test set =', len(y_test))

    except Exception as e:
        print(e)
        st.error("There is some error")

    st.write("Training set: ", X_train, y_train)
    st.write("Testing set: ", X_test, y_test)
   
    
    # 1 model function 
    def evaluate_single_model(model, X_test, y_test):
        # Make predictions using the model
        y_pred = model.predict(X_test)
    
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
    
        # Create a dictionary to store the evaluation results
        evaluation_metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': str(cm) 
    }
    
        return evaluation_metrics    
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),  
       
}


    def manual_prediction():
            st.header("Manual Prediction")
            # Create input fields for each feature
            input_features = {}
            for feature in cols_X:
                if feature in all_categorical_cols:
                    # For categorical features, use text_input
                    input_features[feature] = st.text_input(f"Enter value for {feature}")
                else:
                    # For numeric features, use number_input
                    input_features[feature] = st.number_input(f"Enter value for {feature}", min_value=0.0)

            # Create a button to trigger the prediction
            if st.button("Predict"):
                # Create a DataFrame from the user-input data
                user_input_df = pd.DataFrame([input_features])

                # Label encode categorical columns
                for feature in all_categorical_cols:
                    if feature in user_input_df.columns:
                        encoder = LabelEncoder()
                        user_input_df[feature] = encoder.fit_transform(user_input_df[feature])

                # Apply standardization to all columns
                user_input_df[transform_cols] = sc.transform(user_input_df[transform_cols])

                # Make predictions using the selected model
                user_predictions = selected_model.predict(user_input_df)
            
                # Display the predictions to the user
                st.subheader("Predicted Output")
                st.write(user_predictions[0])
    
    
    
    
    
  
    st.markdown("<h1 style='font-family: Arial, sans-serif;'>Modeling and Evaluation</h1>", unsafe_allow_html=True)
    
    modeling_choice = st.radio("Choose modeling type", ["Classification", "Regression", "Clustering"])
    if modeling_choice == "Classification":

        evaluation_choice = st.radio("Choose evaluation option", ["Single Model Evaluation", "Compare All Models"])
        if evaluation_choice == "Single Model Evaluation":
            selected_model_name = st.selectbox("Select a model", list(models.keys()))
        
            if selected_model_name == "K-NN":
                knn_k = st.slider("Select the value of k for K-NN", min_value=1, max_value=100, step=1)
                selected_model = KNeighborsClassifier(n_neighbors=knn_k)
            elif selected_model_name == "Decision Tree":
                decision_tree_param = st.slider("Select the random state for Decision Tree", min_value=1, max_value=100)
                selected_model = DecisionTreeClassifier(random_state=decision_tree_param)
            elif selected_model_name == "Random Forest":
                rf_n_estimators = st.slider("Select the number of estimators for Random Forest", min_value=1, max_value=100, step=1)
                selected_model = RandomForestClassifier(n_estimators=rf_n_estimators)
            else:
                selected_model = models[selected_model_name]
            # Train the selected model
            selected_model.fit(X_train, y_train)
            evaluation_metrics = evaluate_single_model(selected_model, X_test, y_test)
            # Display the evaluation metrics
            st.subheader(f"Model Evaluation Metrics for {selected_model_name}")
            metrics_df = pd.DataFrame(evaluation_metrics, index=[0])
            table_style = """
                <style>
                    th, td {
                        text-align: center;
                    }
                    tr:first-child {
                        display: none;
                    }
                </style>
                """
            st.write(table_style, unsafe_allow_html=True)

            # Display the table without the row index
            st.dataframe(metrics_df)
            

            manual_prediction()        
                    
            
    
    
    
        if evaluation_choice == "Compare All Models":
            # Create an empty dictionary to store evaluation results
            all_model_results = {}

            # Iterate through all models and evaluate each
            for model_name, model in models.items():
            # Train the selected model
                model.fit(X_train, y_train)
                model_results = evaluate_single_model(model, X_test, y_test)
                all_model_results[model_name] = model_results
            # Create a DataFrame for evaluation
            df_evaluation = pd.DataFrame(all_model_results).T

            st.subheader("Comparison of Evaluation Metrics for All Models")
            st.write(df_evaluation)  # Display the evaluation DataFrame
            plot_data = []
            for model_name, metrics_dict in all_model_results.items():
                for metric, value in metrics_dict.items():
                    if metric != 'Confusion Matrix':
                        plot_data.append({
                            "Model": model_name,
                            "Metric": metric,
                            "Value": value
                        })

                df_plot = pd.DataFrame(plot_data)

            st.subheader("Comparison Using a Bar Plot")

            # Create a bar chart using Plotly Express
            fig = px.bar(df_plot, x='Metric', y='Value', color='Model', title='Comparison of Evaluation Metrics', barmode='group')

            # Update y-axis range to emphasize differences
            y_range = [df_plot['Value'].min() - 0.08, df_plot['Value'].max() + 0.03]
            fig.update_yaxes(range=y_range)

            # Show the plot using st.plotly_chart
            st.plotly_chart(fig)

    elif modeling_choice == "Regression":
        
        regression_models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": PolynomialFeatures(degree=2),
        "K-NN": KNeighborsRegressor(15) 
    }
    
        st.header("Regression Modeling")
        regression_evaluation_choice = st.radio("Choose evaluation option", ["Train Single Model", "Compare All Models"])
    
        if regression_evaluation_choice == "Train Single Model":
            selected_regression_model_name = st.selectbox("Select a regression model", list(regression_models.keys()))
            selected_model = regression_models[selected_regression_model_name]
        
            if selected_regression_model_name == "Polynomial Regression":
                poly_degree = st.slider("Select the degree of the polynomial", min_value=1, max_value=10, value=2)
                poly = PolynomialFeatures(degree=poly_degree)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                selected_model = LinearRegression()
                selected_model.fit(X_train_poly, y_train)
                y_pred = selected_model.predict(X_test_poly)
            else:
                selected_model.fit(X_train, y_train)
                y_pred = selected_model.predict(X_test)
        
            # Calculate evaluation metrics for regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader(f"Model Evaluation Metrics for {selected_regression_model_name}")
            st.write(f"R-squared (R2): {r2:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            manual_prediction()
            
        elif regression_evaluation_choice == "Compare All Models":
            all_regression_model_results = {}

            # Iterate through all regression models and evaluate each
            for model_name, model in regression_models.items():
                if model_name == "Polynomial Regression":
                    poly_degree = st.slider("Select the degree of the polynomial", min_value=1, max_value=10, value=2)
                    poly = PolynomialFeatures(degree=poly_degree)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    selected_model = LinearRegression()
                    selected_model.fit(X_train_poly, y_train)
                    y_pred = selected_model.predict(X_test_poly)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                all_regression_model_results[model_name] = {
                    "R-squared (R2)": r2,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                }

            df_regression_evaluation = pd.DataFrame(all_regression_model_results).T

            st.subheader("Comparison of Regression Model Evaluation Metrics")
            st.write(df_regression_evaluation)  # Display the evaluation DataFrame

            plot_data_regression = []
            for model_name, metrics_dict in all_regression_model_results.items():
                for metric, value in metrics_dict.items():
                    plot_data_regression.append({
                        "Model": model_name,
                        "Metric": metric,
                        "Value": value
                    })

            df_plot_regression = pd.DataFrame(plot_data_regression)

    elif modeling_choice == "Clustering":
        st.header("Clustering")
    
        clustering_models = {
            "K-Means": KMeans()
        }
    
        selected_clustering_model_name = st.selectbox("Select a clustering model", list(clustering_models.keys()))
        selected_clustering_model = clustering_models[selected_clustering_model_name]
        
        if selected_clustering_model_name == "K-Means":
                inertia = []
                K = range(1, 10)
                for i in K:
                    kmean = KMeans(n_clusters=i)
                    kmean.fit(clust_X)
                    inertia.append(kmean.inertia_)
        
                fig = px.line(x=K, y=inertia, markers=True)
                fig.update_layout(
                    title="Elbow Method",
                    xaxis_title="Number of Clusters",
                    yaxis_title="Sum of Squared Distance",
                    showlegend=False
                )
        
                st.plotly_chart(fig)

                num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)
                selected_clustering_model = KMeans(n_clusters=num_clusters)
                selected_clustering_model.fit(clust_X)
                
                cluster_labels = selected_clustering_model.predict(clust_X)
                
                
                st.write(pd.Series(cluster_labels).value_counts())
                silhouette_avg = silhouette_score(clust_X, selected_clustering_model.labels_)
                st.write(f"Number of Clusters: {num_clusters}")
                st.write(f"Silhouette Score: {silhouette_avg:.4f}") 
                
                       
                kmeans_df = pd.DataFrame(df)
                kmeans_df['KMeans_Clusters'] = cluster_labels
                st.write(kmeans_df)
                
                # Scatter plot with user-selected features and KMeans labels
                st.subheader("Scatter Plot with KMeans Clusters")
        
                x_axis = st.selectbox("Select the x-axis feature", df.columns)
                y_axis = st.selectbox("Select the y-axis feature", df.columns)
        
                scatter_fig = px.scatter(df, x=x_axis, y=y_axis, color="KMeans_Clusters", title="Scatter Plot with KMeans Clusters")
                scatter_fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    showlegend=True
                )
                st.plotly_chart(scatter_fig)
                
                
                cluster_label_names = {}  # Dictionary to store cluster names based on label

                for cluster_label in range(num_clusters):
                    default_name = f"Cluster {cluster_label}"
                    cluster_name = st.text_input(f"Enter a name for Cluster {cluster_label}:", default_name)
                    cluster_label_names[cluster_label] = cluster_name

                # Calculate and display the number of countries in each cluster
                for cluster_label, cluster_name in cluster_label_names.items():
                    countries_in_cluster = kmeans_df[kmeans_df['KMeans_Clusters'] == cluster_label]['country']
                    st.write(f"Number of {cluster_name} countries: {len(countries_in_cluster)}")

                # Ask the user for a cluster name to display its rows
                selected_cluster_label = st.selectbox("Select a cluster to display its rows:", list(cluster_label_names.keys()))
                selected_cluster_name = cluster_label_names[selected_cluster_label]
                selected_cluster_rows = kmeans_df[kmeans_df['KMeans_Clusters'] == selected_cluster_label]
                # Display the rows of the selected cluster
                st.write(f"Rows in {selected_cluster_name} cluster:")
                st.write(selected_cluster_rows)