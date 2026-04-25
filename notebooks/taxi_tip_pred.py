# %% [markdown]
# # ADSP 31013 IP01 Big Data and Cloud Computing
# 
# ## Group Project – NYC Yellow Taxi Trips (2024/01/01 – 2025/10/15)
# 
# ### NYC Yellow Taxi Trips – **Tip Rate Classification Model Training**
# 
# This notebook presents the **classification model development pipeline** for **Group 8**. While our previous work focused on predicting the raw fare amount (Regression), this analysis pivots to understanding **passenger tipping behavior** (Classification).
# 
# ---
# 

# %% [markdown]
# **Business Context: Why Classify Tip Rates?**
# 
# Understanding tipping behavior is distinct from predicting the fare. While the fare is determined by distance and regulation, the tip is a behavioral signal reflecting passenger satisfaction, cultural norms, and economic sentiment.
# 
# Classifying trips into "Tip Tiers" (e.g., No Tip vs. Generous Tip) supports:
# 
# * **Driver Incentives:** Identifying trip characteristics (route, weather, time) highly correlated with "Generous" tipping to optimize driver dispatch.
# * **Customer Segmentation:** Profiling high-value passengers for loyalty programs.
# * **Service Quality Monitoring:** A "No Tip" prediction might indicate routes or conditions prone to poor customer experiences (e.g., high congestion).
# 
# ---

# %% [markdown]
# **Business Problem – Multiclass Classification**
# 
# The objective is to predict the **Tip Rate Category** of a trip based on pre-payment information.
# 
# **Target Definition:**
# We define the variable $Tip Rate = \frac{Tip Amount}{Fare Amount}$.
# This continuous variable is discretized into four mutually exclusive classes:
# 
# * **Class 0 (No Tip):** Tip Rate = 0%
# * **Class 1 (Low):** 0% < Tip Rate < 5%
# * **Class 2 (Medium-low):** 5% ≤ Tip Rate < 10%
# * **Class 3 (Standard):** 10% ≤ Tip Rate < 15%
# * **Class 4 (Medium-high):** 15% ≤ Tip Rate < 20%
# * **Class 5 (High):** 20% ≤ Tip Rate < 25%
# * **Class 6 (Generous):** Tip Rate ≥ 25%
# 
# ---

# %% [markdown]
# **Modeling Approach (PySpark MLlib)**
# 
# We employ the same rigorous engineering standards as our regression pipeline:
# 
# 1.  **Label Engineering:** Computing rates and handling edge cases (e.g., zero fares).
# 2.  **Feature Engineering:** Reusing our robust **Weather Binning** and **Spatial Encoding** pipelines.
# 3.  **Model Selection:**
#     * **Logistic Regression (Multinomial):** For baseline performance and interpretability (odds ratios).
#     * **Random Forest Classifier:** To capture non-linear interactions between weather, location, and tipping behavior.
# 
# 4.  **Evaluation:**
# * Accuracy (overall hit rate)
# * **Weighted F1-Score** (crucial for handling class imbalance)
# * Confusion Matrix Analysis
# 
# ---

# %% [markdown]
# ### **0. Environment Setup and Spark Session**
# We import the necessary Spark MLlib classification and evaluation components.

# %%
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, RandomForestClassificationModel


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

import seaborn as sns
import matplotlib.pyplot as plt

# %%
# from IPython.core.display import HTML
# display(HTML("<style>pre { white-space: pre !important; }</style>"))

# %%
spark = SparkSession.builder.appName("NYC-Taxi-Tip-Classification").getOrCreate()

# %% [markdown]
# ### **1. Data Loading & Sanity Checks**
# We load the same cleaned, weather-enriched dataset used in the regression workflow.

# %%
bucket_url = "gs://msca-bdp-student-gcs/group_8_project/datasets/"

df = spark.read.parquet(bucket_url + "cleaned/parquet/")
# print(f"Total records loaded: {df.count():,}")
# df.printSchema()

# %% [markdown]
# ### **2. Target Definition: Tip Rate Binning**
# 
# Unlike the regression model where `fare_amount` was the target, here we must engineer the classification label `tip_class`.
# 
# **Rules:**
# 1. Filter out rows where `fare_amount` <= 0 to avoid division by zero errors.
# 2. Calculate `tip_rate`.
# 3. Bin `tip_rate` into 7 distinct classes.

# %%
df = df.filter(F.col("fare_amount") > 0)

# Create Target Label 'tip_class'
# Logic:
# 0.00      -> Class 0 (No Tip)
# 0.00-0.50 -> Class 1 (Low)
# 0.50-0.10 -> Class 2 (Medium-low)
# 0.10-0.15 -> Class 3 (Standard)
# 0.15-0.20 -> Class 4 (Medium-high)
# 0.20-0.25 -> Class 5 (High)
# 0.25+     -> Class 6 (Generous)

df = df.withColumn(
    "tip_class",
    F.when(F.col("tip_rate") == 0, 0)
    .when(F.col("tip_rate") < 0.05, 1)
    .when(F.col("tip_rate") < 0.10, 2)
    .when(F.col("tip_rate") < 0.15, 3)
    .when(F.col("tip_rate") < 0.20, 4)
    .when(F.col("tip_rate") < 0.25, 5)
    .otherwise(6)
    .cast(IntegerType())
)

# print("Distribution of Tip Classes:")
# df.groupBy("tip_class").count().orderBy("tip_class").show()

# %% [markdown]
# **3. Feature Selection & Prevention of Leakage**
# 
# **Crucial Step:** We must remove any columns that reveal the tip amount or are calculated *after* the trip is completed and paid for (except for the basic trip metrics known at dropoff).
# 
# * **Target:** `tip_class`
# * **Drop:** `tip_amount`, `total_amount`, `tip_rate` (original), and surcharges that might imply total cost directly.
# * **Keep:** Trip physics (distance, duration), Location IDs, Weather, Time (implied via weather/traffic), Vendor/RateCode.

# %%
# Define features to exclude
leakage_cols = [
    "tip_amount",
    "total_amount",
    "tip_rate",
    "tpep_pickup_datetime", # Removing raw timestamps as per original notebook strategy
    "tpep_dropoff_datetime",
    "pu_borough", "pu_zone", "do_borough", "do_zone" # Removing redundant text locations
]

# Billing components (optional to keep as they are known at meter stop, but we exclude to mirror regression strictness)
billing_cols = ["extra", "mta_tax", "fare_amount", "tolls_amount", "improvement_surcharge", "congestion_surcharge", "airport_fee"]

# Select columns
cols_to_keep = [c for c in df.columns if c not in leakage_cols and c not in billing_cols]

# Explicitly ensure our target 'tip_class' is kept, and 'fare_amount' is kept as a feature 
# (Passengers often tip based on the meter fare, so fare_amount is a valid predictor).
if "tip_class" not in cols_to_keep: cols_to_keep.append("tip_class")

df = df.select(cols_to_keep)

# print(f"Modeling Feature Count: {len(df.columns)}")
# print(f"Features: {df.columns}")

# %% [markdown]
# ### **4. Feature Engineering Pipeline**
# 
# We apply the exact same **Weather Binning** logic used in the regression notebook to ensure consistency in how we interpret environmental factors.

# %%
def bucket_temp(col):
    """Temperature in °C: freezing, cold, cool, warm, hot."""
    return (
        F.when(col.isNull(), None)
         .when(col < 0, "freezing")
         .when(col < 10, "cold")
         .when(col < 20, "cool")
         .when(col < 30, "warm")
         .otherwise("hot")
    )

def bucket_dwpt(col):
    """Dew point in °C: comfort-oriented bins."""
    return (
        F.when(col.isNull(), None)
         .when(col <= 0, "very_dry_cold")
         .when(col < 10, "dry")
         .when(col < 16, "comfortable")
         .when(col < 21, "humid")
         .otherwise("oppressive")
    )

def bucket_rhum(col):
    """Relative humidity in %: dry, comfortable, humid, very_humid."""
    return (
        F.when(col.isNull(), None)
         .when(col < 30, "dry")
         .when(col < 60, "comfortable")
         .when(col < 80, "humid")
         .otherwise("very_humid")
    )

def bucket_prcp(col):
    """Precipitation in mm/hour: none, trace, light, moderate, heavy."""
    return (
        F.when(col.isNull(), None)
         .when(col <= 0, "none")
         .when(col < 0.5, "trace")
         .when(col < 4, "light")
         .when(col < 8, "moderate")
         .otherwise("heavy")
    )

def bucket_wdir(col):
    """Wind direction in degrees: 8 cardinal sectors."""
    return (
        F.when(col.isNull(), None)
         .when((col >= 337.5) | (col < 22.5), "N")
         .when(col < 67.5, "NE")
         .when(col < 112.5, "E")
         .when(col < 157.5, "SE")
         .when(col < 202.5, "S")
         .when(col < 247.5, "SW")
         .when(col < 292.5, "W")
         .otherwise("NW")
    )

def bucket_wspd(col):
    """Wind speed (km/h, Meteostat metric): calm, light, moderate, fresh, strong_gale."""
    return (
        F.when(col.isNull(), None)
         .when(col < 2, "calm")
         .when(col < 11, "light")
         .when(col < 28, "moderate")
         .when(col < 49, "fresh")
         .otherwise("strong_gale")
    )

def bucket_pres(col):
    """Pressure in hPa: low, normal, high."""
    return (
        F.when(col.isNull(), None)
         .when(col < 1000, "low")
         .when(col <= 1020, "normal")
         .otherwise("high")
    )

def bucket_coco(col):
    """
    Meteostat weather condition code -> broad categories.
    See: Meteostat COCO codes (1–27).
    """
    return (
        F.when(col.isNull(), "unknown")
         .when(col.isin(1, 2), "clear_fair")
         .when(col.isin(3, 4), "cloudy_overcast")
         .when(col.isin(5, 6), "fog")
         .when(col.isin(7, 8, 9, 17, 18), "rain")
         .when(col.isin(10, 11), "freezing_rain")
         .when(col.isin(12, 13, 19, 20), "mixed_precip")
         .when(col.isin(14, 15, 16, 21, 22), "snow")
         .when(col.isin(23, 25, 26, 27), "thunderstorm")
         .when(col == 24, "hail")
         .otherwise("other")
    )

weather_prefixes = ["pu", "do"]

for prefix in weather_prefixes:
    df = (
        df
        # Temperature & dew point
        .withColumn(f"{prefix}_temp_bin", bucket_temp(F.col(f"{prefix}_temp")))
        .withColumn(f"{prefix}_dwpt_bin", bucket_dwpt(F.col(f"{prefix}_dwpt")))
        # Relative humidity
        .withColumn(f"{prefix}_rhum_bin", bucket_rhum(F.col(f"{prefix}_rhum")))
        # Precipitation
        .withColumn(f"{prefix}_prcp_bin", bucket_prcp(F.col(f"{prefix}_prcp")))
        # Wind direction & speed
        .withColumn(f"{prefix}_wdir_bin", bucket_wdir(F.col(f"{prefix}_wdir")))
        .withColumn(f"{prefix}_wspd_bin", bucket_wspd(F.col(f"{prefix}_wspd")))
        # Pressure
        .withColumn(f"{prefix}_pres_bin", bucket_pres(F.col(f"{prefix}_pres")))
        # Weather condition code
        .withColumn(f"{prefix}_coco_bin", bucket_coco(F.col(f"{prefix}_coco")))
    )

df.select(
    "pu_temp", "pu_temp_bin",
    "pu_dwpt", "pu_dwpt_bin",
    "pu_rhum", "pu_rhum_bin",
    "pu_prcp", "pu_prcp_bin",
    "pu_wdir", "pu_wdir_bin",
    "pu_wspd", "pu_wspd_bin",
    "pu_pres", "pu_pres_bin",
    "pu_coco", "pu_coco_bin"
).show(10, truncate=False)

df = df.drop(*["pu_temp", "pu_dwpt", "pu_rhum", "pu_prcp", "pu_wdir", "pu_wspd", "pu_pres", "pu_coco",
              "do_temp", "do_dwpt", "do_rhum", "do_prcp", "do_wdir", "do_wspd", "do_pres", "do_coco",])

# %% [markdown]
# 
# We construct a Spark ML Pipeline to:
# 1.  **StringIndex** categorical strings (weather bins, flags).
# 2.  **OneHotEncode** these indices.
# 3.  **Assemble** all features (numeric + vectors) into a single `features` vector.

# %%
# 1. Identify Categorical & Numeric Columns
cat_cols = [c for c in df.columns if c.endswith("_bin") or c == "store_and_fwd_flag"]
num_cols = ["passenger_count", "trip_distance", "trip_seconds", "pu_location_id", "do_location_id", "vendor_id", "ratecode_id", "payment_type"]

stages = []

# 2. Indexing and Encoding stages
for col in cat_cols:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[f"{col}_vec"])
    stages += [indexer, encoder]

# 3. Assembler
assembler_inputs = [f"{c}_vec" for c in cat_cols] + num_cols

assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
stages.append(assembler)

# Create the Preparation Pipeline
prep_pipeline = Pipeline(stages=stages)

# Fit and Transform data
print("Fitting Feature Engineering Pipeline...")
prep_model = prep_pipeline.fit(df)
final_df = prep_model.transform(df)

final_df.show(5, truncate=False)

# %% [markdown]
# ### **5. Model Training & Evaluation**
# 
# We will split the data into training (80%) and testing (20%) sets and benchmark two distinct classifiers.
# 
# Before we train, we need to solve the class imbalance problem, we solve it by calculating a new column `class_weight` for each class and then assign it to each row. When we initialize the model, we specify the weight column to it.

# %%
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

N = train_df.count()

print(f"Training Counts: {N:,}")
print(f"Testing Counts:  {test_df.count():,}")

# %%
train_df.groupBy("tip_class").agg(F.count("*").alias("count")).show()

# %%
label_counts = (
    train_df.groupBy("tip_class")
      .agg(F.count("*").alias("count"))
)

K = train_df.select("tip_class").distinct().count()

label_weights = (
    label_counts
    .withColumn("class_weight", (F.lit(N) / (F.lit(K) * F.col("count"))))
)

train_df = train_df.join(label_weights.select("tip_class", "class_weight"), on="tip_class", how="left")
train_df.show()

# %% [markdown]
# #### **5.1 Baseline: Logistic Regression**
# 
# We use Multinomial Logistic Regression as our baseline. It provides a linear decision boundary and is computationally efficient for large datasets.

# %%
# lr = LogisticRegression(
#     featuresCol="features",
#     labelCol="tip_class",
#     family="multinomial",
#     weightCol="class_weight"
# )

# lr_model = lr.fit(train_df)

# # %%
# lr_predictions = lr_model.transform(test_df)

# lr_predictions.select('tip_class','prediction').show(5,truncate=False)

# # %%
# e = MulticlassClassificationEvaluator(labelCol="tip_class", predictionCol="prediction")

# print("Logistic Regression - accuracy: %.3f" % e.evaluate(lr_predictions, {e.metricName: "accuracy"}))
# print("Logistic Regression - f1 score: %.3f" %e.evaluate(lr_predictions, {e.metricName: "f1"}))
# print("Logistic Regression - precision: %.3f" %e.evaluate(lr_predictions, {e.metricName: "weightedPrecision"}))
# print("Logistic Regression - recall: %.3f" %e.evaluate(lr_predictions, {e.metricName: "weightedRecall"}))

# %% [markdown]
# #### **5.2 Advanced: Random Forest Classifier**

# %%
# rf = RandomForestClassifier(
#     featuresCol="features", 
#     labelCol="tip_class",
#     weightCol="class_weight",
#     seed=42
# )

# rf_model = rf.fit(train_df)

# # %%
# rf_predictions = rf_model.transform(test_df)

# rf_predictions.select('tip_class','prediction').show(5,truncate=False)

# # %%
# e = MulticlassClassificationEvaluator(labelCol="tip_class", predictionCol="prediction")

# print("Random Forest - accuracy: %.3f" % e.evaluate(rf_predictions, {e.metricName: "accuracy"}))
# print("Random Forest - f1 score: %.3f" %e.evaluate(rf_predictions, {e.metricName: "f1"}))
# print("Random Forest - precision: %.3f" %e.evaluate(rf_predictions, {e.metricName: "weightedPrecision"}))
# print("Random Forest - recall: %.3f" %e.evaluate(rf_predictions, {e.metricName: "weightedRecall"}))

# %%
# models = [("Logistic Regression", lr_predictions),
#           ("Random Forest", rf_predictions)]

# fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# for ax, (name, df) in zip(axes, models):
#     confusion_pdf = (
#         df
#         .groupBy("tip_class")
#         .pivot("prediction", [0,1,2,3,4,5,6])
#         .count()
#         .na.fill(0)
#         .orderBy("tip_class")
#     ).toPandas()
    
#     confusion_pdf = confusion_pdf.set_index("tip_class")
    
#     sns.heatmap(confusion_pdf, annot=True, fmt="d", cmap="Blues", ax=ax)
#     ax.set_title(f"{name} Confusion Matrix", fontsize=14)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")

# plt.tight_layout()
# plt.show()

# %% [markdown]
# ### Model Comparison Summary
# 
# After applying class weighting to address the significant class imbalance in the taxi tip classification task, both Logistic Regression (LR) and Random Forest (RF) showed notable changes in performance. However, based on the updated metrics and confusion matrices, Random Forest emerges as the more promising model for further refinement.
# 
# **Logistic Regression**, even with class weights, remains limited by its linear decision boundaries. While its weighted F1 score is comparable to Random Forest, the confusion matrix shows that LR continues to distribute many mid-range classes (1–4) across multiple predicted categories with relatively weak class separation. This indicates that LR has largely reached its performance ceiling under the current feature representation, and additional tuning is unlikely to produce substantial improvement.
# 
# **Random Forest**, on the other hand, benefits substantially from class weighting. Its overall accuracy and recall are superior to LR, and its confusion matrix reveals clearer, more structured patterns—especially for minority classes. RF demonstrates an improved ability to distinguish between mid-range tip classes and produces more meaningful decision boundaries across the full range of categories. Unlike LR, RF still has considerable room for improvement through hyperparameter tuning (e.g., increasing tree depth, number of trees, or maxBins).
# 
# **In summary**, class weighting successfully stabilized both models, but Random Forest now shows stronger performance, greater robustness across classes, and significantly more growth potential. Therefore, RF is the recommended model to continue optimizing through targeted hyperparameter tuning.
# 

# %% [markdown]
# ### **6. Hyperparameter Tuning for the Random Forest**

# %%
sample_ratio = 0.001
train_sample = train_df.sample(
    withReplacement=False,
    fraction=sample_ratio,
    seed=42
)

print("Original train size:", train_df.count())
print("Sample train size:", train_sample.count())

# %%
print("=======Start Training=======")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="tip_class",
    weightCol="class_weight",
    seed=42,
    subsamplingRate=0.8
)

paramGrid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [50])
    .addGrid(rf.maxDepth, [8, 12, 16])
    .addGrid(rf.maxBins, [64, 128])
    .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"])
    .build()
)

evaluator = MulticlassClassificationEvaluator(
    labelCol="tip_class",
    predictionCol="prediction",
    metricName="f1"
)

tvs = TrainValidationSplit(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8
)

tvs_model = tvs.fit(train_sample)

rf_best = tvs_model.bestModel
print("Best numTrees:", rf_best.getNumTrees)
print("Best maxDepth:", rf_best.getOrDefault("maxDepth"))
print("Best maxBins:", rf_best.getOrDefault("maxBins"))
print("Best featureSubsetStrategy:", rf_best.getOrDefault("featureSubsetStrategy"))
print("=======Finished Training=======")

# %%
best_maxDepth = rf_best.getOrDefault("maxDepth")
best_maxBins = rf_best.getOrDefault("maxBins")
best_fss = rf_best.getOrDefault("featureSubsetStrategy")

print("=======Fitting the whole training set=======")
rf_final = RandomForestClassifier(
    featuresCol="features",
    labelCol="tip_class",
    weightCol="class_weight",
    seed=42,
    numTrees=50,
    maxDepth=best_maxDepth,
    maxBins=best_maxBins,
    featureSubsetStrategy=best_fss,
    subsamplingRate=0.8
)

rf_final_model = rf_final.fit(train_df)
print("=======Finished fitting=======")

print("=======Saving the model=======")
rf_final_model.write().overwrite().save("gs://msca-bdp-student-gcs/group_8_project/models/tip_rf_model")
print("=======Finished Saving=======")

# %%
rf_final_model = RandomForestClassificationModel.load("gs://msca-bdp-student-gcs/group_8_project/models/tip_rf_model")


rf_final_predictions = rf_final_model.transform(test_df)

rf_final_predictions.select("tip_class", "prediction").show(20, truncate=False)

# %% [markdown]
# **Conclusion: Tuned Multinomial Logistic Regression Performance**
# 
# After performing manual hyperparameter tuning with a custom k-fold cross-validation procedure, the final multinomial Logistic Regression model achieves an accuracy of **0.671** and an overall F1-score of **0.600** on the test set. These results are consistent with, and slightly stronger than, the untuned baseline model, confirming that regularization tuning helps stabilize the classifier without overfitting.
# 
# The precision (**0.575**) and recall (**0.671**) indicate that the model correctly captures the overall distribution of tip-percentage classes, although its performance varies across individual classes. As seen in the sample predictions, the model performs reliably on the more frequent classes (e.g., class 0), while performance remains more challenging on the less frequent high-tip categories (e.g., classes 5 and 6). This imbalance is expected given the distribution of tip classes, and it explains the moderate precision despite relatively strong recall.
# 
# Overall, the tuned Logistic Regression model provides a balanced and interpretable multiclass classifier that performs well under the constraints of the dataset. It successfully captures the main behavioral patterns of tipping, generalizes consistently across folds, and offers a strong baseline for downstream comparison with more complex models.
# 

# %% [markdown]
# ### **7. Post Evaluation**

# %%
# from pyspark.ml.classification import LogisticRegressionModel

# lr = LogisticRegressionModel.load("gs://msca-bdp-student-gcs/group_8_project/models/tip_lr_model")
# pred_df = lr.transform(test_df)
# pred_df.show(5, truncate=False)

# %%


# %%


# %%


# %%



