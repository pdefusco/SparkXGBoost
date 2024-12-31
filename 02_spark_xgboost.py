#****************************************************************************
# (C) Cloudera, Inc. 2020-2024
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os
import mlflow
import numpy as np
import pandas as pd
from pyspark import SparkContext
from datetime import datetime
import dbldatagen as dg
import cml.data_v1 as cmldata
from pyspark.ml.feature import VectorAssembler
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import LongType, FloatType, IntegerType, StringType, \
                              DoubleType, BooleanType, ShortType, \
                              TimestampType, DateType, DecimalType, \
                              ByteType, BinaryType, ArrayType, MapType, \
                              StructType, StructField

# Sample in-code customization of spark configurations
SparkContext.setSystemProperty('spark.executor.cores', '2')
SparkContext.setSystemProperty('spark.executor.memory', '4g')
SparkContext.setSystemProperty('spark.executor.instances', '4')
SparkContext.setSystemProperty('spark.driver.cores', '2')
SparkContext.setSystemProperty('spark.driver.memory', '4g')
SparkContext.setSystemProperty('spark.dynamicAllocation.enabled', 'false')

CONNECTION_NAME = "se-aws-edl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("https://spark-"+os.environ["CDSW_ENGINE_ID"] + "."+ os.environ["CDSW_DOMAIN"])

df = spark.sql("SELECT * FROM spark_catalog.default.biomarkers_table")

# assume the label column is named "class"
label_name = "asthmatic_bronchitis"

# get a list with feature column names
feature_names = [x.name for x in df.schema if x.name != label_name]

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
df = assembler.transform(df)

# Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# create a xgboost pyspark classifier estimator and set device="cuda"
xgb_classifier = SparkXGBClassifier(
  features_col="features",
  label_col=label_name,
  num_workers=4)

"""
xgb_classifier = SparkXGBClassifier(max_depth=5, missing=0.0,
validation_indicator_col='isVal', weight_col='weight',
early_stopping_rounds=1, eval_metric='logloss', num_workers=2)
"""

xgb_clf_model = xgb_classifier.fit(train_data)
predictions = xgb_clf_model.transform(test_data).show()

mlflow.set_experiment("PySpark XGBoost CLF")

# Log the model with MLflow
with mlflow.start_run():
    mlflow.spark.log_model(xgb_clf_model, "xgboost_model")

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol=label_name, predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})

    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)
