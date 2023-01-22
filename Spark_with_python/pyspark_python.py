# -*- coding: utf-8 -*-
"""
Pyspark is a connection between Apache Spark and Python.
It is a Spark Python API and helps you connect with Resilient Distributed Datasets (RDDs) 
to Apache Spark and Python. Let’s talk about the basic concepts of Pyspark RDD, DataFrame, and spark files.
@author: Avinash G
"""
# import dependencies
import pyspark
import pandas as pd

# Before using Pyspark we have to create the spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("pyspark app1").getOrCreate()

df1 = spark.read.csv("test_car_data.csv",header=True, inferSchema=True)

df1.show()

## 2
## Check DataType
df1.dtypes

# Get Describe
df1.describe().show()

# Add the column
df1 = df1.withColumn("Milege increase by 2", df1["Milege"] + 2)

# Drop the Column
df1 = df1.drop("Milege increase by 2")

df1.show()

df11 = df1

# Rename the Column
df1.withColumnRenamed("Milege", "Avg Milege").show()

# Drop the Row with respect to Nan Values
df1.na.drop().show()

df1.printSchema()

# Drop using "any"
df1.na.drop(how="any").show()

# drop using Threshold
# thresh = 3 means atleast 3 non nan values are present in row so then row not drop otherwise it drop
df1.na.drop(how="any", thresh=3).show() 

# Drop using Subset
# Drop nan values only on specific column
df1.na.drop(how="any", subset=["Km"]).show()

# Fill the missing values
# Where the Nan Values are present it will replaced with "Missing Value" word
df1.na.fill("Missing Value").show() 

# Where the Nan Values are present it will replaced with "Missing Value" word on particular column
df1.na.fill("0", "Km").show() 

# Fill nan Values with mean
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["Milege"],
    outputCols=["{}_inputed".format(c) for c in ["Milege"]]
).setStrategy("median") # Change the setStrategy with median and mode also


imputer.fit(df1).transform(df1).show()

imputed_data=imputer.fit(df1).transform(df1)

### 3
# PySpark DataFrame
# Filter Operation pyspark


#Filter Operation pyspark
# Milege of the Passenger less than or equal to 15
df1.filter("Milege<=15").show()

filter_df1=df1.filter("Milege<=15")

#As you can see that Milege Column all the values under Milege Column is less than or equal to 15.

df1.filter("Milege<=15").select("Model", "Fuel_Type").show()

# create new dataframe for filter result store
filter_df2=df1.filter("Milege<=15").select("Model", "Fuel_Type")

# show Milege not less than or equal to 15 records
filter_df2=df1.filter(~("Milege<=15")).show()


####### 4
#GroupBy Operation using pyspark

# Grouped to find the maximum salary
df1.groupBy("Fuel_Type").sum().show()

fuel_group_df=df1.groupBy("Fuel_Type").sum()


# Groupby Departments which gives maximum Kilometer
df1.groupBy("Km").sum().show()

df1.groupBy("Model").mean().show()

# Grouped Departments to count the each car Model
df1.groupBy("Model").count().show()

# which Person has maximum Kilometer
df1.groupBy("Km").max().show()


# total sum of Kilometer
df1.agg({"Km":"sum"}).show()

##### 5
#PySpark Tutorial | PySpark ML | Empoyee Data set
dfe = spark.read.csv("emp_data.csv",header=True, inferSchema=True)

dfe.columns

#In PySpark we have to Grouped independent feature together [Exp,Salary] and convert that into a new column
from pyspark.ml.feature import VectorAssembler
feature_assember = VectorAssembler(inputCols=["Exp","Salary"], outputCol="Independent Features")

output = feature_assember.transform(dfe)
output.show()

finalized_data = output.select("Independent Features", "Salary")
finalized_data.show()

from pyspark.ml.regression import LinearRegression

# split train test data
train_data,test_data=finalized_data.randomSplit([0.75,0.25])

# Train the LinearRegression Model
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Salary')
regressor=regressor.fit(train_data)


# Coefficients
regressor.coefficients

# Intercept
regressor.intercept

# Prediction
pred_results = regressor.evaluate(test_data)

pred_results.predictions.show()

pred_results.meanAbsoluteError,pred_results.meanSquaredError


