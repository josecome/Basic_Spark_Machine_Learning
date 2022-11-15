#Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

#Creating app
spark=SparkSession.builder.appName('Basic_Spark_ML').getOrCreate()

#Get dataset
ds = spark.read.csv('iris.csv',header=True,inferSchema=True)

#Displaying data
ds.show()

#Assign numbers for categories
ds = ds.withColumn("variety", when(ds.variety == "Setosa","1") \
      .when(ds.variety == "Versicolor","2") \
      .when(ds.variety == "Virginica","3") \
      .otherwise(ds.variety))

#Cast variety to number
ds = ds.withColumn("variety",col("variety").cast(IntegerType()))

#Display Schema of dataset
ds.printSchema()

#Displays columns
ds.columns

#Functition to concatenate columns in each row to predict 'variety'
featureassembler = VectorAssembler(inputCols = ["sepal_length","sepal_width","petal_length","petal_width"],outputCol = "features")

#Concatenate columns 
output=featureassembler.transform(ds)

#Display reesult after run VectorAssembler
output.show()

output.columns

#Only get feature and variety as feature has data from all other columns
ml_data = output.select("features","variety")
ml_data.show()



#Let split our dataset having 80% for trainning and 20% for test
train_data,test_data = ml_data.randomSplit([0.8,0.2])

#Let define our linear regression function
regressor = LinearRegression(featuresCol='features', labelCol='variety')

#Trainning data
regressor = regressor.fit(train_data)

#Coefficients
regressor.coefficients

#Intercepts
regressor.intercept

#Prediction using test data
predition = regressor.evaluate(test_data)

#Display results
predition.predictions.show()


predition.meanAbsoluteError,predition.meanSquaredError