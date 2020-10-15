import os
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *


storage = os.getenv("STORAGE")

spark = SparkSession\
  .builder\
  .appName("Airline Data Exploration")\
  .config("spark.executor.memory","8g")\
  .config("spark.executor.cores","4")\
  .config("spark.driver.memory","6g")\
  .config("spark.yarn.access.hadoopFileSystems",storage)\
  .getOrCreate()

flight_df = spark.sql("select * from airline_ontime_parquet.flights_external")

flight_df.persist()

flight_df.printSchema()

sample_normal_flights = flight_df  .filter("cancelled == 0")  .sample(withReplacement=False, fraction=0.03, seed=3)
cancelled_flights = flight_df  .filter("cancelled == 1")
all_flight_data = cancelled_flights.union(sample_normal_flights)
all_flight_data.persist()
all_flight_data = all_flight_data.withColumn("date",to_date(concat_ws("-","year","month","dayofmonth"))).withColumn("week",weekofyear("date"))

all_flight_data = all_flight_data\
  .withColumn(
      'hour', 
      substring(    
          when(length(col("crsdeptime")) == 4,col("crsdeptime")).otherwise(concat(lit("0"),col("crsdeptime")))
      ,1,2).cast('integer')

  )

smaller_all_flight_data = all_flight_data.select( 
  "date",
  "uniquecarrier",
  "flightnum",
  "origin",
  "dest",
  "crsdeptime",
  "crsarrtime",
  "cancelled",
  "crselapsedtime",
  "distance",
  "hour",
  "week"
)

smaller_all_flight_data.printSchema()
smaller_all_flight_data.write.csv(storage + "/datalake/data/airlines/csv/all_data",mode='overwrite',header=True)
#smaller_all_flight_data = spark.read.csv(storage + "/datalake/data/airlines/csv/all_data",inferSchema=True,header=True)
smaller_all_flight_data_pd = smaller_all_flight_data.toPandas()
smaller_all_flight_data_pd.to_csv('data/all_flight_data.csv', index=False )
spark.stop()