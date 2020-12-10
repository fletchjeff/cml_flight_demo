# Part 1: Data Ingest
# A data scientist should never be blocked in getting data into their environment,
# so CML is able to ingest data from many sources.
# Whether you have data in .csv files, modern formats like parquet or feather,
# in cloud storage or a SQL database, CML will let you work with it in a data
# scientist-friendly environment.

# Access local data on your computer
#
# Accessing data stored on your computer is a matter of [uploading a file to the CML filesystem and
# referencing from there](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-local-data-from-your-computer.html).
#
# > Go to the project's **Overview** page. Under the **Files** section, click **Upload**, select the relevant data files to be uploaded and a destination folder.
#
# If, for example, you upload a file called, `mydata.csv` to a folder called `data`, the
# following example code would work.

# ```
# import pandas as pd
#
# df = pd.read_csv('data/mydata.csv')
#
# # Or:
# df = pd.read_csv('/home/cdsw/data/mydata.csv')
# ```

# Access data in S3
#
# Accessing [data in Amazon S3](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-data-in-amazon-s3-buckets.html)
# follows a familiar procedure of fetching and storing in the CML filesystem.
# > Add your Amazon Web Services access keys to your project's
# > [environment variables](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-environment-variables.html)
# > as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
#
# To get the the access keys that are used for your in the CDP DataLake, you can follow
# [this Cloudera Community Tutorial](https://community.cloudera.com/t5/Community-Articles/How-to-get-AWS-access-keys-via-IDBroker-in-CDP/ta-p/295485)

#
# The following sample code would fetch a file called `myfile.csv` from the S3 bucket, `data_bucket`, and store it in the CML home folder.
# ```
# # Create the Boto S3 connection object.
# from boto.s3.connection import S3Connection
# aws_connection = S3Connection()
#
# # Download the dataset to file 'myfile.csv'.
# bucket = aws_connection.get_bucket('data_bucket')
# key = bucket.get_key('myfile.csv')
# key.get_contents_to_filename('/home/cdsw/myfile.csv')
# ```


# Access data from Cloud Storage or the Hive metastore
#
# Accessing data from [the Hive metastore](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-data-from-apache-hive.html)
# that comes with CML only takes a few more steps.
# But first we need to fetch the data from Cloud Storage and save it as a Hive table.
#
# > Specify `STORAGE` as an
# > [environment variable](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-environment-variables.html)
# > in your project settings containing the Cloud Storage location used by the DataLake to store
# > Hive data. On AWS it will `s3a://[something]`, on Azure it will be `abfs://[something]` and on
# > on prem CDSW cluster, it will be `hdfs://[something]`
#
# This was done for you when you ran `0_bootstrap.py`, so the following code is set up to run as is.
# It begins with imports and creating a `SparkSession`.

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

#    .config("spark.dynamicAllocation.enabled","true")\
#    .config("spark.shuffle.service.enabled","true")\

storage = os.environ['STORAGE']

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","2")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances","4")\
    .config("spark.yarn.access.hadoopFileSystems",storage)\
    .getOrCreate()

# **Note:**
# Our file isn't big, so running it in Spark local mode is fine but you can add the following config
# if you want to run Spark on the kubernetes cluster
#
# > .config("spark.yarn.access.hadoopFileSystems",os.getenv['STORAGE'])\
#
# and remove `.master("local[*]")\`
#

# Since we know the data already, we can add schema upfront. This is good practice as Spark will
# read *all* the Data if you try infer the schema.

#!hdfs dfs -copyFromLocal /home/cdsw/data/all_flight_data.csv $STORAGE/datalake/data/flights-external/all_flight_data.csv
#```
#month,dayofmonth,dayofweek,deptime,crsdeptime,arrtime,crsarrtime,uniquecarrier,flightnum,tailnum,actualelapsedtime,crselapsedtime,airtime,arrdelay,depdelay,origin,dest,distance,taxiin,taxiout,cancelled,cancellationcode,diverted,carrierdelay,weatherdelay,nasdelay,securitydelay,lateaircraftdelay,year
#12,8,1,1123,1124,1559,1550,NW,921,N240NW,516,506,493,9,-1,MSP,HNL,3972,5,18,0,"",0,0,0,0,0,0,2003
#12,7,7,1124,1124,1550,1550,NW,921,N229NW,506,506,482,0,0,MSP,HNL,3972,3,21,0,"",0,0,0,0,0,0,2003
#```
set_1_schema = StructType(
    [
      StructField("month", DoubleType(), True),
      StructField("dayofmonth", DoubleType(), True),
      StructField("dayofweek", DoubleType(), True),
      StructField("deptime", DoubleType(), True),
      StructField("crsdeptime", DoubleType(), True),
      StructField("arrtime", DoubleType(), True),
      StructField("crsarrtime", DoubleType(), True),
      StructField("uniquecarrier", StringType(), True),
      StructField("flightnum", DoubleType(), True),
      StructField("tailnum", StringType(), True),
      StructField("actualelapsedtime", DoubleType(), True),
      StructField("crselapsedtime", DoubleType(), True),
      StructField("airtime", DoubleType(), True),
      StructField("arrdelay", DoubleType(), True),
      StructField("depdelay", DoubleType(), True),
      StructField("origin", StringType(), True),
      StructField("dest", StringType(), True),
      StructField("distance", DoubleType(), True),
      StructField("taxiin", DoubleType(), True),
      StructField("taxiout", DoubleType(), True),
      StructField("cancelled", DoubleType(), True),
      StructField("cancellationcode", StringType(), True),
      StructField("diverted", DoubleType(), True),
      StructField("carrierdelay", DoubleType(), True),
      StructField("weatherdelay", DoubleType(), True),
      StructField("nasdelay", DoubleType(), True),
      StructField("securitydelay", DoubleType(), True),
      StructField("lateaircraftdelay", DoubleType(), True),
      StructField("year", DoubleType(), True)
    ]
)

# Now we can read in the data from Cloud Storage into Spark...


flights_data_1 = spark.read.csv(
    "{}/datalake/data/flight_data/set_1/".format(
        storage),
    header=True,
    #inferSchema = True,
    schema=set_1_schema,
    sep=','
)

# ...and inspect the data.

flights_data_1.show()

flights_data_1.printSchema()

#```
#FL_DATE,OP_CARRIER,OP_CARRIER_FL_NUM,ORIGIN,DEST,CRS_DEP_TIME,DEP_TIME,DEP_DELAY,TAXI_OUT,WHEELS_OFF,WHEELS_ON,TAXI_IN,CRS_ARR_TIME,ARR_TIME,ARR_DELAY,CANCELLED,CANCELLATION_CODE,DIVERTED,CRS_ELAPSED_TIME,ACTUAL_ELAPSED_TIME,AIR_TIME,DISTANCE,CARRIER_DELAY,WEATHER_DELAY,NAS_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY,Unnamed: 27
#2009-01-01,XE,1204,DCA,EWR,1100,1058.0,-2.0,18.0,1116.0,1158.0,8.0,1202,1206.0,4.0,0.0,,0.0,62.0,68.0,42.0,199.0,,,,,,
#2009-01-01,XE,1206,EWR,IAD,1510,1509.0,-1.0,28.0,1537.0,1620.0,4.0,1632,1624.0,-8.0,0.0,,0.0,82.0,75.0,43.0,213.0,,,,,,
#```

set_2_schema = StructType([StructField("FL_DATE", DateType(), True),
    StructField("OP_CARRIER", StringType(), True),
    StructField("OP_CARRIER_FL_NUM", StringType(), True),
    StructField("ORIGIN", StringType(), True),
    StructField("DEST", StringType(), True),
    StructField("CRS_DEP_TIME", StringType(), True),
    StructField("DEP_TIME", StringType(), True),
    StructField("DEP_DELAY", DoubleType(), True),
    StructField("TAXI_OUT", DoubleType(), True),
    StructField("WHEELS_OFF", StringType(), True),
    StructField("WHEELS_ON", StringType(), True),
    StructField("TAXI_IN", DoubleType(), True),
    StructField("CRS_ARR_TIME", StringType(), True),
    StructField("ARR_TIME", StringType(), True),
    StructField("ARR_DELAY", DoubleType(), True),
    StructField("CANCELLED", DoubleType(), True),
    StructField("CANCELLATION_CODE", StringType(), True),
    StructField("DIVERTED", DoubleType(), True),
    StructField("CRS_ELAPSED_TIME", DoubleType(), True),
    StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
    StructField("AIR_TIME", DoubleType(), True),
    StructField("DISTANCE", DoubleType(), True),
    StructField("CARRIER_DELAY", DoubleType(), True),
    StructField("WEATHER_DELAY", DoubleType(), True),
    StructField("NAS_DELAY", DoubleType(), True),
    StructField("SECURITY_DELAY", DoubleType(), True),
    StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True)])


flights_data_2 = spark.read.csv(
    "{}/datalake/data/flight_data/set_2/".format(
        storage),
    schema=set_2_schema,
    header=True,
    sep=',',
    nullValue='NA'
)

flights_data_2.show()

flights_data_1 = flights_data_1\
  .withColumn("FL_DATE",
              to_date(
                concat_ws('-',col("year"),col("month"),col("dayofmonth")),
              'yyyy.0-MM.0-dd.0')
                )

flights_data_1 = flights_data_1\
.withColumnRenamed("deptime","DEP_TIME")\
.withColumnRenamed("crsdeptime","CRS_DEP_TIME")\
.withColumnRenamed("arrtime","ARR_TIME")\
.withColumnRenamed("crsarrtime","CRS_ARR_TIME")\
.withColumnRenamed("uniquecarrier","OP_CARRIER")\
.withColumnRenamed("flightnum","OP_CARRIER_FL_NUM")\
.withColumnRenamed("actualelapsedtime","ACTUAL_ELAPSED_TIME")\
.withColumnRenamed("crselapsedtime","CRS_ELAPSED_TIME")\
.withColumnRenamed("airtime","AIR_TIME")\
.withColumnRenamed("arrdelay","ARR_DELAY")\
.withColumnRenamed("depdelay","DEP_DELAY")\
.withColumnRenamed("origin","ORIGIN")\
.withColumnRenamed("dest","DEST")\
.withColumnRenamed("distance","DISTANCE")\
.withColumnRenamed("taxiin","TAXI_IN")\
.withColumnRenamed("taxiout","TAXI_OUT")\
.withColumnRenamed("cancelled","CANCELLED")\
.withColumnRenamed("cancellationcode","CANCELLATION_CODE")\
.withColumnRenamed("diverted","DIVERTED")\
.withColumnRenamed("carrierdelay","CARRIER_DELAY")\
.withColumnRenamed("weatherdelay","WEATHER_DELAY")\
.withColumnRenamed("nasdelay","NAS_DELAY")\
.withColumnRenamed("securitydelay","SECURITY_DELAY")\
.withColumnRenamed("lateaircraftdelay","LATE_AIRCRAFT_DELAY")


flights_data_1 = flights_data_1.select(["FL_DATE","DEP_TIME","CRS_DEP_TIME","ARR_TIME","CRS_ARR_TIME","OP_CARRIER","OP_CARRIER_FL_NUM","ACTUAL_ELAPSED_TIME","CRS_ELAPSED_TIME","AIR_TIME","ARR_DELAY","DEP_DELAY","ORIGIN","DEST","DISTANCE","TAXI_IN","TAXI_OUT","CANCELLED","CANCELLATION_CODE","DIVERTED","CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"])
flights_data_2 = flights_data_2.select(["FL_DATE","DEP_TIME","CRS_DEP_TIME","ARR_TIME","CRS_ARR_TIME","OP_CARRIER","OP_CARRIER_FL_NUM","ACTUAL_ELAPSED_TIME","CRS_ELAPSED_TIME","AIR_TIME","ARR_DELAY","DEP_DELAY","ORIGIN","DEST","DISTANCE","TAXI_IN","TAXI_OUT","CANCELLED","CANCELLATION_CODE","DIVERTED","CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"])

flights_data_all = flights_data_1.unionByName(flights_data_2)

# Now we can store the Spark DataFrame as a file in the local CML file system
# *and* as a table in Hive used by the other parts of the project.

#flights_data.coalesce(1).write.csv(
#    "file:/home/cdsw/raw/telco-data/",
#    mode='overwrite',
#    header=True
#)

spark.sql("show databases").show()

spark.sql("show tables in default").show()

# Create the Hive table
# This is here to create the table in Hive used be the other parts of the project, if it
# does not already exist.

if ('flights_data_all' not in list(spark.sql("show tables in default").toPandas()['tableName'])):
    print("creating the flights_data_all database")
    flights_data_all\
        .write.format("parquet")\
        .mode("overwrite")\
        .saveAsTable(
            'default.flights_data_all'
        )

# Show the data in the hive table
spark.sql("select * from default.flights_data_all").show()

# To get more detailed information about the hive table you can run this:
spark.sql("describe formatted default.flights_data_all").toPandas()

# Other ways to access data

# To access data from other locations, refer to the
# [CML documentation](https://docs.cloudera.com/machine-learning/cloud/import-data/index.html).

# Scheduled Jobs
#
# One of the features of CML is the ability to schedule code to run at regular intervals,
# similar to cron jobs. This is useful for **data pipelines**, **ETL**, and **regular reporting**
# among other use cases. If new data files are created regularly, e.g. hourly log files, you could
# schedule a Job to run a data loading script with code like the above.

# > Any script [can be scheduled as a Job](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html).
# > You can create a Job with specified command line arguments or environment variables.
# > Jobs can be triggered by the completion of other jobs, forming a
# > [Pipeline](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-pipeline.html)
# > You can configure the job to email individuals with an attachment, e.g. a csv report which your
# > script saves at: `/home/cdsw/job1/output.csv`.

# Try running this script `1_data_ingest.py` for use in such a Job.


## Hive 3 code
# from __future__ import print_function
# import os
# import sys
# from pyspark.sql import SparkSession
# from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType
# from pyspark_llap.sql.session import HiveWarehouseSession
# storage=os.getenv("STORAGE")
# spark = SparkSession\
#     .builder\
#     .appName("CDW-CML-JDBC-Integration")\
#     .config("spark.executor.memory","8g")\
#     .config("spark.executor.cores","4")\
#     .config("spark.driver.memory","6g")\
#     .config("spark.security.credentials.hiveserver2.enabled","false")\
#     .config("spark.datasource.hive.warehouse.read.via.llap","false")\
#     .config("spark.datasource.hive.warehouse.read.jdbc.mode", "client")\
#     .config("spark.yarn.access.hadoopFileSystems",s3_bucket)\
#     .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
#     .getOrCreate()
# hive = HiveWarehouseSession.session(spark).build()
# hive.showDatabases().show()
# hive.sql("SHOW databases").show()
# hive.sql("select * from airline_ontime_parquet.flights limit 10").show()

# query_string = """
# create external table airline_ontime_parquet.flights_external
# STORED AS parquet
# LOCATION {}/datalake/warehouse/tablespace/external/hive/flights'
# AS SELECT * FROM airline_ontime_parquet.flights
# """
# .format(storage)
# hive.sql(query_string)
