from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType
from pyspark_llap.sql.session import HiveWarehouseSession
storage=os.getenv("STORAGE")
spark = SparkSession\
    .builder\
    .appName("CDW-CML-JDBC-Integration")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.security.credentials.hiveserver2.enabled","false")\
    .config("spark.datasource.hive.warehouse.read.via.llap","false")\
    .config("spark.datasource.hive.warehouse.read.jdbc.mode", "client")\
    .config("spark.yarn.access.hadoopFileSystems",s3_bucket)\
    .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
    .getOrCreate()
hive = HiveWarehouseSession.session(spark).build()
hive.showDatabases().show()
hive.sql("SHOW databases").show()
hive.sql("select * from airline_ontime_parquet.flights limit 10").show()

query_string = """
create external table airline_ontime_parquet.flights_external
STORED AS parquet
LOCATION {}/datalake/warehouse/tablespace/external/hive/flights'
AS SELECT * FROM airline_ontime_parquet.flights
"""
.format(storage)
hive.sql(query_string)
