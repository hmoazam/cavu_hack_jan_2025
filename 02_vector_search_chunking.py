# Databricks notebook source
# Packages required by all code.
# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
# Versions of open source packages are locked since package authors often make backwards compatible changes
%pip install -qqqq -U \
  databricks-vectorsearch databricks-agents pydantic databricks-sdk mlflow mlflow-skinny `# For agent & data pipeline code` \
  pypdf==4.1.0  `# PDF parsing` \
  markdownify==0.12.1  `# HTML parsing` \
  pypandoc_binary==1.13  `# DOCX parsing` \
  transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0. `# get_recursive_character_text_splitter` \

# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data for creating embeddings and the vector search index

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunk data
# MAGIC
# MAGIC

# COMMAND ----------

from typing import Literal, Optional, Any, Callable
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode
import pyspark.sql.functions as func
from typing import Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import tiktoken
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType

# COMMAND ----------

TABLE_NAME = "wikipedia"
INPUT_DELTA_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_NAME}"
CHUNKED_DELTA_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.wikipedia_chunked"

# COMMAND ----------

spark_df = spark.table(INPUT_DELTA_TABLE)
df = spark_df.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

from transformers import OpenAIGPTTokenizer

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# Function to count the tokens in the document
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# COMMAND ----------

df['Plot'].tail().apply(count_tokens)

# COMMAND ----------

context_window_size = 512

# Function to chunk the text based on the context window size of the llm and an overlap of 5% of the context window size
def chunk_text(text, chunk_size=context_window_size, overlap=context_window_size*0.05):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=context_window_size - (context_window_size*0.05),
                            chunk_overlap=context_window_size*0.05,
                            length_function=len,
                            is_separator_regex=False,
                    )
    return text_splitter.split_text(text)

# COMMAND ----------

def process_text(text, chunk_size=512):
    # Check the token count using the count_tokens function
    token_count = count_tokens(text)
    
    # If the token count exceeds the specified chunk_size, apply chunking
    if token_count > chunk_size:
        print(f"Token count ({token_count}) exceeds {chunk_size}. Applying chunking...")
        return chunk_text(text, chunk_size=chunk_size)
    else:
        print(f"Token count ({token_count}) is within the limit ({chunk_size}). No chunking needed.")
        return [text]  # Return the text as a single chunk


# COMMAND ----------

from pyspark.sql.functions import explode, col
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import pandas_udf

# pandas_udf function for parallelising the process_text function
@pandas_udf(ArrayType(StringType()))
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(process_text)
  
(spark.table(INPUT_DELTA_TABLE)
      .filter(col('Plot').isNotNull())
      .withColumn('chunks', explode(parse_and_split(col('Plot'))))  # Apply the UDF and explode the chunks
      .write.mode('overwrite').saveAsTable(CHUNKED_DELTA_TABLE))

# Display the processed table
display(spark.table(CHUNKED_DELTA_TABLE))

# COMMAND ----------

spark.sql(f"ALTER TABLE {CHUNKED_DELTA_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the vector search index

# COMMAND ----------

VECTOR_INDEX_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{TABLE_NAME}_vector_index"
# EMBEDDING_MODEL_NAME="databricks-gte-large-en"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

index = client.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  source_table_name=CHUNKED_DELTA_TABLE,
  index_name=VECTOR_INDEX_NAME,
  pipeline_type="TRIGGERED",
  primary_key="id",
  embedding_source_column="chunks",
  embedding_model_endpoint_name=EMBEDDING_MODEL_NAME
)

# COMMAND ----------


