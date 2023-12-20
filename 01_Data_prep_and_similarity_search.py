# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")
import os, os.path
import glob
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pyspark.pandas as ps
import torch
import pickle
from PIL import Image
import requests
from io import BytesIO
import base64

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS NRF_CATALOG_AVI;
# MAGIC USE CATALOG nrf_catalog_avi;
# MAGIC CREATE DATABASE IF NOT EXISTS nrf_base;
# MAGIC USE nrf_BASE;
# MAGIC CREATE VOLUME IF not exists product_images;

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset is from the following Kaggle link: https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip -o "/Volumes/nrf_catalog_avi/nrf_base/product_images/all_images.zip" -d "/Volumes/nrf_catalog_avi/nrf_base/product_images"

# COMMAND ----------

# MAGIC %sh
# MAGIC find "/Volumes/nrf_catalog_avi/nrf_base/product_images/all_images" -type f | wc -l

# COMMAND ----------

directory = "/Volumes/nrf_catalog_avi/nrf_base/product_images/all_images"
number_of_files = len(os.listdir(directory))
print(number_of_files)

# COMMAND ----------

item_details = spark.sql("SELECT * FROM nrf_catalog_avi.nrf_base.fashion").toPandas()
display(item_details)

# COMMAND ----------

image_names = item_details.Image.to_list()
indexable_images = [directory+'/'+ str(name) for name in image_names]

# COMMAND ----------

# MAGIC %md
# MAGIC Find which of the images in the previously unzipped images folder have corresponding recipes in the recipes dataframe. We want a 1:1 relationship here

# COMMAND ----------

img_names = list(glob.glob(directory+'/*.jpg'))

# COMMAND ----------

#get the actual indexable images that are in the images folder
final_index_images = []
for image in indexable_images:
  if image in img_names:
    final_index_images.append(image)
len(final_index_images)

# COMMAND ----------

df = spark.createDataFrame(pd.DataFrame(final_index_images, columns = ['Path']))
display(df)

# COMMAND ----------

# Register DataFrame as a temporary view
df.createOrReplaceTempView("temp_view")

# Use SQL to write the data into a table
spark.sql("CREATE TABLE image_paths AS SELECT * FROM temp_view")


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 1. Computing embeddings from scratch (pickled embedding file is provided following this section)
# MAGIC The cells below show how to compute the embeddings in a distributed manner and build an index during the run of this notebook

# COMMAND ----------

##Create embeddings
#Load the ViT-Clip model
model = SentenceTransformer('clip-ViT-B-32')

# COMMAND ----------

#Define function for embedding calculation
from io import BytesIO
#https://stackoverflow.com/questions/56880941/how-to-fix-attributeerror-jpegimagefile-object-has-no-attribute-read

def get_embeddings(img_loc):
  with open(img_loc, 'rb') as f:
    return model.encode(Image.open(BytesIO(f.read())).convert('RGB'), batch_size=128, convert_to_tensor=False, show_progress_bar=False)

# COMMAND ----------

display(df)

# COMMAND ----------

#Compute embeddings in a distributed manner using the pandas on pyspark API
ps.set_option('compute.default_index_type', 'distributed')
df_ = df.pandas_api()

# COMMAND ----------

df_['embeddings'] = df_['Path'].apply(get_embeddings)

# COMMAND ----------

#Save the embeddings to delta and create table in database
df_sp = df_.to_spark()
display(df_sp)

# COMMAND ----------

# Register DataFrame as a temporary view
df_sp.createOrReplaceTempView("temp_view_2")

# Use SQL to write the data into a table
spark.sql("CREATE TABLE image_embeddings AS SELECT * FROM temp_view_2")

# COMMAND ----------

#Convert embeddings to a format amnenable for indexing with sentence transformers
df_pd = df_.to_pandas()
np_array_list = df_pd.embeddings.to_list()
img_emb = torch.tensor(np_array_list)

# COMMAND ----------

# Next, we define a search function.
def search(query, k=3):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
    
    print("Query:")

    return [final_index_images[hit['corpus_id']] for hit in hits]

# COMMAND ----------

# MAGIC %md
# MAGIC Save the image paths and corresponding image embeddings as pickle files for ease of reproducibility

# COMMAND ----------

embedding_paths = df_pd.Path.to_list()

# COMMAND ----------

final_index_imagespicke_path = "/Volumes/nrf_catalog_avi/nrf_base/product_images/pickled_paths_embeddings_final.pkl"
with open(final_index_imagespicke_path, "wb") as fOut:
    pickle.dump({'image_paths':embedding_paths ,'embeddings': img_emb},fOut)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2. Load the paths and image embeddings saved as a pickle file for reproducibility

# COMMAND ----------

final_index_imagespicke_path = "/Volumes/nrf_catalog_avi/nrf_base/product_images/pickled_paths_embeddings_final.pkl"

pickle_location = final_index_imagespicke_path

# COMMAND ----------

unpickled = pickle.load(open(pickle_location, "rb"))

# COMMAND ----------

embeddings = unpickled['embeddings']
paths = unpickled['image_paths']
#paths = embedding_paths
#embeddings = img_emb
paths[:3]

# COMMAND ----------

len(paths), len(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Convert images into bs64 encoded strings
# MAGIC Convert each image into a bs64 encoded string (put it in a dictionary) and pickle it alongside the embeddings. This is such that the embedding, path and bs64 encoded string order is preserved

# COMMAND ----------

def get_bs64(image_path):
  img = Image.open(image_path).convert("RGB")
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()).decode()
  return img_str

# COMMAND ----------

#Check to make sure the above step worked
# get_bs64(paths[3])

# COMMAND ----------

img_bs64_list = []
for path in paths:
  bs64_img = get_bs64(path)
  img_bs64_list.append(bs64_img)

# COMMAND ----------

index_images_pickle_path = "/Volumes/nrf_catalog_avi/nrf_base/product_images/pickled_paths_embeddings_bs64.pkl"
with open(index_images_pickle_path, "wb") as fOut:
    pickle.dump({'image_paths':paths ,'embeddings': embeddings, 'bs64_images':img_bs64_list},fOut)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. Define function to lookup query string or image (multi-modal model) in the indexed embeddings
# MAGIC And return the corresponding image paths which can be rendered

# COMMAND ----------

#Load the ViT-Clip model
model = SentenceTransformer('clip-ViT-B-32')

index_images_pickle_path = "/Volumes/nrf_catalog_avi/nrf_base/product_images/pickled_paths_embeddings_bs64.pkl"
loaded_pickle = pickle.load(open(index_images_pickle_path, "rb"))
paths = loaded_pickle['image_paths']
embeddings = loaded_pickle['embeddings']
img_bs64_list = loaded_pickle['bs64_images']

# COMMAND ----------

# Next, we define a search function.
def search(query, k=3):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, embeddings, top_k=k)[0]
    
    print("Query:")

    return [img_bs64_list[hit['corpus_id']] for hit in hits]

# COMMAND ----------

results = search("Kid's shoes")

# COMMAND ----------

renders = []
for result in results:
  im = Image.open(BytesIO(base64.b64decode(str(result)))).convert('RGB')
  renders.append(im)
display(renders[0], renders[1], renders[2])

# COMMAND ----------


