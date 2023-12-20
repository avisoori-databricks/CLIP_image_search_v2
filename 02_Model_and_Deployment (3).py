# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 0: Installing and loading all the required libraries

# COMMAND ----------

import os, os.path
import glob
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pyspark.pandas as ps
import torch
import pickle
from PIL import Image
import mlflow.pyfunc
from sys import version_info
import numpy as np
import cloudpickle
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import base64
from io import BytesIO
import json
import pandas as pd

# COMMAND ----------

#Load the ViT-Clip model
model = SentenceTransformer('clip-ViT-B-32')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS NRF_CATALOG_AVI;
# MAGIC USE CATALOG nrf_catalog_avi;
# MAGIC CREATE DATABASE IF NOT EXISTS nrf_base;
# MAGIC USE nrf_BASE;
# MAGIC CREATE VOLUME IF not exists product_images;

# COMMAND ----------

pickle_location = "/Volumes/nrf_catalog_avi/nrf_base/product_images/pickled_paths_embeddings_bs64.pkl"

# COMMAND ----------

loaded_pickle = pickle.load(open(pickle_location, "rb"))

# COMMAND ----------

paths = loaded_pickle['image_paths']
embeddings = loaded_pickle['embeddings']
img_bs64_list = loaded_pickle['bs64_images']

assert len(paths) == len(embeddings) == len(img_bs64_list)

# COMMAND ----------

paths[:3]

# COMMAND ----------

model_input = img_bs64_list[:1][0]

# COMMAND ----------

#Extracting the image name from the paths
paths[0].split('/')[-1]

# COMMAND ----------

item_details = spark.sql("SELECT * FROM nrf_catalog_avi.nrf_base.fashion").toPandas()
display(item_details)

# COMMAND ----------

item_details = item_details.drop(['ImageURL'], axis=1)
display(item_details)

# COMMAND ----------

item_details.columns

# COMMAND ----------

item_details['Information'] = item_details.apply(lambda row: {'Gender': row['Gender'], 'Category': row['Category'], 'SubCategory': row['SubCategory'], 'ProductType': row['ProductType'], 'Color': row['Colour'], 'Usage': row['Usage'], 'ProductTitle': row['ProductTitle']}, axis=1)

# COMMAND ----------

item_details = item_details[['ProductId','Information', 'Image']]
display(item_details)

# COMMAND ----------

product_status = spark.sql("SELECT * FROM nrf_catalog_avi.nrf_base.product_status").toPandas()
display(product_status)

# COMMAND ----------

refined_status = product_status.groupby("ProductId", as_index=False).agg(list) 
display(refined_status)

# COMMAND ----------

merged_item_details = pd.merge(item_details, refined_status, on="ProductId") 
display(merged_item_details)

# COMMAND ----------

merged_item_details.to_csv('/Volumes/nrf_catalog_avi/nrf_base/product_images/item_details.csv')

# COMMAND ----------

list(merged_item_details.columns)

# COMMAND ----------

clip_vit_model_path = "/Volumes/nrf_catalog_avi/nrf_base/product_images/model/"
product_path = '/Volumes/nrf_catalog_avi/nrf_base/product_images/item_details.csv'

# COMMAND ----------

model.save(clip_vit_model_path)

# COMMAND ----------



# COMMAND ----------

artifacts = {
  "clip_vit_model_path":clip_vit_model_path,
  "pickle_location": pickle_location,
  "product_path": product_path
}

# COMMAND ----------

artifacts

# COMMAND ----------

text_input = img_bs64_list[:1][0]

payload_pd = pd.DataFrame([[text_input]],columns=['text'])
payload_pd

# COMMAND ----------

import base64
img_path = "/Volumes/nrf_catalog_avi/nrf_base/product_images/tshirt.jpeg"
image  = Image.open(img_path)
buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())
# img_str

# COMMAND ----------

example_np = np.array(model_input)

# COMMAND ----------

from PIL import Image
from sentence_transformers import SentenceTransformer, util
import base64
from io import BytesIO
import json
import pandas as pd

image_string = base64.b64decode(img_str)
image_string = Image.open(BytesIO(image_string)).convert('RGB')
display(image_string)

# COMMAND ----------

#These all work
# paths = [path.split('/')[-1] for path in pickle.load(open(artifacts["pickle_location"], "rb"))['image_paths']]
# paths[:3]
# embeddings = pickle.load(open(artifacts["pickle_location"], "rb"))['embeddings']
# embeddings[:3]
# img_bs64_list = pickle.load(open(artifacts["pickle_location"], "rb"))['bs64_images']
# img_bs64_list[:3]

# COMMAND ----------

# model = SentenceTransformer(artifacts["clip_vit_model_path"])
# #Create a dict with image paths as keys that would allow the similar items to be looked up in O(1)
# product_dict = pd.read_csv(artifacts["product_path"])[['ProductId','Information','Image','Week','units_sold','units_onhand','status','OOS']].set_index('ProductId').T.to_dict()
# paths = [path.split('/')[-1] for path in pickle.load(open(artifacts["pickle_location"], "rb"))['image_paths']]
# embeddings = pickle.load(open(artifacts["pickle_location"], "rb"))['embeddings']
# img_bs64_list = pickle.load(open(artifacts["pickle_location"], "rb"))['bs64_images']

# COMMAND ----------

# query_emb = model.encode([image_string], convert_to_tensor=True, show_progress_bar=False)

# # Then, we use the util.semantic_search function, which computes the cosine-similarity
# # between the query embedding and all image embeddings.
# # It then returns the top_k highest ranked images, which we output (ANN using hsnw)
# #Using KNN for the timebeing
# hits = util.semantic_search(query_emb, embeddings, top_k=3)[0]

# COMMAND ----------

# hits

# COMMAND ----------

# product_dict

# COMMAND ----------

# matches = []
# details = []
# bs64_images = []

# for i in range(3):
#     matches.append(paths[(hits[i]["corpus_id"])])
#     # #img_bs64_list
#     bs64_images.append(img_bs64_list[(hits[i]["corpus_id"])])
    
#     # #This finds the corresponding recipes for the images
#     details.append(product_dict[int(matches[i].split('.')[0])])

# results = {'images': matches, 'bs64_images' : bs64_images , 'info':details}
# # results


# COMMAND ----------

import mlflow.pyfunc

class recipeKNNModelWrapper(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    from sentence_transformers import SentenceTransformer, util
    from PIL import Image
    import glob
    import torch
    import pickle
    from PIL import Image
    import base64
    from io import BytesIO
    import json
    import pandas as pd

    self.model = SentenceTransformer(context.artifacts["clip_vit_model_path"])
    #Create a dict with image paths as keys that would allow the similar items to be looked up in O(1)
    self.product_dict = pd.read_csv(context.artifacts["product_path"])[['ProductId','Information','Image','Week','units_sold','units_onhand','status','OOS']].set_index('ProductId').T.to_dict()
    self.paths = [path.split('/')[-1] for path in pickle.load(open(context.artifacts["pickle_location"], "rb"))['image_paths']]
    self.embeddings = pickle.load(open(context.artifacts["pickle_location"], "rb"))['embeddings']
    self.img_bs64_list = pickle.load(open(context.artifacts["pickle_location"], "rb"))['bs64_images']
    
  def predict(self, context, model_input):
    from PIL import Image
    from sentence_transformers import SentenceTransformer, util
    import base64
    from io import BytesIO
    import json
    import pandas as pd
    # convert search element into an embedding
    question = model_input.iloc[:,0].to_list()[0] # get the first column
    if len(question) > 512 : # if search string is likely an encoded image
      image_string = base64.b64decode(question)
      image_string = Image.open(BytesIO(image_string)).convert('RGB')
    else:
      image_string = question

    query_emb = self.model.encode([image_string], convert_to_tensor=True, show_progress_bar=False)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output (ANN using hsnw)
    #Using KNN for the timebeing
    hits = util.semantic_search(query_emb, self.embeddings, top_k=3)[0]

    matches = []
    details = []
    bs64_images = []

    for i in range(3):
      matches.append(self.paths[(hits[i]["corpus_id"])])
      #img_bs64_list
      bs64_images.append(self.img_bs64_list[(hits[i]["corpus_id"])])
      
      # #This finds the corresponding recipes for the images
    details.append(self.product_dict[int(matches[i].split('.')[0])])

    results = {'images': matches, 'bs64_images' : bs64_images , 'info':details}
    return json.dumps(results)

# COMMAND ----------

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
# results

# COMMAND ----------

example_np = np.array(model_input)

# COMMAND ----------

import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'sentence_transformers',
          'pandas',
          'Pillow',
          'cloudpickle=={}'.format(cloudpickle.__version__),
          'torch'],
      },
    ],
    'name': 'st_env'
}

mlflow_pyfunc_model_path = "VIT_CLIP_MULTIMODAL_KNN_Embeddings_3"

# COMMAND ----------

mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=recipeKNNModelWrapper(),artifacts=artifacts,
        conda_env=conda_env, input_example = example_np)

# COMMAND ----------


