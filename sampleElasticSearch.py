from elasticsearch import Elasticsearch
# Found in the 'Manage this deployment' page
CLOUD_ID = "4cfc2437cfd94fd48672f76d09105426"
# Found in the 'Management' page under the section 'Security'
API_KEY = "essu_Y20xNVdrVktWVUpDTTNOVFlURm5jVXRXU3pFNmVHOW9iM0p2VUROVVkzazRiVEpvVUZCek1rbFJVUT09AAAAAOb3Gis="
# Create the client instance
client = Elasticsearch(
    cloud_id=CLOUD_ID,
    api_key=API_KEY,
)
          
