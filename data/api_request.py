import json
import boto3
from dotenv import load_dotenv
import os

#load env variables
load_dotenv()
DATA_SET_ID = os.environ.get("DATA_SET_ID")
REVISION_ID = os.environ.get("REVISION_ID")
ASSET_ID    = os.environ.get("ASSET_ID")
API_KEY     = os.environ.get("API_KEY")

# Instantiate DataExchange client
CLIENT = boto3.client('dataexchange', region_name='us-east-1')

# Query we saved earlier
query_file = open("titanicRatingsQuery.graphql", "r")
query = query_file.read()
query_file.close()
BODY = json.dumps({'query': query})

METHOD = 'POST'
PATH = '/v1'

response = CLIENT.send_api_asset(
    DataSetId=DATA_SET_ID,
    RevisionId=REVISION_ID,
    AssetId=ASSET_ID,
    Method=METHOD,
    Path=PATH,
    Body=BODY,
    RequestHeaders={
        'x-api-key': API_KEY
    },
)

# This will print the IMDb API GraphQL Response
print(f"Response Body: {response['Body']}")