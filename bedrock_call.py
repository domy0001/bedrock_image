import os
import json
import boto3
import pickle
import streamlit as st
import numpy as np
import botocore
import langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from botocore.config import Config

print(f'Numpy version: {np.__version__}')
print(f'Langchain version: {langchain.__version__}')
print(f'Boto version: {boto3.__version__}')
print(f'Botocore version: {botocore.__version__}')

bedrock_region = "us-east-1"
bedrock_role = "arn:aws:iam::743456971407:role/BedrockCustomerPoc-CAAutoBank"
bedrock_endpoint = None
model_id = "amazon.titan-tg1-large"
boto3_kwargs = {'endpoint_url': bedrock_endpoint}
session_kwargs = {"region_name": bedrock_region}
client_kwargs = {**session_kwargs}
s3_client = boto3.client("s3")
output_type = os.environ.get('RESPONSE_TYPE')
session = boto3.Session(**session_kwargs)
config = Config(
    region_name=bedrock_region,
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

parameters = {
    "maxTokenCount": 100,
    "stopSequences": [],
    "temperature": 0,
    "topP": 0.9
}

accept = "application/json"
content_type = "application/json"


def get_client():
    if bedrock_role:
        print(f"  Using role: {bedrock_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(bedrock_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if bedrock_endpoint:
        client_kwargs["endpoint_url"] = bedrock_endpoint

    bedrock_client = session.client(
        service_name="bedrock",
        config=config,
        **client_kwargs
    )
    return bedrock_client


bedrock_client = get_client()

@st.cache_resource
def get_bedrock_embeddigns_model():
    return BedrockEmbeddings(model_id="amazon.titan-embed-g1-text-02", client=bedrock_client)

bedrock_embeddings = get_bedrock_embeddigns_model()

def get_index_pkl():
    response = s3_client.get_object(
        Bucket="vect-db",
        Key="vectdb/index.pkl")
    return response['Body'].read()

index_string_data = get_index_pkl()

def get_response(input_text, model_id, accept, content_type):
    vectorstore_faiss_pkl = pickle.loads(index_string_data)
    vectorstore_faiss = FAISS.deserialize_from_bytes(embeddings=bedrock_embeddings, serialized=vectorstore_faiss_pkl)

    query = input_text
    query_embedding = vectorstore_faiss.embedding_function(query)
    np.array(query_embedding)
    relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    context = os.environ.get('CONTEXT')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
        print('---')
        context = context + "\n" + "<doc>" + rel_doc.page_content + "</doc>"
    context += "</context>"
    prompt_template = f"\n\nHuman: {context}\n Question: {query}\n\nAssistant:"
    body = dict()
    body['prompt'] = prompt_template
    body['temperature'] = 0
    body['top_p'] = 0.1
    body['top_k'] = 4
    body['max_tokens_to_sample'] = 100000
    body['stop_sequences'] = ["\n\nHuman:"]
    response = bedrock_client.invoke_model(body=json.dumps(body), modelId="anthropic.claude-v2", accept=accept,
                                           contentType=content_type)
    response_body = json.loads(response.get("body").read())
    answer = response_body.get("completion")
    print(f'\n Answer: {answer}')
    return answer


def handler(event, context):
    print(json.dumps(event))
    print("input transcription:" + event['inputTranscript'])
    if len(event['inputTranscript']) == 0:
        content = "Mi dispiace, ma non ho compreso la sua domanda. Potrebbe essere più chiaro?"
    else:
        content = get_response(
            input_text=event['inputTranscript'],
            model_id=model_id,
            accept=accept,
            content_type=content_type)
    response = {
        'sessionState': {
            'sessionAttributes': None,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': {
                'name': 'FallbackIntent',
                'state': 'Fulfilled'
            }
        },
        'messages': [
            {
                'contentType': output_type,
                'content': content
            }
        ]
    }
    print(f'Response: {response}')
    return response



