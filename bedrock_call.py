import os
import json
import boto3
import pickle
import numpy as np
import time
import botocore
import langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from botocore.config import Config

level = "[INFO] - "

print(f'{level} Numpy version: {np.__version__}')
print(f'{level} Langchain version: {langchain.__version__}')
print(f'{level} Boto version: {boto3.__version__}')
print(f'{level} Botocore version: {botocore.__version__}')

bedrock_region = "us-east-1"
s3_client = boto3.client("s3")
output_type = os.environ.get('RESPONSE_TYPE')
s3 = boto3.resource('s3')
obj = s3.Object("vect-db", f"context/{os.environ.get('CONTEXT')}")
rules_context = obj.get()['Body'].read().decode('utf-8') 

parameters = {
    "maxTokenCount": 100,
    "stopSequences": [],
    "temperature": 0,
    "topP": 0.9
}

accept = "application/json"
content_type = "application/json"

bedrock_client = boto3.client('bedrock-runtime')

def get_bedrock_embeddigns_model():
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_index_pkl():
    response = s3_client.get_object(
        Bucket="vect-db",
        Key="vectdb/index.pkl")
    return response['Body'].read()

index_string_data = get_index_pkl()

def get_response(input_text, accept, content_type):
    doc_words_count = 0
    query = input_text
    start_embedding = time.time()
    query_embedding = vectorstore_faiss.embedding_function(query)
    np.array(query_embedding)
    end_embedding = time.time()
    print(f'{level} Embedding input text execution time: {end_embedding-start_embedding}')
    start_faiss = time.time()
    relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
    end_faiss = time.time()
    print(f'{level} Faiss similarity_search_by_vector execution time: {end_faiss-start_faiss}')
    context = rules_context
    for i, rel_doc in enumerate(relevant_documents):
        doc_words_count += len(rel_doc.page_content)
        context = context + "\n" + "<doc>" + rel_doc.page_content + "</doc>"
    context += "</context>"
    print(f'Documents token: {doc_words_count}')
    print(f'Context tokens: {len(context.split())}')
    prompt_template = f"\n\nHuman: {context}\n Question: {query}\n\nAssistant:"
    body = dict()
    body['prompt'] = prompt_template
    body['temperature'] = 0.6
    body['top_p'] = 0.999
    body['top_k'] = 250
    body['max_tokens_to_sample'] = 2000
    body['stop_sequences'] = ["\n\nHuman:"]

    start_bedrock = time.time()
    response = bedrock_client.invoke_model(body=json.dumps(body), modelId="anthropic.claude-v2", accept=accept,
                                           contentType=content_type)
    end_bedrock = time.time()
    print(f'{level} Bedrock invoke model execution time: {end_bedrock-start_bedrock}')                                        
    response_body = json.loads(response.get("body").read())
    answer = response_body.get("completion")
    print(f'\n {level} Answer: {answer}')
    return answer

bedrock_embeddings = get_bedrock_embeddigns_model()

vectorstore_faiss_pkl = pickle.loads(index_string_data)
vectorstore_faiss = FAISS.deserialize_from_bytes(embeddings=bedrock_embeddings, serialized=vectorstore_faiss_pkl)

def handler(event, context):
    print(f"{level} Input transcription:" + event['inputTranscript'])
    if len(event['inputTranscript']) == 0:
        content = "Mi dispiace, ma non ho compreso la sua domanda. Potrebbe essere più chiaro?"
    else:
        content = get_response(
            input_text=event['inputTranscript'],
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
    print(f'{level} Response: {response}')
    return response



