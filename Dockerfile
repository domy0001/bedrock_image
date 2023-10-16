FROM public.ecr.aws/lambda/python:3.11

# Copy function code

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY boto3-1.28.21-py3-none-any.whl ${LAMBDA_TASK_ROOT}
COPY botocore-1.31.21-py3-none-any.whl ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt 

COPY bedrock_call.py ${LAMBDA_TASK_ROOT}
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "bedrock_call.handler" ]