import boto3

s3 = boto3.resource('s3')

obj = s3.Object("vect-db", "context/context_3.txt")
context = obj.get()['Body'].read().decode('utf-8') 
print(context)
