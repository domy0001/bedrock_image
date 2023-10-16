#! /bin/bash
docker rmi bedrock:test
docker build --platform linux/amd64 -t bedrock:test .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 668487117877.dkr.ecr.us-east-1.amazonaws.com
docker tag bedrock:test 668487117877.dkr.ecr.us-east-1.amazonaws.com/bedrock_repo_2:latest
docker push 668487117877.dkr.ecr.us-east-1.amazonaws.com/bedrock_repo_2:latest