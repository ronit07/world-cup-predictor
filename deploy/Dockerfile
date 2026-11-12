# Lambda container image. xgboost-cpu is the CPU-only build, the normal
# xgboost wheel pulls in NVIDIA libraries that blow past Lambda's size limit.
FROM public.ecr.aws/lambda/python:3.12

COPY deploy/requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY src/ ${LAMBDA_TASK_ROOT}/

CMD ["api.handler"]
