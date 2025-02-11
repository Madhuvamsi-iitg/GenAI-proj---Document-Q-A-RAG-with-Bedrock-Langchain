import boto3
import json

prompt_data = """
Act as a shakespeare and write a poem on Machine learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt":"[INST]" + prompt_data + "[/INST]",
    "max_tokens":200,
    "temperature":0.8,
    "top_p":0.9,
    "top_k":50

}


body = json.dumps(payload)
model_id = "mistral.mistral-large-2402-v1:0"
response = bedrock.invoke_model(
    body=body,
    modelId = model_id,
    accept = "application/json",
    contentType ="application/json"
)

# Parse the response
response_body = json.loads(response.get("body").read())
response_text = response_body['outputs'][0]['text']  # Mistral's response structure

# Print the generated text
print(response_text)
