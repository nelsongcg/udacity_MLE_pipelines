import json
import boto3
import base64

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png

    s3.download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor
from sagemaker import Session

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2022-01-20-01-18-32-688'
session = Session()


def lambda_handler(event, context):
    # Decode the image data

    image_data = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        session=session
    )

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences =  predictor.predict(image_data)

    # We return the data back to the Step Function
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }



import json
import ast

THRESHOLD = .93


def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = json.loads(event["body"])["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(ast.literal_eval(inferences)) > THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise ValueError("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }