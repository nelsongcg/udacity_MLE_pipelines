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