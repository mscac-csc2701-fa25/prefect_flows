from dotenv import load_dotenv
import os

load_dotenv()

BUCKET = 'smoke-fire-detection-bucket'
BATCH_SIZE = 10
INCOMING_PREFIX = 'datasets/incoming/raw/'
PROCESSED_PREFIX = 'datasets/incoming/processed/'
