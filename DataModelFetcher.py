import requests
import tempfile
import numpy as np

class DataModelFetcher:
    def fetch_model(self, url):
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            return tmp.name

    def fetch_data(self, url):
        response = requests.get(url)
        return np.load(io.BytesIO(response.content))