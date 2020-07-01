import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import tempfile
import pickle

# Load model module ... containing model creation and inference methods!
import model

# This will serve as an MLflow wrapper for the model
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["artifact_file"], 'r') as handle:
            args = handle.read()

        with open(context.artifacts["meta"], 'rb') as handle:
            self.meta = pickle.load(handle)

        print(f'Loading model ...')
        self.model = model.load_simple_model(args)

    # Create a predict function for our models
    def predict(self, context, model_input):
        return self.model.inference(model_input)


def pack(package_path):

    artifacts = {
        "artifact_file": "model_arguments.txt"
    }

    meta = {
        'author': 'Manuel Pasieka',
        'version': '1.0',
        'whatever': None
    }

    with tempfile.TemporaryDirectory() as tmp:
        # Meta data about this package
        meta_path = f'{tmp}/meta.pkl'
        with open(meta_path, 'wb') as handle:
            pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        artifacts.update({'meta': meta_path})

        #env = mlflow.pyfunc.get_default_conda_env() 
        mlflow.pyfunc.save_model(path=package_path,
                                python_model=ModelWrapper(),
                                artifacts=artifacts,
                                code_path=['model.py'])


def unpack(package_path):
    model = mlflow.pyfunc.load_model(package_path)

    meta = model._model_impl.python_model.meta
    print(f'Loaded model with meta: {meta}')

    prediction = model.predict('Running inference in unpacked model.')
    print(f'Prediction: {prediction}')


if __name__ == '__main__':
    print('Packaging the model ...')
    pack('test_package')

    print('Finished packaging the model!')

    print('Loading the model ...')
    unpack('test_package')