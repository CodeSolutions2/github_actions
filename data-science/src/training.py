# -------------------------------------
# ScriptRunConfig
# -------------------------------------

# Import libraries
import sys
import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# -------------------------------------
# Change variables here
# -------------------------------------
# laquelle_args = ['train', 'input', 'prep', 'reg', 'learnrate', 'n_estimator']
laquelle_args = ['reg', 'input']
# -------------------------------------
# Definer X, y, et location des données
X_COLUMNS = ['area', 'perimeter', 'compactness', 'kernel_length', 
             'kernel_width', 'asymmetry_coefficient', 'groove_length']
Y_COLUMNS =  ['species']

# -------------------------------------
# laquelle_metrics = ['acc', 'auc']
laquelle_metrics = ['acc']
# -------------------------------------
# Definier le location, le nom, et le tag d'enregisterment de model
# SAVE_NAME_FOLDER = 'outputs'
PKL_MODEL_NOM_DE_ENREGISTERMENT = 'modelfilename'
NOM_DE_MODEL = 'model'
NOM_DE_MODEL_TAG = 'Training context'
RESPONSE_TAG = 'ScriptRunConfig'
# -------------------------------------


# Differences between Python SDK and GitHub Actions

# Python SDK
# FILE_PATH = sys.argv[1]


# -------------------------------------
# Get parameters
parser = argparse.ArgumentParser()


parser.add_argument("--train_data", type=str, help="Path to train dataset")
parser.add_argument("--model_output", type=str, help="Path of output model")

for i in laquelle_args:
    if i == 'training':
        NOM_DE_DATA = f"--{i}-data"
        DATA_TYPE = str
        DATA_ARG_DEST = f'{i}_dataset'
        DATA_NAME_OF_REGISTERED_DATASET = f"{i} dataset"
        
    elif i == 'reg':
        # -------------------------------------
        NOM_DE_DATA = f'--{i}ularization'
        DATA_TYPE = float
        DATA_ARG_DEST = f'{i}_rate'
        DATA_DEFAULT = 0.01
        DATA_NAME_OF_REGISTERED_DATASET = f'{i}ularization rate'
        # -------------------------------------
        
    else:
        # 'input'
        # -------------------------------------
        NOM_DE_DATA = f'--{i}-data'
        DATA_TYPE = str
        DATA_ARG_DEST = f'{i}_dataset'
        DATA_NAME_OF_REGISTERED_DATASET = f'{i} dataset'
        # -------------------------------------
        # Make something that exports the Environmental variables back to Jupyter-notebook
    
    if i in ['reg']:
        parser.add_argument(NOM_DE_DATA, 
                        type=DATA_TYPE, 
                        dest=DATA_ARG_DEST,
                        default=DATA_DEFAULT,
                        help=DATA_NAME_OF_REGISTERED_DATASET)
    else:
        parser.add_argument(NOM_DE_DATA, 
                        type=DATA_TYPE, 
                        dest=DATA_ARG_DEST,
                        help=DATA_NAME_OF_REGISTERED_DATASET)

args = parser.parse_args()
# -------------------------------------


# Get the experiment run context
run = Run.get_context()
# secret_value = run.get_secret(name="secret-name")
# print("Got secret value {} , but don't write it out!".format(len(secret_value) * "*"))

# -------------------------------------
# Get the training dataset
# df = run.input_datasets['madeup_name'].to_pandas_dataframe()
df = pd.read_csv(Path(args.train_data))

# Separate features and labels
X, y = df[X_COLUMNS].values, df[Y_COLUMNS].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
# -------------------------------------


# -------------------------------------
# Train a logistic regression model
reg = args.reg_rate
run.log('Regularization Rate',  np.float(reg))
globals()[f"{NOM_DE_MODEL}"] = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
# -------------------------------------


# -------------------------------------
PROPERTIES = {}
for i in laquelle_metrics:
    if i == 'acc':
        # -------------------------------------
        print('Calculate accuracy')
        y_hat = model.predict(X_test)
        acc = np.average(y_hat == y_test)
        print('Accuracy:', acc)
        run.log('Accuracy', np.float(acc))
        # -------------------------------------
        PROPERTIES['Accuracy'] = np.float(acc)
        
    elif i == 'auc':
        # -------------------------------------
        print('Calculate AUC')
        y_scores = model.predict_proba(X_test)
        auc = roc_auc_score(y_test,y_scores[:,1])
        print('AUC: ' + str(auc))
        run.log('AUC', np.float(auc))
        # -------------------------------------
        PROPERTIES['AUC'] = np.float(auc)
# -------------------------------------


# os.makedirs(SAVE_NAME_FOLDER, exist_ok=True)
# model_file = os.path.join(SAVE_NAME_FOLDER, f'{PKL_MODEL_NOM_DE_ENREGISTERMENT}.pkl')
# OR
model_file = os.path.join(args.model_output, f'{PKL_MODEL_NOM_DE_ENREGISTERMENT}.pkl')

joblib.dump(value=model, filename=model_file)


# -------------------------------------
print('Register the model')
from azureml.core.model import Model
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = NOM_DE_MODEL,
               tags={NOM_DE_MODEL_TAG:RESPONSE_TAG},
               properties=PROPERTIES)
# -------------------------------------


run.complete()
