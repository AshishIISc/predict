import joblib
import numpy as np
from onnxmltools.convert import convert_xgboost
from onnxconverter_common import FloatTensorType

# Load trained model
xgb = joblib.load("model.joblib")

# Get feature count from preprocessor params
pp_params = np.load("preprocessor_params.npz", allow_pickle=True)
num_features = len(pp_params['num_mean']) + len(pp_params['cat_categories'][0])

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 412]))]
onnx_model = convert_xgboost(xgb, initial_types=initial_type)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
