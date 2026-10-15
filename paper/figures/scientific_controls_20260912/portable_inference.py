"""Run CNN and MLP8 inference using NumPy.

Inputs are finite arrays of shape (N, 24); outputs are (N, 2) positions in
micrometres. Model NPZ files contain numeric weights and scaling parameters
and are loaded with allow_pickle=False. The CLI writes predictions and JSON
metadata; use a ``.npy`` output name so both files share the same stem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SHAPES = {
    'cnn': {'conv1_weight': (64, 1, 3), 'conv1_bias': (64,),
            'conv2_weight': (128, 64, 3), 'conv2_bias': (128,),
            'conv3_weight': (256, 128, 3), 'conv3_bias': (256,),
            'dense1_weight': (128, 4608), 'dense1_bias': (128,),
            'dense2_weight': (64, 128), 'dense2_bias': (64,),
            'dense3_weight': (2, 64), 'dense3_bias': (2,)},
    'mlp8': {'dense1_weight': (8, 24), 'dense1_bias': (8,),
             'dense2_weight': (2, 8), 'dense2_bias': (2,),
             'feature_mean': (24,), 'feature_scale': (24,),
             'target_mean': (2,), 'target_scale': (2,)},
}
SCALERS = {'feature_mean', 'feature_scale', 'target_mean', 'target_scale'}


def conv1d_valid(x, weight, bias):
    """Apply Torch-compatible valid cross-correlation from ``(N,C,L)`` to ``(N,O,L-K+1)``."""
    n, channels, length = x.shape
    out_channels, weight_channels, kernel = weight.shape
    assert channels == weight_channels and x.dtype == weight.dtype == bias.dtype == np.float32
    windows = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=2)
    # Order each patch by input channel, then kernel position, exactly as weights.
    patches = windows.transpose(0, 2, 1, 3).reshape(n * (length - kernel + 1), channels * kernel)
    out = patches @ weight.reshape(out_channels, channels * kernel).T
    out += bias
    assert out.dtype == np.float32
    return out.reshape(n, length - kernel + 1, out_channels).transpose(0, 2, 1)


def dense(x, weight, bias):
    out = x @ weight.T
    out += bias
    assert out.dtype == np.float32
    return out


def relu(x):
    return np.maximum(x, np.float32(0), out=x)


class PortableEstimator:
    """NumPy predictor for a saved CNN or MLP8 model."""

    def __init__(self, model: str, weights_path):
        if model not in SHAPES:
            raise ValueError('Model must be cnn or mlp8.')
        self.model = model
        with np.load(weights_path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(SHAPES[model]):
                raise ValueError('Checkpoint keys do not match the unchanged topology.')
            self.arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
        for key, value in self.arrays.items():
            expected_dtype = np.float64 if key in SCALERS else np.float32
            if value.shape != SHAPES[model][key] or value.dtype != expected_dtype or not np.isfinite(value).all():
                raise ValueError('Invalid numeric state: ' + key)
        if model == 'mlp8' and (np.any(self.arrays['feature_scale'] <= 0) or np.any(self.arrays['target_scale'] <= 0)):
            raise ValueError('Standardisation scales must be positive.')

    def predict(self, features):
        """Map ``(N, 24)`` log features to ``(N, 2)`` positions in micrometres."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != 24 or not np.isfinite(features).all():
            raise ValueError('Expected a finite N x 24 log-feature matrix.')
        p = self.arrays
        if self.model == 'cnn':
            x = np.asarray(features, dtype=np.float32)[:, None, :]
            for i in (1, 2, 3):
                x = relu(conv1d_valid(x, p[f'conv{i}_weight'], p[f'conv{i}_bias']))
            x = x.reshape(len(features), 18 * 256)
            x = relu(dense(x, p['dense1_weight'], p['dense1_bias']))
            x = relu(dense(x, p['dense2_weight'], p['dense2_bias']))
            x = dense(x, p['dense3_weight'], p['dense3_bias'])
            return x.astype(np.float64)
        x = ((np.asarray(features, dtype=np.float64) - p['feature_mean']) / p['feature_scale']).astype(np.float32)
        x = relu(dense(x, p['dense1_weight'], p['dense1_bias']))
        x = dense(x, p['dense2_weight'], p['dense2_bias'])
        return x.astype(np.float64) * p['target_scale'] + p['target_mean']


def predict_in_batches(estimator, features, batch_size):
    """Predict all rows in fixed-size chunks while preserving row order."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    return np.concatenate([estimator.predict(features[start:start+batch_size])
                           for start in range(0, len(features), batch_size)], axis=0)


def primitive_self_test():
    """Check the cross-correlation primitive without importing Torch or scikit-learn."""
    x = np.arange(24, dtype=np.float32).reshape(2, 2, 6) / np.float32(8)
    weights = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / np.float32(16)
    bias = np.array([.125, -.5, .25], dtype=np.float32)
    expected = np.empty((2, 3, 4), dtype=np.float32)
    for n in range(2):
        for o in range(3):
            for t in range(4):
                expected[n, o, t] = np.sum(x[n, :, t:t+3] * weights[o], dtype=np.float32) + bias[o]
    assert np.array_equal(conv1d_valid(x, weights, bias), expected)
    assert not any(name == 'torch' or name.startswith('torch.') or name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules)


def main():
    """Write prediction arrays and metadata for one CLI inference request.

    The JSON records backend, model, row and batch counts, output dtype, NumPy
    version, import and primitive-test checks, and the prediction-byte SHA-256.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'mlp8'], required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=[1,256], required=True)
    args = parser.parse_args()
    primitive_self_test()
    estimator = PortableEstimator(args.model, args.weights)
    features = np.load(args.input, allow_pickle=False)
    output = predict_in_batches(estimator, features, args.batch_size)
    assert output.shape == (len(features), 2) and np.isfinite(output).all()
    forbidden = [name for name in sys.modules if name == 'torch' or name.startswith('torch.') or name == 'sklearn' or name.startswith('sklearn.')]
    assert not forbidden
    np.save(args.output, output, allow_pickle=False)
    result = {'backend':'numpy_only','model':args.model,'rows':len(features),
              'batch_size':args.batch_size,'last_batch_rows':((len(features)-1)%args.batch_size)+1,
              'output_dtype':str(output.dtype),'numpy_version':np.__version__,
              'forbidden_imports':forbidden,'primitive_cross_correlation_test':'exact_pass',
              'prediction_sha256':hashlib.sha256(output.tobytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
