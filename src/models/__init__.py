from .baseline_mlp import BaselineMLP
from .lstm import LSTMClassifier
from .transformer import TransformerEncoderClassifier
from .cnn_lstm import CNNLSTMHybrid

MODEL_REGISTRY = {
    "baseline_mlp": BaselineMLP,
    "lstm": LSTMClassifier,
    "transformer": TransformerEncoderClassifier,
    "cnn_lstm": CNNLSTMHybrid,
}
