import math
from symgene.primitives.squash import Squash

def make_activation(squash: Squash) -> dict:
    def sigmoid(x):
        v = squash(x)
        return squash(squash.lim * (1.0 / (1.0 + math.exp(-v))))
    def relu(x):      return squash(max(0.0, x))
    def softplus(x):  v = squash(x); return squash(math.log(1.0 + math.exp(min(v, 700.0))))
    def softsign(x):  return squash(x / (1.0 + abs(x)))
    def swish(x):
        v = squash(x)
        return squash(v / (1.0 + math.exp(-v)))
    def elu(x):       v = squash(x); return squash(v if v >= 0 else math.exp(min(v, 700.0)) - 1.0)
    def gaussian(x):  v = squash(x); return squash(squash.lim * math.exp(-v ** 2 / 2.0))
    def sinc(x):      v = squash(x); return squash(math.sin(v) / v if abs(v) > 1e-9 else 1.0)
    return {
        "sigmoid": sigmoid, "relu": relu, "softplus": softplus,
        "softsign": softsign, "swish": swish, "elu": elu,
        "gaussian": gaussian, "sinc": sinc,
    }
