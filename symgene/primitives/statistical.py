from symgene.primitives.squash import Squash

def make_statistical(squash: Squash) -> dict:
    def mean2(a, b):       return squash((a + b) / 2)
    def mean3(a, b, c):    return squash((a + b + c) / 3)
    def mean4(a, b, c, d): return squash((a + b + c + d) / 4)
    def max2(a, b):        return squash(max(a, b))
    def max3(a, b, c):     return squash(max(a, b, c))
    def max4(a, b, c, d):  return squash(max(a, b, c, d))
    def min2(a, b):        return squash(min(a, b))
    def min3(a, b, c):     return squash(min(a, b, c))
    def min4(a, b, c, d):  return squash(min(a, b, c, d))
    def harmonic2(a, b):
        if abs(a) < 1e-9 or abs(b) < 1e-9: return 0.0
        return squash(2 * a * b / (a + b))
    def geometric2(a, b):
        import math
        return squash(math.copysign(math.sqrt(abs(a * b)), a * b))
    return {
        "mean2": mean2, "mean3": mean3, "mean4": mean4,
        "max2": max2, "max3": max3, "max4": max4,
        "min2": min2, "min3": min3, "min4": min4,
        "harmonic_mean2": harmonic2, "geometric_mean2": geometric2,
    }
