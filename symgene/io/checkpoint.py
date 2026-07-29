import os

try:
    import dill as _pickle
except ImportError:
    import pickle as _pickle


def save_checkpoint(evolver, gen: int) -> str:
    os.makedirs(evolver.checkpoint_dir, exist_ok=True)
    path = os.path.join(evolver.checkpoint_dir, f"gen_{gen:04d}.sgk")
    with open(path, "wb") as f:
        _pickle.dump(evolver, f)
    return path


def load_checkpoint(path: str):
    try:
        import dill as pkl
    except ImportError:
        import pickle as pkl
    with open(path, "rb") as f:
        return pkl.load(f)
