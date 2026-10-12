"""Print what the run environment actually provides. Run after setup."""
import os
import platform
import sys

ok = True


def line(name, value, good=True):
    global ok
    ok &= good
    print(f"{'OK ' if good else 'BAD'}  {name:<22} {value}")


line("python", platform.python_version(), sys.version_info[:2] == (3, 8))

try:
    import torch

    line("torch", torch.__version__, torch.__version__.startswith("1.10"))
    cuda = torch.cuda.is_available()
    print(f"     {'cuda available':<22} {cuda}")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"     {'gpu':<22} {p.name}, {p.total_memory / 2**30:.1f} GB, sm_{p.major}{p.minor}")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        line("gpu matmul", "ran", y == y)  # y == y is False only for NaN
except Exception as e:  # noqa: BLE001
    line("torch", repr(e), False)

for mod, want in [("numpy", "1.22.0"), ("pandas", "1.4.2"), ("scipy", "1.7.3"), ("sklearn", "1.0.2")]:
    try:
        v = __import__(mod).__version__
        line(mod, v, v == want)
    except Exception as e:  # noqa: BLE001
        line(mod, repr(e), False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from data import cpu_count  # noqa: E402

print(f"     {'cpus (quota)':<22} {cpu_count()}  (host reports {os.cpu_count()})")

print("\nenvironment OK" if ok else "\nsome checks failed (see BAD lines)")
sys.exit(0 if ok else 1)
