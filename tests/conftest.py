import os

# The suite checks numerics against CPU references; set JAX_PLATFORMS to
# override before running.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
