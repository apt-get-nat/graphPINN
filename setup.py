from setuptools import setup, find_packages

install_requires = ["torch", "torch-sparse", "torch-scatter", "torch-geometric"]

setup(
  name="graphPINN",
  version = "0.0.1",
  maintainer="Nat Mathews",
  license="GNU3",
  packages=find_packages(),
  install_requires=install_requires,
  python_requires=">=3.9",
)
