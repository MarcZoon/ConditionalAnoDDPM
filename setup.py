from setuptools import setup

setup(
    name="canoddpm",
    py_modules=["canoddpm"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
