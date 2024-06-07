from setuptools import setup

setup(
    name="package",
    version="0.1",
    description="",
    author="Theodore",
    author_email="theodoreadiputra@gmail.com",
    packages=["package.feature", "package.ml_training","package.utils"],
    install_requires=["numpy", "pandas", "scikit-learn", "matplotlib", "mlflow"]
)