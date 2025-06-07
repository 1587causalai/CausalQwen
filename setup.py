from setuptools import setup, find_packages

setup(
    name="causal-qwen",
    version="0.1.0",
    description="A simplified causal language model architecture",
    author="CausalQwen Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "transformers>=4.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ],
        "docs": [
            "docsify-cli>=4.4.3",
            "mkdocs>=1.2.3",
        ],
    },
)

