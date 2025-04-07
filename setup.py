from setuptools import setup, find_packages

setup(
    name="ManureTransport",
    version="0.1.0",
    author="wangyb",
    author_email="charleskeeling65@163.com",
    description="粪便养分运输优化和分配库",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CharlesKeeling65/ManureTransport",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "rasterio",  # 假设需要读写栅格数据
        "scipy",
    ],
)
