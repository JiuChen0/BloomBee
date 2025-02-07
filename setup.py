from setuptools import setup, find_packages  

setup(  
    name="bloombee",  # Change the name to "bloombee"  
    version="0.1.1",  
    author="Your Name",  
    author_email="your.email@example.com",  
    description="A short description of your project",  
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",  
    url="https://github.com/yottalabsai/BloomBee",  
    package_dir={"": "src"},  # Specify the source directory  
    packages=find_packages(where="src"),  # Find packages in the "src" directory  
    install_requires=[  
        "torch>=1.12",  
        "bitsandbytes==0.41.1",  
        "accelerate>=0.27.2",  
        "huggingface-hub>=0.11.1,<1.0.0",  
        "tokenizers>=0.13.3",  
        "transformers==4.43.1",  # if you change this, please also change version assert in bloombee/__init__.py  
        "speedtest-cli==2.1.3",  
        "hivemind",  
        "tensor_parallel==1.0.23",  
        "humanfriendly",  
        "async-timeout>=4.0.2",  
        "cpufeature>=0.2.0; platform_machine == 'x86_64'",  
        "packaging>=20.9",  
        "sentencepiece>=0.1.99",  
        "peft==0.8.2",  
        "safetensors>=0.3.1",  
        "Dijkstar>=2.6.0",  
        "numpy<2",  
    ],  
    classifiers=[  
        "Programming Language :: Python :: 3",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",  
    ],  
    python_requires=">=3.9",  
    extras_require={  
        "dev": [  
            "pytest==6.2.5",  
            "pytest-forked",  
            "pytest-asyncio==0.16.0",  
            "black==22.3.0",  
            "isort==5.10.1",  
            "psutil",  
        ],  
    },  
)