from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    HYPHEN_E_DOT = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[lines.replace("\n","") for lines in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name = "mlproject",
    version="0.0.1",
    author = "Ayush",
    author_email = "ayushsojitra1@gmail.com",
    package = find_packages(),
    install_requires = get_requirements("requirements.txt")
)