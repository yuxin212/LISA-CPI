from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='LISA-CPI',
        version='1.0',
        packages=find_packages(),
        description='LISA-CPI: Ligand Image and receptor Structure-Aware deep learning framework for Compound-Protein Interaction prediction',
        author='Yuxin Yang, Yunguang Qiu, Jianying Hu, Michal Rosen-Zvi, Qiang Guan, and Feixiong Cheng',
        python_requires='>=3.7,<3.11',
    )
    setup(name='GPCR', version='1.0', packages=find_packages())