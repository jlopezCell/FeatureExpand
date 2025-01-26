from setuptools import setup, find_packages

setup(
    name='nombre_del_proyecto',  # Nombre del paquete
    version='0.1',               # Versión del paquete
    packages=find_packages(),    # Encuentra automáticamente los paquetes
    install_requires=[],         # Dependencias necesarias
    author='Tu Nombre',          # Tu nombre
    author_email='tu@email.com', # Tu email
    description='Descripción corta del proyecto',  # Descripción corta
    long_description=open('README.md').read(),  # Descripción larga (README)
    long_description_content_type='text/markdown',  # Tipo de contenido
    url='https://github.com/tu_usuario/nombre_del_proyecto',  # URL del proyecto
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Versión de Python requerida
)
