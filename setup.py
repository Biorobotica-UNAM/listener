from setuptools import setup
import os
from glob import glob

package_name = 'listener'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        (os.path.join('share', package_name, 'grammar'), glob('grammar/*.json')),

        *[
            (os.path.join('share', package_name, root), [os.path.join(root, f)])
            for root, _, files in os.walk('model')
            for f in files
        ],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fidel Martinez',
    maintainer_email='jorge.martinezram@ingenieria.unam.edu',
    description='Speech recognition node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listener = listener.listener:main',
        ],
    },
)

