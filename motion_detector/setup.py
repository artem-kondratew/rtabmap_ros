from setuptools import find_packages, setup

package_name = 'motion_detector'
submodules = 'motion_detector/submodules'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artem-kondratew',
    maintainer_email='artemkondratev5@gmail.com',
    description='Semantic segmentation for dynamic SLAM',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_detector = motion_detector.motion_detector:main',
            'fake_detector = motion_detector.fake_detector:main',
            'decoder = motion_detector.decoder:main',
            'visualizer = motion_detector.visualizer:main',
        ],
    },
)
