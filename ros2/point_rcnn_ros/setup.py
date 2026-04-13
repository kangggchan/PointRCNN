from setuptools import setup


package_name = 'point_rcnn_ros'


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/point_rcnn.launch.py']),
        ('share/' + package_name + '/config', ['config/point_rcnn_node.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PointRCNN User',
    maintainer_email='khang.buiphuoc@gmail.com',
    description='ROS2 PointRCNN node for realtime PointCloud2 inference and RViz box visualization.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'point_rcnn_node = point_rcnn_ros.point_rcnn_node:main',
        ],
    },
)