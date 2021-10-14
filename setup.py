from distutils.core import setup
setup(
    name='efficientdet_aituring',         # How you named your package folder (MyLib)
    version='0.0.1',      # Start with a small number and increase it with every change you make
    description='AITuring EfficientDet Repositorio Refactorizado',   # Give a short description about your library
    author='Daniel Tobon Collazos',                   # Type in your name
    author_email='daniel.tobon@aituring.co',      # Type in your E-Mail
    url='https://github.com/danielTobon43',   # Provide either the link to your github or to your website
    packages=['efficientdet_aituring'],   # Chose the same as "name"
    install_requires=[            # I get to this in a second
        'absl-py==0.13.0',
        'configparser==5.0.2',
        'lxml==4.6.3',
        'Pillow==8.3.2',
        'pycocotools==2.0.2; sys_platform=="linux"',
        'pycocotools-windows==2.0.0.2; sys_platform=="win32"',
        'opencv-python==4.5.3.56',
        'PyYAML==5.4.1',
        'tensorboard==2.5.0',
        'tensorboard-data-server==0.6.0',
        'tensorboard-plugin-wit==1.8.0',
        'tensorflow>=2.5.0',
        'tensorflow-estimator==2.5.0',
        'tensorflow-model-optimization==0.5.1.dev0'
        'wget==3.2; sys_platform=="win32"',
        'tensorflow-addons'
    ]
)
