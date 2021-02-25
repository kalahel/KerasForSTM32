# Keras pour STM32
Le but de ce projet est de produire un modèle capable de réaliser une classification sur la base MNIST et de transférer ce modèle sur une carte STM32 nucleo F446Re pour y réaliser l'inférence.

## Entraînement du modèle

Le script `produceModel.py` permet d'entrainer un petit réseau de convolution sur la base MNIST et d'exporter ses poids sous la forme d'un fichier `.h`.

Le modèle prends en entrée les images en format 28*28, la valeur des pixels à été divisée par 255 pour obtenir des valeurs comprises entre [0,1].

La sortie est un vecteur de 10 nombres à virgule flottante représentant les valeurs [0,...,9] en encodage one-hot.

## Test d'un modèle

Le script `checkModel.py` permet d'évaluer les performances d'un modèle entrainé dans les conditions décrites précédemment.

## Utilisation du modèle déployer sur stm32

Le script `SerialSenderReceiver.py` est à utilisé conjointement avec une carte STM32 connectée avec le programme `STM/mnistSmallerUsartV1/main.c` chargé dessus.

Attention il est nécessaire d'avoir nommer son modèle `mnistsmall` lors de son chargement sous Cube MX.