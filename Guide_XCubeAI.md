# X-CUBE-AI

## Introduction

Nous allons travailler sur la base de données MNIST qui récence 60 000 images de 28*28 pixels de chiffre manuscrit compris entre 0 et 9.

Le but est de produire à l'aide de la libraire Keras un réseau de neurones capable de reconnaitre  ces chiffres avec une grande précision.

Le réseau prendra en entrée une image 28*28 et aura comme sortie la classification utilisant un encodage one-hot, un vecteur de 10 valeurs représentant la probabilité pour chaque "label".

L'objectif est de produire un modèle assez léger pour pouvoir être intégré sur une carte STM32 tout en maximisant la précision du modèle.

Voici les différentes étapes suivies pour obtenir un projet fonctionnel.

## Python

J'utilise l'ide **Pycharm** pour toute la partie python.

Créez un nouveau projet :

`new project -> Pure Python-> Create`, laissez les paramètres par défauts (Virtualenv).

Créez un nouveau script python.

Voici le code que j'ai utilisé pour importer les données, les visualiser, entrainer un modèle et le sauvegarder.

```Python
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils

# Loading mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Plot dataset first 9 entries
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: " + str(y_train[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data from 255 to [0,1] to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

inputs = Input(shape=(28, 28, 1))

x = Conv2D(8, (5, 5), input_shape=(28, 28, 1), use_bias=False, padding="same", kernel_regularizer=l2(0.0001))(inputs)
x = Activation("relu")(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Conv2D(4, (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(0.0001))(x)
x = Activation("relu")(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, "relu")(x)
x = Dense(10, "sigmoid")(x)
model = Model(inputs, x, name="mnistSmall")
# compiling the sequential model
model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
                    batch_size=128, epochs=30,
                    validation_data=(X_test, Y_test))

model.save('./mnistSmall.h5')
```

Importez les différentes librairies :

`Alt + Entrée -> install package` sur les différentes librairies manquantes.

Lancez le script, cela devrait prendre quelques minutes en fonction de la puissance de votre CPU ou GPU (si cuda est installé).

### Résultats d'entrainement

Les résultats d'entrainement peuvent varier, mais ils devraient être proches de ces valeurs.

| Précision entraînement | Pertes entraînement | Précision Validation | Pertes Précision |
| ---------------------- | ------------------- | -------------------- | ---------------- |
| 99.45%                 | 0.0019              | 99.12%               | 0.0022           |

Le modèle a dû normalement être sauvegardé dans le même dossier que le script sous le nom `mnistSmall.h5`

## Visualisation du réseau (optionnel)

Vous pouvez utiliser [Netron](https://github.com/lutzroeder/netron) pour visualiser le réseau produit en se basant sur le `.h5`.

![mnistsmall](https://i.ibb.co/q7FDYHK/mnist-Small.png)

## Explications du réseau

Le réseau produit doit répondre à deux critères, d'abord de dimension et ensuite de précision. Le modèle doit pouvoir être utilisé sur une carte nucleo F446Re qui n'a que 512 Ko de mémoire flash et 128 Ko de RAM.

Pour ne pas dépasser ses capacités il faut donc veiller à réduire le nombre de poids, mais aussi la taille totale qui doit être allouée pour les différentes opérations.

Par exemple, une couche de convolution avec 100 filtres 5x5, ne représente que 5x5x100, soit 2500 poids, étant chacun stocké en nombre flottant cela représente 2500x4 = 10 ko. En revanche, la mémoire nécessaire pour stocker le résultat de ces convolutions représenterai dans notre cas 28x28x100x4 = 313 600 octets, ce qui est plus de deux fois supérieur à la mémoire RAM disponible.

L'implémentation exacte de la librairie Cube AI n'est pas accessible à l'utilisateur, mais on peut spéculer sur le fait qu'elle utilise certaines astuces pour réduire le nombre de valeurs intermédiaires à stocker lors l'inférence. Elle applique certainement la fonction d'activation puis le MaxPooling (s'il y en a un) filtre après filtre, plutôt que de réaliser toutes les convolutions avant d'appliquer l'activation et le pooling.

L'architecture présentée, utilise des couches de convolution, ce qui permet de détecter des paternes. La fonction d'activation (Relu) à la sortie de chaque convolution permet d'éliminer les valeurs négatives et d'introduire de la non-linéarité lors de l'entraînement. Les données sont essentialisées grâce au Max Pooling  en ne gardant que les valeurs maximales locales. Les deux couches Dense (appelé aussi fully connected) réalisent la classification en elle-même.

Les réseau de convolution sont des architectures qui ont faites leurs preuves dans le domaine de la reconnaissance d'image, celle-ci en est un petit modèle ne nécessitant que 42.74 Ko de mémoire Flash et 11.01 Ko de RAM pour fonctionner sur la carte STM32 (valeurs données par Cube MX).



## Cube MX

Lancez CubeMX.

Installez X Cube AI :

Cliquez sur `INSTALL/REMOVE` dans `Install or remove embedded sofware packages` sur la page de garde. Onglet `STMicroelectronics`, cocher dans `X-CUBE-AI` ,`Artificial Inteligence` puis `install now`.

Fermez la fenêtre puis faites, `File->New project`, dans les filtres cochez `Artificial Intelligence->Enable`.

Sélectionnez `Keras` comme `Model`,  `Saved model` comme `Type`.

Sélectionnez votre model puis faite `Analyze` (vous pouvez aussi compresser votre model, mais pour celui-ci, cela n'a pas été nécessaire).

On utilise ici la carte`STM32F446RE`, sélectionnez la et faites `Start Project`.



Allez dans `Software Packs->STMicroelectronics X-CUBE-AI`, dans `X-CUBE-AI` cochez `Core` et dans `Application` sélectionnez `ApplicationTemplate`.

Désormais un onglet `Software Packs` est apparu à gauche de la fenêtre de projet, allez dedans. Dans `Configuration` sélectionnez votre réseau, changez le modèle input en `mnistsmall` (pour avoir les mêmes signatures de fonctions que celles utilisées dans la section suivante). Vous pouvez désormais compresser, analyser et valider le réseau.

Vous pouvez paramétrer les différents ports (UART,...) ainsi qu'augmentez la fréquence d'horloge.

Une fois toutes ces opérations terminées, générez le code en augmentant `Minimum Heap Size` pour une valeur de `0x2000`.

## Cube IDE

Créez le projet à partir du code généré par CubeMX. Il est nécessaire avant tout de le **build**, ceci permettra de générer les différent fichiers nécessaires à la suite du développement.

Les fichiers liés à notre réseau de neurones sont :

```
YourProjectName
+-- X-CUBE-AI
|	+-- YourNetwork.c
|	+-- YourNetwork.h
|	+-- YourNetwork_data.c
```

**YourNetwork_data.c** contient les poids de votre réseau.

**YourNetwork.h** contient la signature des fonctions à appeler pour instancier et utiliser votre réseau dans votre programme.

**YourNetwork.c** contient l'implémentation des fonctions liées au réseau.

#### Code minimal nécessaire

Dans `main.c` ajoutez les includes suivant :

```c
#include "stdio.h"
#include "mnistsmall.h"
```

**Attention**, ici c'est `mnistsmall.h` mais cela dépend du nom donné à votre réseau lors de la création du projet sous CubeMX.

Ce code est pour un réseau appelé `mnistsmall`. Il est à intégré dans le `main` entre les commentaires   `/* USER CODE BEGIN 2 */`.

```c
float data[AI_MNISTSMALL_IN_1_SIZE] = {...} // Insert here values from a 28*28 image 
ai_handle mnist_model = AI_HANDLE_NULL;

// Input/output buffer initialization
ai_buffer inputs[AI_MNISTSMALL_IN_NUM] = AI_MNISTSMALL_IN;
ai_buffer ouputs[AI_MNISTSMALL_OUT_NUM] = AI_MNISTSMALL_OUT;

// Chunk of memory used to hold intermediate values for neural network
AI_ALIGNED(4) ai_u8 activations_buffer[AI_MNISTSMALL_DATA_ACTIVATIONS_SIZE];

// Set working memory and get weights from model
ai_network_params params_model = {
AI_MNISTSMALL_DATA_WEIGHTS(ai_mnistsmall_data_weights_get()),
AI_MNISTSMALL_DATA_ACTIVATIONS(activations_buffer) };

// Buffers used to store input and output tensors
AI_ALIGNED(4) ai_i8 in_data[AI_MNISTSMALL_IN_1_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 out_data[AI_MNISTSMALL_OUT_1_SIZE_BYTES];

// Set pointers wrapper structs to our data buffers
inputs[0].n_batches = 1;
inputs[0].data = AI_HANDLE_PTR(in_data);
ouputs[0].n_batches = 1;
ouputs[0].data = AI_HANDLE_PTR(out_data);

// -- MODEL DEFINITION AND INITIALIZATION --

// Create instance of neural network
ai_error ai_err = ai_mnistsmall_create(&mnist_model,
AI_MNISTSMALL_DATA_CONFIG);
if (ai_err.type != AI_ERROR_NONE) {
	// todo Handle error
	printf("Error on model creation\r\n");
}

// Initialize neural network
if (!ai_mnistsmall_init(mnist_model, &params_model)) {
	// todo handle error
	printf("Error on model initialization\r\n");
}
// Filling input buffer
for (int i = 0; i < AI_MNISTSMALL_IN_1_SIZE; ++i) {
	((float*) in_data)[i] = data[i];
}
ai_mnistsmall_run(mnist_model, inputs, ouputs);
float simple_outputs[AI_MNISTSMALL_OUT_1_SIZE] = { 0 };
for (int i = 0; i < AI_MNISTSMALL_OUT_1_SIZE; ++i) {
	simple_outputs[i] = ((float*) ouputs->data)[i];
}
```

Remplacez les données lignes 1 par celles en annexes.

Vous pouvez utiliser le debugger pour vérifier les données contenues dans `simple_ouputs`. Voici les résultats  :

```c
simple_outputs[0]	float	8.6129802e-008	
simple_outputs[1]	float	0.00114235596	
simple_outputs[2]	float	0.000401871046	
simple_outputs[3]	float	3.86849308e-008	
simple_outputs[4]	float	0.998965263	
simple_outputs[5]	float	4.87507623e-006	
simple_outputs[6]	float	3.02201158e-007	
simple_outputs[7]	float	1.31850845e-006	
simple_outputs[8]	float	2.95835844e-006	
simple_outputs[9]	float	1.60644595e-005	
```

Nous utilisons un encodage one-hot pour la sortie, on peux donc voir que c'est bien un 4 qui à été reconnu.

## Utilisation de la liaison série

Il est nécessaire de valider le modèle sur la carte, mais sa mémoire étant limité il est impossible de stocker un jeu de test important.

Une alternative simple est de procéder à l'envoie des images par la liaison série puis de récolter les résultats. Ce processus bien que lent permet de s'affranchir des contraintes mémoire de la carte.

Voici la succession d'échanges entre les deux programmes :

![serialComExchange](https://i.ibb.co/GsSMXKG/Stm32-Serial.png)

Le `checksum` est calculé en faisant simplement la somme de tous les pixels de l'image. Cela permet de vérifier qu'il n'y a pas eu de dégradation des données lors de l'envoi.

Le script python est `SerialModelTester.py`.

Le code  C pour la carte STM32 est `STM32/mnistSmallerUsartFullTestV0/main.c`.

### Résultats

Le programme python va ensuite récolter les résultats de classification envoyés par la carte puis les comparer aux classifications réelles pour déterminer les performances du modèle.

Voici les résultats sur la carte pour 1000 images : 

| Précision de classification | Erreur moyenne par image |
| --------------------------- | ------------------------ |
| 0.9968                      | 0.058855                 |

On constate que la précision du modèle reste supérieur à 99%, il ne semble donc pas y avoir eu de perte lié à son import sur la carte.

l'erreur moyenne par image est calculée en faisant la somme des différences absolues entre les valeurs attendues pour une images et les résultats de classification.

Le programme complet à pris 302 secondes pour être effectué soit 0.302 secondes par image. La moitié du temps de traitement est dû à l'inférence du modèle sur la carte.

## Sources

[Tutorial officiel](https://www.google.com/search?client=firefox-b-d&q=getting+started+xcube+ai#kpvalbx=_P3siYLzbFtCEadiouIAH12)

[Tutorial complet video](https://www.youtube.com/watch?v=crJcDqIUbP4&feature=emb_title) & [blog](https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0)

## Annexes

### Données utilisées sous Cube IDE

Voici les valeurs des pixels d'une image 28*28 représentant un 4.

```c
float data[AI_MNISTSMALL_IN_1_SIZE] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.2627450980392157, 0.9098039215686274, 0.15294117647058825, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24313725490196078,
			0.3176470588235294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.47058823529411764, 0.7058823529411765,
			0.15294117647058825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.49411764705882355, 0.6392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00784313725490196, 0.6,
			0.8235294117647058, 0.1568627450980392, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.8627450980392157, 0.6392156862745098, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.10588235294117647, 0.996078431372549, 0.6352941176470588, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8705882352941177,
			0.6392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.7176470588235294, 0.996078431372549,
			0.49019607843137253, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.1803921568627451, 0.9607843137254902, 0.6392156862745098, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.7764705882352941, 0.996078431372549, 0.2196078431372549, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47058823529411764,
			0.996078431372549, 0.6392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09019607843137255,
			0.9058823529411765, 0.996078431372549, 0.11372549019607843, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6235294117647059,
			0.996078431372549, 0.47058823529411764, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6392156862745098,
			0.996078431372549, 0.8470588235294118, 0.06274509803921569, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6235294117647059,
			0.996078431372549, 0.2627450980392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.054901960784313725, 0.33725490196078434,
			0.6980392156862745, 0.9725490196078431, 0.996078431372549,
			0.3568627450980392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.6235294117647059, 0.996078431372549, 0.3333333333333333, 0.0,
			0.0, 0.0, 0.1843137254901961, 0.19215686274509805,
			0.4549019607843137, 0.5647058823529412, 0.5882352941176471,
			0.9450980392156862, 0.9529411764705882, 0.9176470588235294,
			0.7019607843137254, 0.9450980392156862, 0.9882352941176471,
			0.1568627450980392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.5882352941176471, 0.9921568627450981, 0.9294117647058824,
			0.8117647058823529, 0.8117647058823529, 0.8117647058823529,
			0.9921568627450981, 0.996078431372549, 0.9803921568627451,
			0.9411764705882353, 0.7764705882352941, 0.5607843137254902,
			0.3568627450980392, 0.10980392156862745, 0.0196078431372549,
			0.9137254901960784, 0.9803921568627451, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4666666666666667,
			0.6941176470588235, 0.6941176470588235, 0.6941176470588235,
			0.6941176470588235, 0.6941176470588235, 0.3843137254901961,
			0.2196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.996078431372549,
			0.8627450980392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.6627450980392157, 0.996078431372549,
			0.5372549019607843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.6627450980392157, 0.996078431372549,
			0.2235294117647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.6627450980392157, 0.996078431372549,
			0.2235294117647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.6627450980392157, 1.0, 0.3686274509803922, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.6627450980392157, 0.996078431372549, 0.3764705882352941, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.6627450980392157, 0.996078431372549, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6627450980392157, 1.0, 0.6,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.3764705882352941, 0.996078431372549, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
```

### Code python pour passer les données sous forme de tableau C

Voici le script que j'ai utilisé pour convertir les données sous forme de tableau C unidimensionnel.

```python
from keras.datasets import mnist

# Loading mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

data = X_train[2]

data_str_tmp = '{'
for i_row in range(len(data)):
    for i in range(len(data[i_row])):
        if i == len(data[i_row]) - 1 and i_row == len(data) - 1:
            data_str_tmp = data_str_tmp + str(data[i_row][i] / 255)
        else:
            data_str_tmp = data_str_tmp + str(data[i_row][i] / 255) + ','
    if i_row == len(data) - 1:
        data_str_tmp = data_str_tmp + '}'

print(data_str_tmp)
```



