# Tensor
A place where I  practice codes on ML and Python 


TensorFlow 2.0:
  Fashion MNIST:
  
  
      import tensorflow as tf
      import numpy as np
      import math
      import matplotlib.pyplot as plt
      import tensorflow_datasets as tfds
      import logging
      logger = tf.get_logger()
      logger.setLevel(logging.ERROR)

    dataset, metadata=tfds.load("fashion_MNIST", as_supervised=True, with_info=True)
    train , test=dataset['train'], dataset['test']

    x=metadata.splits['train'].num_examples
    y=metadata.splits['test'].num_examples
    print(y)
    print(x)  

    def normalise(images,labels):
      images=tf.cast(images,tf.float32)
      images/=255
      return images, labels

    train=train.map(normalise)
    test=test.map(normalise)
    train=train.cache()
    test=test.cache()

     for image, label in test.take(2):
      break
    image= image.numpy().reshape((28,28))
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.show()
  
    *model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), padding='same', activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64,(3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)                           
    ]
    )*

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    batch=32
    train=train.cache().repeat().shuffle(x).batch(batch)
    test=test.cache().batch(batch)

    model.fit(train, epochs=10, steps_per_epoch=math.ceil(x/32))

    for test_images, test_labels in test.take(1):
      test_images=test_images.numpy()
      test_labels=test_labels.numpy()
      predictions=model.predict(test_images)

  
Try 2: Horses or humans:

            import tensorflow as tf
            import tensorflow_datasets as tfds
            import math
            import numpy as np
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            import matplotlib.pyplot as plt

    ds,ds_in=tfds.load('horses_or_humans',as_supervised=True,with_info=True)
    test,train=ds['test'],ds['train']

    c=len(test)
    b=len(train)
    print(c)
    print(b)

    for image,labels in train.take(1):
    break
    image=image.numpy().reshape((300,300,3))
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    def rescale(images,label):
    images=tf.cast(images,tf.float32)
    images/=255
    return images,label
    
    test=test.map(rescale)
    train=train.map(rescale)
    test=test.cache()
    train=train.cache()

    model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')])    

    model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
    
    train=train.cache().repeat().shuffle(1027).batch(30)
    model.fit(train,epochs=3,steps_per_epoch=math.ceil(1027/30))
    test=test.cache().batch(30)
    
    for image in test.take(1):
    prediction=model.predict(image)
    predictions.shape
  
Try 3 :MNIST
    
    import tensorflow as tf
    import numpy as np
    import tensorflow_datasets as tfds
    import matplotlib.pyplot as plt
    import math
    dataset,metadata=tfds.load('mnist',as_supervised=True,with_info=True)
    test,train=dataset['test'],dataset['train']
    cnames=metadata.features['label'].names
    print(cnames)
    def normalize(images,labels):
    tf.cast(images,tf.float32)
    images/=255
    return images,labels
    train=train.map(normalize)
    test=test.map(normalize)
    train=train.cache()
    test=test.cache()
    for image,label in test.take(1):
    break
    image=image.numpy().reshape((28,28))
    plt.figure()
    plt.imshow(image,cmap=plt.cm.binary)
    model=tf.keras.Sequential([tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)),
                            tf.keras.layers.MaxPool2D((2,2),strides=2),
                            tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
                            tf.keras.layers.MaxPool2D((2,2),strides=2),
                            tf.keras.layers.Dropout(0.5),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(512,activation='relu'),
                            tf.keras.layers.Dense(128,activation='relu'),
                            tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    num_test=metadata.splits['test'].num_examples
    num_train=metadata.splits['train'].num_examples
    batch=1500
    train=train.cache().repeat().shuffle(num_train).batch(batch)
    test=test.cache().batch(batch)
    model.fit(train,epochs=10,steps_per_epoch=math.ceil(num_train/batch))
