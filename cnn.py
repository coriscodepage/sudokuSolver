import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import Sequence
import keras.layers as kl
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

data = pd.read_csv("data_rust5.csv")
try:
    data = pd.DataFrame({"quizzes":data["puzzle"],"solutions":data["solution"]})
    #data = data.head(int(1e6 * 0.5))
except:
    pass
data.head()

data.info()

class DataGenerator(Sequence):
    def __init__(self, df,batch_size = 16,subset = "train",shuffle = False, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.info = info
        
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle==True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self,index):
        X = np.empty((self.batch_size, 9,9,1))
        y = np.empty((self.batch_size,81,1))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['quizzes'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = (np.array(list(map(int,list(f)))).reshape((9,9,1))/9)-0.5
        if self.subset == 'train': 
            for i,f in enumerate(self.df['solutions'].iloc[indexes]):
                self.info[index*self.batch_size+i]=f
                y[i,] = np.array(list(map(int,list(f)))).reshape((81,1)) - 1
        if self.subset == 'train': return X, y
        else: return X

LOAD_EXISTING_MODEL = True
MODEL_PATH = "best_weights.keras"

model = None

if LOAD_EXISTING_MODEL:
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"Loaded existing model from {MODEL_PATH}")
        print("Continuing training from saved weights...")
    except:
        print(f"Could not load model from {MODEL_PATH}, creating new model...")
        LOAD_EXISTING_MODEL = False

if not LOAD_EXISTING_MODEL:
    model = Sequential()
    
    model.add(kl.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(kl.BatchNormalization())
    
    model.add(kl.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(kl.BatchNormalization())
    
    model.add(kl.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(kl.BatchNormalization())
    
    model.add(kl.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(kl.BatchNormalization())
    
    model.add(kl.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))
    
    model.add(kl.Flatten())
    model.add(kl.Dense(81*9))
    model.add(kl.Reshape((-1, 9)))
    model.add(kl.Activation('softmax'))
    
    adam = keras.optimizers.Adam(learning_rate=0.002)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy']) # type: ignore

model.summary() # type: ignore

train_idx = int(len(data)*0.95)
data = data.sample(frac=1).reset_index(drop=True)
training_generator = DataGenerator(data.iloc[:train_idx], subset = "train", batch_size=256*2, shuffle=True)
validation_generator = DataGenerator(data.iloc[train_idx:], subset = "train",  batch_size=256*2)

filepath1="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.keras"
filepath2 = MODEL_PATH
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(
    factor=0.1,
    monitor='val_loss',
    patience=3,
    verbose=1,
    min_lr=1e-7
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

callbacks_list = [checkpoint1, checkpoint2, reduce_lr, early_stop]

history = model.fit( # type: ignore
    training_generator, 
    validation_data=validation_generator, 
    epochs=50,
    verbose="1",
    callbacks=callbacks_list
)