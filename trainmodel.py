from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
test_actions = actions[:5]  # Train with A-E first
print(f"Training with actions: {test_actions} ({len(test_actions)} letters)")
for action in test_actions:
    for sequence in range(no_sequences):
        window = []
        sequence_complete = True
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            if os.path.exists(npy_path):
                res = np.load(npy_path)
                window.append(res)
            else:
                print(f"Missing file: {npy_path}")
                sequence_complete = False
                break
        if sequence_complete and len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of sequences loaded: {len(sequences)}")

if len(sequences) == 0:
    print("No data loaded! Run data.py first to generate keypoint data.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Improved model architecture
from keras.layers import Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(test_actions), activation='softmax'))

# Better optimizer settings
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train with validation data
history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                   validation_data=(X_test, y_test), verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')