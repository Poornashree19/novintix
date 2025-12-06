import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv('T1.csv', parse_dates=['Date/Time'], dayfirst=True)
df.set_index('Date/Time', inplace=True)
df.rename(columns={
    'LV ActivePower (kW)': 'LV_ActivePower',
    'Wind Speed (m/s)': 'WindSpeed',
    'Theoretical_Power_Curve (KWh)': 'TheoreticalPower',
    'Wind Direction (°)': 'WindDirection'
}, inplace=True)


df['LV_ActivePower'].fillna(method='ffill', inplace=True)


df['PerformanceScore'] = (df['LV_ActivePower'] / df['TheoreticalPower']) * 100


def categorize(score):
    if score >= 90:
        return "Good"
    elif score >= 75:
        return "Moderate"
    else:
        return "Poor"
df['Category'] = df['PerformanceScore'].apply(categorize)


le = LabelEncoder()
df['CategoryLabel'] = le.fit_transform(df['Category'])

def create_image_windows(series, labels, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        window = series[i:i+window_size]
        X.append(window.reshape((window_size, 1))) 
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

window_size = 5
X, y = create_image_windows(df['LV_ActivePower'].values, df['CategoryLabel'].values, window_size)


X = np.expand_dims(X, axis=-1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


num_classes = len(np.unique(y))

model = Sequential([
    Conv2D(16, (2,1), activation='relu', input_shape=(window_size, 1, 1)),
    MaxPooling2D((2,1)),
    Conv2D(32, (2,1), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))


y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()


for i in range(5):
    plt.imshow(X_test[i].squeeze(), cmap='hot', aspect='auto')
    plt.title(f"Actual: {le.classes_[y_test[i]]}, Predicted: {le.classes_[y_pred[i]]}")
    plt.colorbar()
    plt.show()

import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


heatmap = make_gradcam_heatmap(X_test[:1], model, last_conv_layer_name='conv2d')
plt.matshow(heatmap)
plt.show()



