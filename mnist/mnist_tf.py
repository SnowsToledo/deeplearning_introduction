import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


def train_and_log_tf(params):
    """

    :type params: Dict
    """
    # Habilita o autolog do MLflow com TensorFlow
    mlflow.tensorflow.autolog()

    # Hiperparâmetros
    batch_size = params["batch_size"]
    lr = params["lr"]
    epochs = params["epochs"]

    # Carregar e preparar os dados
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    with mlflow.start_run():
        # Log manual dos hiperparâmetros
        mlflow.log_params(params)

        # Criar o modelo
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # Compilar o modelo
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Treinar o modelo
        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=2
        )

        # Avaliação final
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_test_accuracy", test_acc)

        # Logar o modelo como artefato
        mlflow.tensorflow.log_model(model, artifact_path="model")


# Hiperparâmetros
params = {
    "batch_size": 64,
    "lr": 0.01,
    "epochs": 5
}

if __name__ == "__main__":
    train_and_log_tf(params)
