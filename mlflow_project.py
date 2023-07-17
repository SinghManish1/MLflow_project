import mlflow
import mlflow.sklearn
import random
import logging
import warnings
import mlflow.pyfunc
import pandas as pd
import numpy as np
from mlflow import pyfunc
import mlflow.tensorflow
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
mlflow.set_experiment('Transformer model')
# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)

mlflow.tensorflow.autolog()

# Maximum duration of the input audio file we feed to our Wav2Vec 2.0 model.
MAX_DURATION = 1
# Sampling rate is the number of samples of audio recorded every second
SAMPLING_RATE = 16000
BATCH_SIZE = 64  # Batch-size for training and evaluating our model.
NUM_CLASSES = 12  # Number of classes our dataset will have.
HIDDEN_DIM = 768  # Dimension of our model output (768 in case of Wav2Vec 2.0 - Base).
MAX_SEQ_LENGTH = MAX_DURATION * SAMPLING_RATE  # Maximum length of the input audio file.
# Wav2Vec 2.0 results in an output frequency with a stride of about 20ms.
MAX_FRAMES = 49
MAX_EPOCHS = 10  # Maximum number of training epochs.

MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # Name of pretrained model from Hugging Face Model Hub

from datasets import load_dataset

speech_commands_v1 = load_dataset("superb", "ks")

print(speech_commands_v1)

speech_commands_v1 = speech_commands_v1["train"].train_test_split(

    train_size=0.3, test_size=0.3, stratify_by_column="label"
)

speech_commands_v1 = speech_commands_v1.filter(

    lambda x: x["label"]
    != (
        speech_commands_v1["train"].features["label"].names.index("_unknown_")

        and speech_commands_v1["train"].features["label"].names.index("_silence_")
    )
)

speech_commands_v1["train"] = speech_commands_v1["train"].select(

    [i for i in range((len(speech_commands_v1["train"]) // BATCH_SIZE) * BATCH_SIZE)]
)
speech_commands_v1["test"] = speech_commands_v1["test"].select(

    [i for i in range((len(speech_commands_v1["test"]) // BATCH_SIZE) * BATCH_SIZE)]
)

print(speech_commands_v1)

labels = speech_commands_v1["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
print(id2label)

from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_CHECKPOINT, return_attention_mask=True
)
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=True,
    )
    return inputs
# This line with pre-process our speech_commands_v1 dataset. We also remove the "audio"
# and "file" columns as they will be of no use to us while training.
processed_speech_commands_v1 = speech_commands_v1.map(
    preprocess_function, remove_columns=["audio", "file"], batched=True
)

#taking only some part of the data
train = processed_speech_commands_v1["train"].shuffle(seed=42).with_format("numpy")
x = int(len(train) * 0.2)
train_data = train[:x]
train=train_data

test = processed_speech_commands_v1["test"].shuffle(seed=42).with_format("numpy")
x = int(len(test) * 0.2)
test_data = test[:x]
test=test_data

from transformers import TFWav2Vec2Model
def mean_pool(hidden_states, feature_lengths):
    pooled_state=tf.reduce_mean(hidden_states, axis=1)
    return pooled_state

class TFWav2Vec2ForAudioClassification(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, model_checkpoint, num_classes):
        super().__init__()
        # Instantiate the Wav2Vec 2.0 model without the Classification-Head
        self.wav2vec2 = TFWav2Vec2Model.from_pretrained(
            model_checkpoint, apply_spec_augment=False, from_pt=True
        )
        self.pooling = layers.GlobalAveragePooling1D()
        # Drop-out layer before the final Classification-Head
        self.intermediate_layer_dropout = layers.Dropout(0.3)
        # Classification-Head
        self.final_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # We take only the first output in the returned dictionary corresponding to the
        # output of the last layer of Wav2vec 2.0
        hidden_states = self.wav2vec2(inputs["input_values"])[0]

        # If attention mask does exist then mean-pool only un-masked output frames
        if tf.is_tensor(inputs["attention_mask"]):
            # Get the length of each audio input by summing up the attention_mask
            # (attention_mask = (BATCH_SIZE x MAX_SEQ_LENGTH) ∈ {1,0})
            audio_lengths = tf.cumsum(inputs["attention_mask"], -1)[:, -1]
            # Get the number of Wav2Vec 2.0 output frames for each corresponding audio input
            # length
            feature_lengths = self.wav2vec2.wav2vec2._get_feat_extract_output_lengths(
                audio_lengths
            )
            pooled_state = mean_pool(hidden_states, feature_lengths)
        # If attention mask does not exist then mean-pool only all output frames
        else:
            pooled_state = self.pooling(hidden_states)

        intermediate_state = self.intermediate_layer_dropout(pooled_state)
        final_state = self.final_layer(intermediate_state)

        return final_state
    def get_config(self):
        config = super(TFWav2Vec2ForAudioClassification, self).get_config()
        config.update({
           'wav2vec2': self.wav2vec2,
           'pooling': self.pooling,
           'intermediate_layer_dropout':self.intermediate_layer_dropout,
           'final_layer': self.final_layer


        })
        return config

def build_model():
    # Model's input
    inputs = {
        "input_values": tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="float32"),
        "attention_mask": tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="int32"),
    }
    # Instantiate the Wav2Vec 2.0 model with Classification-Head using the desired
    # pre-trained checkpoint
    wav2vec2_model = TFWav2Vec2ForAudioClassification(MODEL_CHECKPOINT, NUM_CLASSES)(
        inputs
    )
    # Model
    model = tf.keras.Model(inputs, wav2vec2_model)
    # Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Optimizer
    optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-5)
    # Compile and return
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


model = build_model()

def mlflow_run(params,run_name="Tracking Experiment: Transformer Wav2Vec Model "):
    # Remove targets from training dictionaries
    train_x = {x: y for x, y in train.items() if x != "label"}
    test_x = {x: y for x, y in test.items() if x != "label"}
    with mlflow.start_run(run_name=run_name) as run:
      model.fit(
        train_x,
        train["label"],
        validation_data=(test_x, test["label"]),
        batch_size=BATCH_SIZE,
        epochs=params['epochs'],
       )
    return (run.info.experiment_id, run.info.run_id)

# Short example how to run a MLflow GitHub Project programmatically using
# MLflow Fluent APIs https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.run
#
if __name__ == '__main__':
   # suppress any deprecated warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)

   epochs = 5
   params = {'epochs': epochs}
   (exp_id, run_id) = mlflow_run(params)

   print(f"Finished Experiment id={exp_id} and run id = {run_id}")

