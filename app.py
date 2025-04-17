import math
import time
import datetime
import threading
import random
import csv

import numpy as np
import pandas as pd
import gradio as gr
import keyboard
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports (with placeholder path)
from exp.utils import alphabet_to_int, int_to_alphabet   # Adjust as needed

# Replace directory paths with placeholders, keep file names fixed
SERIAL_SAVE_DIR = "/path/to/serial_data_dir/"  # <-- placeholder
MODEL_CHECKPOINT_DIR = "/path/to/model_checkpoint/"  # <-- placeholder
DATASET_DIR = "/path/to/dataset_dir/"  # <-- placeholder
ASSETS_DIR = "/path/to/assets/sign_language/4x/"  # <-- placeholder
CSS_PATH = "/path/to/custom.css"  # <-- placeholder

from hand_gesture_dataset import HandGestureTransferDatasetExp
from model import SimpleLSTMModel


class ReadLine:
    """
    Helper class to read lines from a serial buffer.
    """
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[: i + 1]
            self.buf = self.buf[i + 1 :]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            idx = data.find(b"\n")
            if idx >= 0:
                r = self.buf + data[: idx + 1]
                self.buf[0:] = data[idx + 1 :]
                return r
            else:
                self.buf.extend(data)


def get_time():
    """Utility to get current datetime."""
    return datetime.datetime.now()


# ---- SERIAL PORT SETUP (Adjust COM port as needed) ----
ser = serial.Serial(
    port="COM22",
    baudrate=9600,
)
rl = ReadLine(ser)

data_collect_file_name = f"serial_data_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
save_path = f"{SERIAL_SAVE_DIR}{data_collect_file_name}"

f = open(save_path, "w", newline="")
writer = csv.writer(f, delimiter=",")
writer.writerow(["time", "sensor1", "sensor2", "label"])

st_time = time.time()
plot_measurements = np.zeros((100, 2))
cur_key_input = 0
toggle = True


def reading_serial():
    """
    Continuously reads from the serial device, updates
    global plot data, and logs to CSV.
    """
    global rl, writer, st_time, plot_measurements, cur_key_input, toggle
    cur_numpad = 0
    while toggle:
        try:
            data = rl.readline().decode("utf-8").strip().split(",")
            measurement = np.array(data[:2], dtype=float)
            plot_measurements = np.concatenate(
                [plot_measurements[1:], measurement.reshape(1, -1)]
            )
            current_time = time.time() - st_time

            pressed_keys = keyboard.get_hotkey_name()
            if pressed_keys:
                try:
                    cur_numpad = alphabet_to_int[pressed_keys]
                except KeyError:
                    pass

            cur_key_input = cur_numpad
            for_log = [current_time, *measurement, cur_numpad]
            writer.writerow(for_log)
        except:
            continue


t1 = threading.Thread(target=reading_serial)


def get_serial_plot():
    """
    Returns the latest sensor data plot and current key input.
    """
    global plot_measurements, cur_key_input
    fig = plt.figure()
    fig.suptitle("Sensor Data", color="white")
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    x = np.arange(100)
    ax.plot(x, plot_measurements[:, 0], label="sensor1", color="#3a86ff")
    ax.plot(x, plot_measurements[:, 1], label="sensor2", color="#ff006e")
    ax.legend()

    return [str(cur_key_input), fig]


def t1_toggle():
    """
    Toggles the serial reading thread on/off.
    """
    global t1, toggle
    if t1.is_alive():
        toggle = False
        t1.join()
        t1 = threading.Thread(target=reading_serial)
        print("Closing file.")
        f.close()
        return "Reading Disconnected"
    else:
        toggle = True
        t1.start()
        return "Reading Connected"


# ---- TRAINING (TRANSFER LEARNING) SETUP ----
criterion = nn.CrossEntropyLoss()
model = None
train_dataset = None
window_size = 64


def train(epoch, model, dataloader, optimizer):
    """
    One epoch of training.
    """
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader)

    for i, s in enumerate(pbar):
        optimizer.zero_grad()
        inputs, labels = s["input"].to("cuda").float(), s["label"].long().to("cuda")
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 2)
        labels = labels[:, -1]

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_description(f"Epoch {epoch} total_loss {total_loss / ((i + 1) * 64)}")

    print(f"Epoch {epoch} Loss {total_loss / len(dataloader)}")
    return model


def transfer_train():
    """
    Stops the serial thread, then loads a pre-trained model
    and re-trains it on newly collected data.
    """
    global t1, toggle, model, train_dataset
    if t1.is_alive():
        toggle = False
        t1.join()
        t1 = threading.Thread(target=reading_serial)

    # Placeholders for your local dataset & model checkpoint
    file_names = [
        f"{DATASET_DIR}sensor_data_a_to_z.csv",
    ]
    model_names = [
        f"{MODEL_CHECKPOINT_DIR}lstm_full_a2z_all_nLayer-5/train_best.pth",
    ]

    # Example: picking the first dataset & model
    idx = 0
    file_name = file_names[idx]
    model_checkpoint = model_names[idx]

    pretrain_dataset = HandGestureTransferDatasetExp(
        file_name, div=window_size
    )
    num_class = len(pretrain_dataset.label_dict.keys())
    print(pretrain_dataset.label_dict.keys(), num_class)

    train_min_res = pretrain_dataset.min_res
    train_max_res = pretrain_dataset.max_res

    # Use the newly collected data
    train_dataset = HandGestureTransferDatasetExp(
        save_path,
        div=window_size,
        label_dict=pretrain_dataset.label_dict,
        min_res=train_min_res,
        max_res=train_max_res,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0
    )

    model = SimpleLSTMModel(
        input_size=2,
        hidden_size=64,
        output_size=num_class,
        num_layers=5,
    ).to("cuda")

    model.load_state_dict(torch.load(model_checkpoint))
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for i in range(20):
        model = train(i, model, train_dataloader, optimizer)

    model.eval()


# ---- PREDICTION LOOP ----
prediction_toggle = False
cur_prediction = ""


def start_prediction():
    """
    Continuously fetches the last window of live sensor data,
    passes it through the trained model, and updates
    global cur_prediction.
    """
    global t1, toggle, model, train_dataset, plot_measurements, cur_prediction

    if not t1.is_alive():
        toggle = True
        t1.start()

    while prediction_toggle:
        num_class = len(train_dataset.label_dict.keys())
        cur_input = plot_measurements[-window_size:].reshape(1, window_size, 2)
        # Normalize using the dataset stats
        cur_input = (cur_input - train_dataset.min_res) / (
            train_dataset.max_res - train_dataset.min_res
        )
        cur_input = torch.from_numpy(cur_input).to("cuda").float()

        output, _ = model(cur_input)
        _, predicted = torch.max(output.data, 1)
        predicted_label = train_dataset.index2label[predicted.item()]
        cur_prediction = int_to_alphabet[int(predicted_label)]

        print("Prediction:", cur_prediction)
        time.sleep(0.1)


t2 = threading.Thread(target=start_prediction)


def t2_toggle():
    """
    Toggles the prediction thread on/off.
    Also toggles the serial reading thread if needed.
    """
    global t2, prediction_toggle
    if t2.is_alive():
        prediction_toggle = False
        t2.join()
        t2 = threading.Thread(target=start_prediction)
        return "Not Predicting"
    else:
        prediction_toggle = True
        t2.start()
        t1_toggle()
        return "Predicting"


# ---- SIGN LANGUAGE IMAGE DICTIONARY (Directory is placeholder) ----
sign_language_img = {
    "a": f"{ASSETS_DIR}a.png",
    "b": f"{ASSETS_DIR}b.png",
    "c": f"{ASSETS_DIR}c.png",
    "d": f"{ASSETS_DIR}d.png",
    "e": f"{ASSETS_DIR}e.png",
    "f": f"{ASSETS_DIR}f.png",
    "g": f"{ASSETS_DIR}g.png",
    "h": f"{ASSETS_DIR}h.png",
    "i": f"{ASSETS_DIR}i.png",
    "j": f"{ASSETS_DIR}j.png",
    "k": f"{ASSETS_DIR}k.png",
    "l": f"{ASSETS_DIR}l.png",
    "m": f"{ASSETS_DIR}m.png",
    "n": f"{ASSETS_DIR}n.png",
    "o": f"{ASSETS_DIR}o.png",
    "p": f"{ASSETS_DIR}p.png",
    "q": f"{ASSETS_DIR}q.png",
    "r": f"{ASSETS_DIR}r.png",
    "s": f"{ASSETS_DIR}s.png",
    "t": f"{ASSETS_DIR}t.png",
    "u": f"{ASSETS_DIR}u.png",
    "v": f"{ASSETS_DIR}v.png",
    "w": f"{ASSETS_DIR}w.png",
    "x": f"{ASSETS_DIR}x.png",
    "y": f"{ASSETS_DIR}y.png",
    "z": f"{ASSETS_DIR}z.png",
}


def load_image():
    """
    Returns a random image from the sign_language_img dictionary.
    """
    file_path = random.choice(list(sign_language_img.values()))
    print(file_path)
    return gr.Image(file_path)


def get_prediction():
    """
    Returns current predicted label and corresponding sign language image.
    """
    global cur_prediction
    if cur_prediction == "":
        return ["", gr.Image(f"{ASSETS_DIR}none.png")]
    file_path = sign_language_img.get(cur_prediction, f"{ASSETS_DIR}none.png")
    return [str(cur_prediction), gr.Image(file_path)]


# ---- GRADIO UI ----
with open(CSS_PATH, "r", encoding="utf-8") as fcss:
    customCSS = fcss.read()

with gr.Blocks(css=customCSS) as demo:
    demo.load(
        None,
        None,
        js="""
            () => {
                const params = new URLSearchParams(window.location.search);
                if (!params.has('__theme')) {
                    params.set('__theme', 'dark');
                    window.location.search = params.toString();
                }
            }
        """,
    )
    with gr.Row():
        with gr.Column():
            connection_status = gr.Textbox(label="Connection Status", elem_classes="textbox")
            sc_button = gr.Button(value="Serial Reading Thread Connect", elem_classes="button")
            sc_button.click(t1_toggle, None, connection_status)
        with gr.Column():
            key_input = gr.Textbox(label="Current Key Input", elem_classes="textbox")
            transfer_button = gr.Button(value="Transfer Data", elem_classes="button")
            transfer_button.click(transfer_train, None, None)

    with gr.Row(equal_height=True):
        with gr.Column():
            plot = gr.Plot(elem_classes="plot")
        with gr.Column():
            prediction_img = gr.Image(f"{ASSETS_DIR}none.png", elem_classes="image")

    with gr.Row():
        with gr.Column():
            prediction_status = gr.Textbox(label="Prediction Status", elem_classes="textbox")
        with gr.Column():
            prediction_label = gr.Textbox(label="Prediction Label", elem_classes="textbox")

    with gr.Row():
        prediction_button = gr.Button(value="Start Predict", elem_classes="button")
        prediction_button.click(t2_toggle, None, prediction_status)

    # Update the plot and current key input ~ every 0.1 seconds
    dep = demo.load(get_serial_plot, None, [key_input, plot], every=0.1)

    # Update the prediction text & image ~ every 0.03 seconds
    dep3 = demo.load(get_prediction, None, [prediction_label, prediction_img], every=0.03)

if __name__ == "__main__":
    demo.queue().launch()
