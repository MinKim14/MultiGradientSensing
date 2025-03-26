import os
import torch
import torch.nn as nn
from tqdm import tqdm
from hand_gesture_dataset import HandGestureTransferDatasetExp
from model import SimpleLSTMModel

# Define global variables/objects as needed
criterion = nn.CrossEntropyLoss()
case_name = "all"  # "all", "inner", or "outer"

def train(epoch, model, dataloader, optimizer):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Train (Epoch {epoch})")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        inputs, labels = batch["input"].to("cuda").float(), batch["label"].long().to("cuda")
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 2)

        # Handle specific case_name logic
        if case_name == "inner":
            inputs = inputs[:, :, 0].unsqueeze(2)
            inputs = torch.cat([inputs, inputs], dim=2)
        elif case_name == "outer":
            inputs = inputs[:, :, 1].unsqueeze(2)
            inputs = torch.cat([inputs, inputs], dim=2)

        # Only the final label is used
        labels = labels[:, -1]

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / ((i + 1) * inputs.shape[0])
        pbar.set_postfix({"avg_loss": avg_loss})

    print(f"Epoch {epoch} Train Loss: {total_loss / len(dataloader)}")


def val(epoch, model, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["input"].to("cuda").float(), batch["label"].long().to("cuda")
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 2)

            if case_name == "inner":
                inputs = inputs[:, :, 0].unsqueeze(2)
                inputs = torch.cat([inputs, inputs], dim=2)
            elif case_name == "outer":
                inputs = inputs[:, :, 1].unsqueeze(2)
                inputs = torch.cat([inputs, inputs], dim=2)

            labels = labels[:, -1]
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Epoch {epoch} Validation Accuracy: {accuracy}")


def test(epoch, model, dataloader, idx):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["input"].to("cuda").float(), batch["label"].long().to("cuda")
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 2)

            if case_name == "inner":
                inputs = inputs[:, :, 0].unsqueeze(2)
                inputs = torch.cat([inputs, inputs], dim=2)
            elif case_name == "outer":
                inputs = inputs[:, :, 1].unsqueeze(2)
                inputs = torch.cat([inputs, inputs], dim=2)

            labels = labels[:, -1]
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Epoch {epoch} | Test Set {idx} Accuracy: {accuracy}")


def main():
    device = "cuda"
    window_size = 64
    batch_size = 8

    # Placeholder file paths (adjust for your data)
    train_file_name = ""
    val_file_name = ""
    test_file_name = ""

    train_dataset = HandGestureTransferDatasetExp(train_file_name, div=window_size)
    train_min_res = train_dataset.min_res
    train_max_res = train_dataset.max_res

    val_dataset = HandGestureTransferDatasetExp(
        val_file_name,
        div=window_size,
        label_dict=train_dataset.label_dict,
        min_res=train_min_res,
        max_res=train_max_res,
    )

    test_dataset1 = HandGestureTransferDatasetExp(
        test_file_name,
        div=window_size,
        label_dict=train_dataset.label_dict,
        min_res=train_min_res,
        max_res=train_max_res,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_dataloader1 = torch.utils.data.DataLoader(
        test_dataset1, batch_size=batch_size, shuffle=False, num_workers=0
    )

    num_class = len(train_dataset.label_dict.keys())

    run_name = "example_run"
    name = f"lstm_{run_name}_{case_name}_nLayer-5"
    os.makedirs(f"model_log/{name}", exist_ok=True)
    os.makedirs(f"train_csv_log/{name}", exist_ok=True)

    model = SimpleLSTMModel(
        input_size=2,
        hidden_size=64,
        output_size=num_class,
        num_layers=5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1000):
        train(epoch, model, train_dataloader, optimizer)
        val(epoch, model, val_dataloader)
        test(epoch, model, test_dataloader1, idx=1)


if __name__ == "__main__":
    main()