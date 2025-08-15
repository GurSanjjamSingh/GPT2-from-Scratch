from model import GPTModel
from data import create_dataloader_v1
from utils import calc_loss_batch, calc_loss_loader
from config import GPT_CONFIG_124M, TRAIN_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])

for epoch in range(TRAIN_CONFIG["num_epochs"]):
    model.train()
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")