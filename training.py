import torch
import torch.nn as nn
import os
from tqdm import tqdm
from validation import run_validation


def train_model(train_dataloader, test_dataloader, model, english_tokenizer, tamil_tokenizer, max_len, device, epochs, lr = 1e-4):

    save_checkpoint = "assets/translation_model_v1.pth"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    checkpoint_path = "assets/translation_model_v1.pth"
    
    initial_epoch = 0

    model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1).to(device)

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading model and optimizer state...")
        checkpoint = torch.load(checkpoint_path ,map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        initial_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        print(f"Model loaded from checkpoint. Resuming from epoch {initial_epoch + 1}.")
    else:
        print("Checkpoint not found. Training from scratch.")


    for epoch in range(initial_epoch, initial_epoch + epochs):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1:02d}")

        for batch in batch_iterator:

            optimizer.zero_grad(set_to_none=True)

            encoder_input = batch['english_token'].to(device)
            decoder_input = batch['tamil_token'].to(device)

            encoder_mask = batch['encoder_mask'].to(device).bool()
            decoder_mask = batch['decoder_mask'].to(device).bool()

            label = batch['tamil_target'].to(device)

            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            loss = loss_fn(
                output.view(-1, output.size(-1)),
                label.view(-1)
            )

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_checkpoint)

        run_validation(
            model,
            test_dataloader,
            english_tokenizer,
            tamil_tokenizer,
            max_len,
            device,
            lambda msg: batch_iterator.write(msg)
        )