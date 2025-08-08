import os
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from model import Transformer

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, n_enc_layers, n_dec_layers, dropout).to(device)
epochs = 15
lr = 10**-4

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

#greedy decoding for validation
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx, eos_idx = 2, 3
    encoder_output = model.encoder(source.to(device), source_mask.to(device))
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decoder(decoder_input, encoder_output, source_mask, decoder_mask)
        next_word = torch.max(out[:, -1], dim=1)[1]
        if next_word == eos_idx: break
        decoder_input = torch.cat([decoder_input, next_word.view(1, 1).to(device)], dim=1)

    return decoder_input.squeeze(0)[1:]


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=1):
    model.to(device)
    model.eval()
    source_texts, expected, predicted = [], [], []
    try:
      console_width = os.get_terminal_size().columns
    except OSError:
      console_width = 80

    with torch.no_grad():
        for count, batch in enumerate(validation_ds, start=1):
            encoder_input, encoder_mask = batch["english_token"].to(device), batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["english"][0]
            target_text = batch["tamil"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg(f"{'SOURCE:':>12}{source_text}\n{'TARGET:':>12}{target_text}\n{'PREDICTED:':>12}{model_out_text}\n{'-'*console_width}")
            if count == num_examples: break

def train_model():

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    checkpoint_path = "/kaggle/input/machine-translation/pytorch/default/1/Translation_Model_Params"
   
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        print("Model loaded from checkpoint.")
    else :
        print("Checkpoint not found. Training from scratch.")
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index= 1).to(device)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        if epoch <= 5:
            print("Training with Level 1 Data")
            train_dataloader = train_dataloader_1
            test_dataloader = test_dataloader_1
        elif epoch > 5 and epoch <= 10:
            print("Training with Level 2 Data")
            train_dataloader = train_dataloader_2
            test_dataloader = test_dataloader_2
        elif epoch > 10: #and epoch <= 15:
            print("Training with Level 3 Data")
            train_dataloader = train_dataloader_3
            test_dataloader = test_dataloader_3
        else:
            print("cant get the data")

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch+1:02d}")
        for batch in batch_iterator:
            encoder_input = batch['english_token'].to(device)
            decoder_input = batch['tamil_token'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            label = batch['tamil_target'].to(device)

            loss = loss_fn(output.view(-1, (len(tamil_tokenizer.word_to_id))), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.save(model.state_dict(), "/kaggle/working/Model_Params.pth")
        run_validation(model, test_dataloader, english_tokenizer, tamil_tokenizer, 24, device, lambda msg: batch_iterator.write(msg))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model()