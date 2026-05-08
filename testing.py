import torch

def translate_loop(
    model,
    tokenizer_src,
    tokenizer_tgt,
    device,
    max_len=100
):

    def causal_mask(size):
        mask = torch.tril(torch.ones((size, size))).bool()
        return mask.unsqueeze(0).unsqueeze(1)

    model.to(device)
    model.eval()

    sos_idx = 2
    eos_idx = 3
    pad_idx = 1

    while True:

        sentence = input("\nEnter English sentence: ")

        if sentence.lower() == "exit":
            print("Exiting translator...")
            break

        # ---------------- TOKENIZE SOURCE ----------------

        source_tokens = tokenizer_src.encode(sentence)

        source = torch.tensor(
            [[sos_idx] + source_tokens + [eos_idx]],
            dtype=torch.long
        ).to(device)

        # ---------------- ENCODER MASK ----------------

        source_mask = (source != pad_idx).unsqueeze(1).unsqueeze(2).to(device)

        with torch.no_grad():

            # ---------------- ENCODER ----------------

            encoder_output = model.encoder(source, source_mask)

            # ---------------- DECODER START ----------------

            decoder_input = torch.tensor(
                [[sos_idx]],
                dtype=torch.long
            ).to(device)

            while decoder_input.size(1) < max_len:

                decoder_mask = causal_mask(
                    decoder_input.size(1)
                ).to(device)

                # ---------------- DECODER ----------------

                out = model.decoder(
                    decoder_input,
                    encoder_output,
                    source_mask,
                    decoder_mask
                )

                # ---------------- NEXT TOKEN ----------------

                next_token = torch.argmax(
                    out[:, -1],
                    dim=1
                )

                # Stop if EOS
                if next_token.item() == eos_idx:
                    break

                # Append predicted token
                decoder_input = torch.cat(
                    [decoder_input, next_token.unsqueeze(0)],
                    dim=1
                )

        # ---------------- DECODE OUTPUT ----------------

        predicted_tokens = decoder_input.squeeze(0).cpu().numpy()

        tamil_text = tokenizer_tgt.decode(
            predicted_tokens,
            skip_special_tokens=True
        )

        print(f"\nTamil Translation: {tamil_text}")