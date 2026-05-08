import torch
from ast import literal_eval
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def create_encoder_mask(src, pad_token):
    # (B, 1, 1, src_len)
    return (src != pad_token).unsqueeze(1).unsqueeze(2)


def create_decoder_mask(tgt, pad_token):
    B, tgt_len = tgt.shape

    # Padding mask → (B, 1, 1, tgt_len)
    padding_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)

    # Causal mask → (1, 1, tgt_len, tgt_len)
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    # Combine → (B, 1, tgt_len, tgt_len)
    return padding_mask & causal_mask


def collate_fn(batch, pad_token=1):

    eng_batch = []
    tam_input_batch = []
    tam_target_batch = []

    for item in batch:

        eng = torch.tensor(item["eng_tokens"])
        tam_in = torch.tensor(item["tam_tokens"][:-1])
        tam_out = torch.tensor(item["tam_tokens"][1:])

        eng_batch.append(eng)
        tam_input_batch.append(tam_in)
        tam_target_batch.append(tam_out)

    # Dynamic padding
    eng_batch = pad_sequence(
        eng_batch,
        batch_first=True,
        padding_value=pad_token
    )

    tam_input_batch = pad_sequence(
        tam_input_batch,
        batch_first=True,
        padding_value=pad_token
    )

    tam_target_batch = pad_sequence(
        tam_target_batch,
        batch_first=True,
        padding_value=pad_token
    )

    # Masks
    encoder_mask = create_encoder_mask(eng_batch, pad_token)
    decoder_mask = create_decoder_mask(tam_input_batch, pad_token)

    return {
        "english_token": eng_batch,
        "tamil_token": tam_input_batch,
        "tamil_target": tam_target_batch,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask
    }

class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.english_col = self._find_column("english_tokens", "Tokenized_English")
        self.tamil_col = self._find_column("tamil_tokens", "Tokenized_Tamil")

    def _find_column(self, *candidates):
        for column in candidates:
            if column in self.df.columns:
                return column
        raise KeyError(
            f"None of the expected columns were found: {candidates}. "
            f"Available columns: {list(self.df.columns)}"
        )

    def _tokens_from_row(self, row, column):
        tokens = row[column]
        if isinstance(tokens, str):
            tokens = literal_eval(tokens)
        return tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "eng_tokens": self._tokens_from_row(row, self.english_col),
            "tam_tokens": self._tokens_from_row(row, self.tamil_col)
        }