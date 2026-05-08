import json

class WordLevelTokenizer:
    def __init__(self, special_tokens=None):
        self.word_to_id = {}
        self.id_to_word = {}

        # Default special tokens
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

        # Build vocab with special tokens first
        self.build_vocab(self.special_tokens)

    def build_vocab(self, tokens):
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

    def tokenize(self, text):
        # Simple tokenizer (you can improve later)
        return text.strip().split()

    def fit_on_texts(self, texts):
        for text in texts:
            tokens = self.tokenize(text)
            self.build_vocab(tokens)

    def encode(self, text, add_special_tokens=False):
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = ["<SOS>"] + tokens + ["<EOS>"]

        return [
            self.word_to_id.get(token, self.word_to_id["<UNK>"])
            for token in tokens
        ]

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for idx in token_ids:
            token = self.id_to_word.get(idx, "<UNK>")

            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        return " ".join(tokens)

    def vocab_size(self):
        return len(self.word_to_id)

    # Save tokenizer
    def save(self, path):
        data = {
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "special_tokens": self.special_tokens
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # Load tokenizer
    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(special_tokens=data["special_tokens"])
        tokenizer.word_to_id = data["word_to_id"]

        # Convert keys back to int (JSON stores them as strings)
        tokenizer.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}

        return tokenizer