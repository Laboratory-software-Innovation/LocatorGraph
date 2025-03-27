import argparse
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def load_dataset():
    dataset, info = tfds.load('reddit_tifu', with_info=True, split='train')
    text_data = ""
    for example in tfds.as_numpy(dataset):
        text_data += example['title'].decode('utf-8') + "\n"
    return text_data

def prepare_tokenizer(text_data, vocab_size, seq_len):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len + 1
    )
    tokenizer.adapt([text_data])
    return tokenizer

def prepare_data(text, tokenizer, seq_len):
    tokens = tokenizer([text])[0]
    inputs = []
    targets = []
    for i in range(0, len(tokens) - seq_len):
        inputs.append(tokens[i:i+seq_len])
        targets.append(tokens[i+1:i+seq_len+1])
    return tf.data.Dataset.from_tensor_slices((inputs, targets))

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, activation_fn):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation=activation_fn),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs, use_causal_mask=True)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

def build_gpt_model(vocab_size, seq_len, embed_dim, num_heads, ff_dim, num_layers, activation_fn="relu"):
    inputs = layers.Input(shape=(seq_len,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, activation_fn)(x)
    x = layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs, x)

def generate_text(model, tokenizer, prompt, seq_len, max_length=100):
    input_ids = tokenizer([prompt])[0][:seq_len]
    input_ids = tf.expand_dims(input_ids, 0)
    for _ in range(max_length):
        predictions = model(input_ids)
        next_id = tf.argmax(predictions[:, -1, :], axis=-1)
        input_ids = tf.concat([input_ids, tf.expand_dims(next_id, -1)], axis=-1)
        input_ids = input_ids[:, -seq_len:]
    inverse_vocab = dict(enumerate(tokenizer.get_vocabulary()))
    decoded_tokens = [inverse_vocab.get(int(i), '') for i in input_ids.numpy()[0]]
    return ''.join(decoded_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--activation_fn", type=str, default="relu")
    args = parser.parse_args()

    print("\nLoading dataset...")
    text_data = load_dataset()
    vocab_size = 256
    seq_len = 128

    tokenizer = prepare_tokenizer(text_data, vocab_size, seq_len)
    train_dataset = prepare_data(text_data, tokenizer, seq_len).batch(32).shuffle(1024).prefetch(tf.data.AUTOTUNE)

    activation_fn = getattr(tf.keras.activations, args.activation_fn)
    model = build_gpt_model(vocab_size, seq_len, args.embedding_dim, args.num_heads, args.ff_dim, args.num_layers, activation_fn)
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()

    print("\nTraining...")
    model.fit(train_dataset, epochs=args.epochs)

    print("\nGenerated Text:")
    print(generate_text(model, tokenizer, "To be or not to be", seq_len=seq_len, max_length=100))

if __name__ == "__main__":
    main()
