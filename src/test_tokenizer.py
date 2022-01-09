from trash.seq2seq import CharacterLevelTokenizer


sentences = [
    "abcdefg",
    "12345"
]

tokenizer = CharacterLevelTokenizer()
tokenizer.build_vocabulary(sentences)
tokenizer.build_int2token_and_token2int()
encoded_sentences = tokenizer.encode(sentences, for_decoder=True)

print(encoded_sentences)
