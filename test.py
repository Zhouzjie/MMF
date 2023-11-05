import data
import opts
import os

opt = opts.parse_opt()
from vocab import Vocabulary, deserialize_vocab
vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))

opt.vocab_size = len(vocab)
train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

for data in val_loader:
    break