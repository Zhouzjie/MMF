import os
import time
import shutil

import torch
import numpy

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import MMF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

import logging
import tensorboard_logger as tb_logger
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    with torch.no_grad():
        for i, (images, image_raw, captions, lengths, ids) in enumerate(data_loader):
            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, image_raw, captions, lengths)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = torch.zeros(len(data_loader.dataset), img_emb.size(1)).cuda()
                cap_embs = torch.zeros(len(data_loader.dataset), cap_emb.size(1)).cuda()
            # print(ids)
            img_embs[list(ids)] = img_emb
            cap_embs[list(ids)] = cap_emb

            # measure accuracy and record loss
            model.forward_loss(img_emb, cap_emb)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
            del image_raw, captions

    return img_embs, cap_embs


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # clear duplicate 5*images and keep 1*images
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r_sum', r_sum, step=model.Eiters)

    return r_sum

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


opt = opts.parse_opt()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
tb_logger.configure(opt.logger_name, flush_secs=5)

# Load Vocabulary Wrapper
vocab = pickle.load(open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
opt.vocab_size = len(vocab)

# Load data loaders
train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.crop_size,
                                                opt.batch_size, opt.workers, opt)


# Construct the model
model = MMF(opt)
batch_time = AverageMeter()
data_time = AverageMeter()
train_logger = LogCollector()

end = time.time()
for i, train_data in enumerate(train_loader):
    # switch to train mode
    model.train_start()

    # measure data loading time
    data_time.update(time.time() - end)

    # make sure train logger is used
    model.logger = train_logger

    # Update the model
    model.train_emb(*train_data)
    break
# #
# for epoch in range(opt.num_epochs):
#     print(opt.logger_name)
#     print(opt.model_name)
#
#     adjust_learning_rate(opt, model.optimizer, epoch)
#
#     # train for one epoch
#     train(opt, train_loader, model, epoch, val_loader)
#
#     # evaluate on validation set
#     r_sum = validate(opt, val_loader, model)
#
#     # remember best R@ sum and save checkpoint
#     is_best = r_sum > best_rsum
#     best_rsum = max(r_sum, best_rsum)
#
#     if not os.path.exists(opt.model_name):
#         os.mkdir(opt.model_name)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'model': model.state_dict(),
#         'best_rsum': best_rsum,
#         'opt': opt,
#         'Eiters': model.Eiters,
#     }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')
