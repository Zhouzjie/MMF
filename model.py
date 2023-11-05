"""SGRAF model"""
import pickle
import torchtext

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from GAT import GATLayer
from resnet import resnet152
from torch.autograd import Variable
import copy



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def cosine_sim1(im, s):
    return im.mm(s.t())

# bottom-up attention 提取的region特征
class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

class EncoderText1(nn.Module):

    def __init__(self, opt):
        super(EncoderText1, self).__init__()
        self.embed_size = opt.embed_size
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)
        vocab = pickle.load(open('vocab/'+opt.data_name+'_vocab.pkl', 'rb'))
        word2idx = vocab.word2idx
        # self.init_weights()
        self.init_weights('glove', word2idx, opt.word_dim)
        self.dropout = nn.Dropout(0.1)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # return out
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = l2norm(cap_emb, dim=-1)


        return cap_emb

class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(cap_emb, lengths, batch_first=True, enforce_sorted=False)

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] + cap_emb[:, :, cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb

# 提取图像全局特征
class EncoderImageFull(nn.Module):

    def __init__(self, opt):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = opt.embed_size  # 1024

        self.cnn = resnet152(pretrained=True)
        # self.fc = nn.Sequential(nn.Linear(2048, self.embed_size), nn.ReLU(), nn.Dropout(0.1))
        self.fc = nn.Linear(opt.img_dim, self.embed_size)  # 2048降到1024
        # if not opt.finetune:
        #     print('image-encoder-resnet no grad!')
        #     for param in self.cnn.parameters():
        #         param.requires_grad = False
        # else:
        #     print('image-encoder-resnet fine-tuning !')
        print('image-encoder-resnet fine-tuning !')
        self.init_weights()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        features_orig = self.cnn(images)
        features_top = features_orig[-1]
        features = features_top.view(features_top.size(0), features_top.size(1), -1).transpose(2, 1)  # b, 49, 2048
        features = self.fc(features)

        return features

# 图像全局使用self-attention   待替换成池化策略
class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global

# 文本self-attention全局特征 待替换成池化策略
class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global

class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, embed_size, sim_dim, module_name='AVE', sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for i in range(sgr_step)])
        else:
            raise ValueError('Invalid input of opt.module_name in opts.py')

        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            for module in self.SGR_module:
                sim_emb = module(sim_emb)
            sim_vec = sim_emb[:, 0, :]

            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.sim = cosine_sim1

        self.sim_size = 16

    def forward(self, scores1, im, s):
        scores2 = self.sim(im, s)
        t = torch.sigmoid(scores1 + scores2)
        # t = 0.8
        scores = t * scores1 + (1-t) * scores2
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

# class ContrastiveLoss_DSRAN(nn.Module):
#     """
#     Compute contrastive loss
#     """
#
#     def __init__(self, margin=0):
#         super(ContrastiveLoss_DSRAN, self).__init__()
#         self.margin = margin
#         self.sim = cosine_sim1
#
#     def forward(self, im, s):
#         # compute image-sentence score matrix
#         scores = self.sim(im, s)
#         diagonal = scores.diag().view(im.size(0), 1)
#
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)
#         im_sn = scores - d1
#         c_sn = scores - d2
#         # compare every diagonal score to scores in its column
#         # caption retrieval
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         # compare every diagonal score to scores in its row
#         # image retrieval
#         cost_im = (self.margin + scores - d2).clamp(min=0)
#         # clear diagonals
#         mask = torch.eye(scores.size(0)) > .5
#         I = Variable(mask)
#         if torch.cuda.is_available():
#             I = I.cuda()
#         cost_s = cost_s.masked_fill_(I, 0)
#         cost_im = cost_im.masked_fill_(I, 0)
#
#         # keep the maximum violating negative for each query
#
#         cost_s = cost_s.max(1)[0]
#         cost_im = cost_im.max(0)[0]
#         return cost_s.sum() + cost_im.sum()



### GAT部分融入

class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph):
        hidden_states = input_graph
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)
        return hidden_states  # B, seq_len, D

class Fusion(nn.Module):
    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.f_size = opt.embed_size
        self.gate0 = nn.Linear(self.f_size, self.f_size)
        self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion0 = nn.Linear(self.f_size, self.f_size)
        self.fusion1 = nn.Linear(self.f_size, self.f_size)

    def forward(self, vec1, vec2):
        features_1 = self.gate0(vec1)
        features_2 = self.gate1(vec2)
        t = torch.sigmoid(self.fusion0(features_1) + self.fusion1(features_2))
        f = t * features_1 + (1 - t) * features_2
        return f

class DSRAN(nn.Module):
    def __init__(self, opt):
        super(DSRAN, self).__init__()
        self.K = opt.K
        self.img_enc = EncoderImageFull(opt) # b,49,1024  49=7x7
        self.rcnn_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        # self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
        #                            opt.embed_size, opt.num_layers,
        #                            use_bi_gru=opt.bi_gru,
        #                            no_txtnorm=opt.no_txtnorm)
        self.txt_enc = EncoderText1(opt)

        config_rcnn = GATopt(opt.embed_size, 1)
        config_img = GATopt(opt.embed_size, 1)
        config_cap = GATopt(opt.embed_size, 1)
        config_joint = GATopt(opt.embed_size, 1)
        # SSR
        self.gat_1 = GAT(config_rcnn)
        self.gat_2 = GAT(config_img)
        self.gat_cap = GAT(config_cap)
        # JSR
        self.gat_cat_1 = GAT(config_joint)
        if self.K == 2:
            self.gat_cat_2 = GAT(config_joint)
            self.fusion = Fusion(opt)
        elif self.K == 4:
            self.gat_cat_2 = GAT(config_joint)
            self.gat_cat_3 = GAT(config_joint)
            self.gat_cat_4 = GAT(config_joint)
            self.fusion = Fusion(opt)
            self.fusion2 = Fusion(opt)
            self.fusion3 = Fusion(opt)

    def forward(self, images, image_raw, captions, lengths):
        img_emb_orig = self.gat_2(self.img_enc(image_raw))
        rcnn_emb = self.rcnn_enc(images)
        img_sgr = self.rcnn_enc(images)
        rcnn_emb = self.gat_1(rcnn_emb)
        img_cat = torch.cat((img_emb_orig, rcnn_emb), 1)
        img_cat_1 = self.gat_cat_1(img_cat)
        img_cat_1 = torch.mean(img_cat_1, dim=1)
        if self.K == 1:
            img_cat = img_cat_1
        elif self.K == 2:
            img_cat_2 = self.gat_cat_2(img_cat)
            img_cat_2 = torch.mean(img_cat_2, dim=1)
            img_cat = self.fusion(img_cat_1, img_cat_2)
        elif self.K == 4:
            img_cat_2 = self.gat_cat_2(img_cat)
            img_cat_2 = torch.mean(img_cat_2, dim=1)
            img_cat_3 = self.gat_cat_3(img_cat)
            img_cat_3 = torch.mean(img_cat_3, dim=1)
            img_cat_4 = self.gat_cat_4(img_cat)
            img_cat_4 = torch.mean(img_cat_4, dim=1)
            img_cat_1_1 = self.fusion(img_cat_1, img_cat_2)
            img_cat_1_2 = self.fusion2(img_cat_3, img_cat_4)
            img_cat = self.fusion3(img_cat_1_1, img_cat_1_2)
        img_emb = l2norm(img_cat)
        cap_emb = self.txt_enc(captions, lengths)
        cap_sgr = self.txt_enc(captions, lengths)
        cap_gat = self.gat_cap(cap_emb)
        cap_embs = l2norm(torch.mean(cap_gat, dim=1))

        return img_emb, cap_embs, img_sgr, cap_sgr, lengths

### GAT部分融入

class MMF(object):
    def __init__(self, opt):
        self.grad_clip = opt.grad_clip
        self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)
        self.DSRAN = DSRAN(opt)

        if torch.cuda.is_available():
            self.sim_enc.cuda()
            self.DSRAN.cuda()
            cudnn.benchmark = True
        # Loss and optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        # self.criterion1 = ContrastiveLoss_DSRAN(margin=opt.margin)
        params = list(self.DSRAN.parameters())
        params += list(self.sim_enc.parameters())

        self.params = params

        # self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.optimizer = torch.optim.SGD(params, lr=opt.learning_rate, momentum=0.8,
                                         weight_decay=1e-2)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.DSRAN.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.DSRAN.load_state_dict(state_dict[0])
        self.sim_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.DSRAN.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.DSRAN.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, image_raw, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            image_raw = image_raw.cuda()
            captions = captions.cuda()

        img_emb, cap_emb, img_sgr, cap_sgr, lengths = self.DSRAN(images, image_raw, captions, lengths)
        return img_emb, cap_emb, img_sgr, cap_sgr, lengths

    def forward_sim(self, img_sgr, cap_sgr, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_sgr, cap_sgr, cap_lens)
        return sims

    def forward_loss(self, sims, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims, img_emb, cap_emb)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, image_raw, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, img_sgr, cap_sgr, cap_lens = self.forward_emb(images, image_raw, captions, lengths)
        sims = self.forward_sim(img_sgr, cap_sgr, cap_lens)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims, img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
