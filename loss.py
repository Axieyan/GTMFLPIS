import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# class Loss(nn.Module):
#     def __init__(self, args):
#         super(Loss, self).__init__()
#         self.epsilon = args.epsilon
#
#     def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
#         """
#         Cross-Modal Projection Matching Loss(CMPM)
#         :param image_embeddings: Tensor with dtype torch.float32
#         :param text_embeddings: Tensor with dtype torch.float32
#         :param labels: Tensor with dtype torch.int32
#         :return:
#             i2t_loss: cmpm loss for image projected to text
#             t2i_loss: cmpm loss for text projected to image
#             pos_avg_sim: average cosine-similarity for positive pairs
#             neg_avg_sim: averate cosine-similarity for negative pairs
#         """
#
#         batch_size = image_embeddings.shape[0]
#         labels_reshape = torch.reshape(labels, (batch_size, 1))
#         labels_dist = labels_reshape - labels_reshape.t()
#         labels_mask = (labels_dist == 0)
#
#         image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
#         text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
#         image_proj_text = torch.matmul(image_embeddings, text_norm.t())
#         text_proj_image = torch.matmul(text_embeddings, image_norm.t())
#
#         # normalize the true matching distribution
#         labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
#
#         # 使用平方根作为归一化计算
#         # labels_mask_norm = labels_mask.float() / torch.sqrt(labels_mask.float().norm(dim=1))
#
#         # 将分母替换为全为 2 的张量
#         # labels_mask_norm = labels_mask.float() / torch.full_like(labels_mask, 2.0).float().norm(dim=1)
#
#         # 不进行归一化：如果你认为归一化不是必需的或不适用于你的情况，你可以直接使用原始的labels_mask作为结果，而不进行归一化操作。
#         # labels_mask_norm = labels_mask.float()
#
#         i2t_pred = F.softmax(image_proj_text, dim=1)
#         i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon)) # (4)
#         t2i_pred = F.softmax(text_proj_image, dim=1)
#         t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))
#
#         cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#
#         return cmpm_loss
#
#     def forward(self, img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
#                 txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels):
#         loss = 0.0
#
#         loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
#                 + self.compute_cmpm_loss(img_f41, txt_f41, labels) \
#                 + self.compute_cmpm_loss(img_f42, txt_f42, labels) \
#                 + self.compute_cmpm_loss(img_f43, txt_f43, labels) \
#                 + self.compute_cmpm_loss(img_f44, txt_f44, labels) \
#                 + self.compute_cmpm_loss(img_f45, txt_f45, labels) \
#                 + self.compute_cmpm_loss(img_f46, txt_f46, labels) \
#                 + self.compute_cmpm_loss(img_f4, txt_f4, labels)
#
#         return loss
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.CMPM = args.CMPM
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm_EN = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text_EN = torch.matmul(image_embeddings, text_norm_EN.t())
        text_proj_image_EN = torch.matmul(text_embeddings, image_norm.t())

        text_norm_CHS = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    image_proj_text_CHS = torch.matmul(image_embeddings, text_norm_CHS.t())
    text_proj_image_CHS = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

    i2t_pred_EN = F.softmax(image_proj_text_EN, dim=1)
    i2t_loss_EN = i2t_pred * (F.log_softmax(image_proj_text_EN, dim=1) - torch.log(labels_mask_norm + self.epsilon))
    t2i_pred_EN = F.softmax(text_proj_image_EN, dim=1)
    t2i_loss_EN = t2i_pred * (F.log_softmax(text_proj_image_EN, dim=1) - torch.log(labels_mask_norm + self.epsilon))

    i2t_pred_CHS = F.softmax(image_proj_text_CHS, dim=1)
    i2t_loss_CHS = i2t_pred * (F.log_softmax(image_proj_text_CHS, dim=1) - torch.log(labels_mask_norm + self.epsilon))
    t2i_pred_CHS = F.softmax(text_proj_image_CHS, dim=1)
    t2i_loss_CHS = t2i_pred * (F.log_softmax(text_proj_image_CHS, dim=1) - torch.log(labels_mask_norm + self.epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss_EN, dim=1)) + torch.mean(torch.sum(t2i_loss_EN, dim=1)) + torch.mean(
        torch.sum(i2t_loss_CHS, dim=1)) + torch.mean(torch.sum(t2i_loss_CHS, dim=1))

    return cmpm_loss


def forward(self, img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
            txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels):
    loss = 0.0
    if self.CMPM:
        loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
               + self.compute_cmpm_loss(img_f41, txt_f41, labels) \
               + self.compute_cmpm_loss(img_f42, txt_f42, labels) \
               + self.compute_cmpm_loss(img_f43, txt_f43, labels) \
               + self.compute_cmpm_loss(img_f44, txt_f44, labels) \
               + self.compute_cmpm_loss(img_f45, txt_f45, labels) \
               + self.compute_cmpm_loss(img_f46, txt_f46, labels) \
               + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    return loss
