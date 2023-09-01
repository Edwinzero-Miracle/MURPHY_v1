import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models

from models.gnn_modules import GraphConv
from models.hrca_module import HRCAModule


class MURPHYNet(nn.Module):
    """

    """

    def __init__(self, type: str, backbone: str, relations: dict,
                 img_fdim: int, gcn_dim: int, out_dim: int, activation: str,
                 hrca_channels: int,
                 num_instrument: int, num_action: int, num_object: int,
                 num_triplets: int, num_step: int, num_task: int,
                 use_rlls_rc_mode: bool,
                 use_multi_layer_gcn: bool, enable_adj_normalize: bool,
                 use_prior_knowledge: bool,
                 disable_batch_type_sim: bool = False,
                 self_weight=1.0):
        super(MURPHYNet, self).__init__()

        self.img_fdim = img_fdim
        self.gcn_dim = gcn_dim
        self.out_dim = out_dim
        self.hrca_channels = hrca_channels
        self.use_rlls_rc_mode = use_rlls_rc_mode
        self.use_multilayer_gcn = use_multi_layer_gcn
        self.enable_adj_normalize = enable_adj_normalize
        self.disable_batch_type_sim = disable_batch_type_sim
        self.relations = relations
        self.use_prior_knowledge = use_prior_knowledge

        print("  [MURPHYNet] :: use relations: ", relations.keys(), " !")
        print("  [MURPHYNet] :: number categories: (step: ", num_step, "), ",
              "(task: ", num_task, "), ",
              "(triplet: ", num_triplets, "), ",
              "(instrument: ", num_instrument, "), ",
              "(action: ", num_action, "), ",
              "(object: ", num_object, "), ", " !")
        print("  [MURPHYNet] :: use convolutional backbone: ", backbone, " !")
        self.type = type
        if 'fuse' in self.type:
            print("  [MURPHYNet] :: use CNN + GNN fusion !")
        print("  [MURPHYNet] :: [module] backbone_output_dim: ", self.img_fdim, " !")
        print("  [MURPHYNet] :: [module] relation RGCN hidden dim: ", self.gcn_dim, " !")
        print("  [MURPHYNet] :: [module] relation RGCN output dim: ", self.out_dim, " !")
        print("  [MURPHYNet] :: [module] realtion HRCA hidden dim: ", self.hrca_channels, " !")

        # encoder part
        self.backbone_name = backbone
        if 'lstm' in self.backbone_name:
            print("  [MURPHYNet] :: use CNN+LSTM model as backbone !, backbone name is ", self.backbone_name)
        else:
            print("  [MURPHYNet] :: use CNN model as backbone !, backbone name is ", self.backbone_name)

        if 'resnet50' in self.backbone_name:
            resnet = torchvision.models.resnet50(pretrained=True)
            self.backbone = torch.nn.Sequential()
            self.backbone.add_module("conv1", resnet.conv1)
            self.backbone.add_module("bn1", resnet.bn1)
            self.backbone.add_module("relu", resnet.relu)
            self.backbone.add_module("maxpool", resnet.maxpool)
            self.backbone.add_module("layer1", resnet.layer1)
            self.backbone.add_module("layer2", resnet.layer2)
            self.backbone.add_module("layer3", resnet.layer3)
            self.backbone.add_module("layer4", resnet.layer4)
            self.backbone.add_module("avgpool", resnet.avgpool)
            self.input_dim = 2048
        else:
            raise NotImplementedError

        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        if self.backbone_name == 'resnet50lstm':
            self.lstm = nn.LSTM(self.input_dim, self.img_fdim, batch_first=True, dropout=0)
            init.xavier_normal_(self.lstm.all_weights[0][0])
            init.xavier_normal_(self.lstm.all_weights[0][1])
        else:
            self.projection = nn.Linear(self.input_dim, self.img_fdim)
            init.xavier_uniform_(self.projection.weight)

        # config downstream classifiers
        self.idle_classifier = None
        self.step_classifier = None
        self.task_classifier = None
        self.triplet_classifier = None
        self.instrument_classifier = None
        self.action_classifier = None
        self.object_classifier = None
        for label_name, label_class_size in relations.items():
            if label_name == 'step':
                self.step_classifier = nn.Linear(self.img_fdim + self.gcn_dim, label_class_size)
                # init.kaiming_uniform_(self.step_classifier.weight)
            if label_name == 'task':
                self.task_classifier = nn.Linear(self.img_fdim + self.gcn_dim, label_class_size)
                # init.kaiming_uniform_(self.task_classifier.weight)
            if label_name == 'triplet':
                print("  [MURPHYNet] :: use HRCA IAO component embedding: >> ", 'hrca' in self.type,
                      " << !!!")
                self.instrument_fea = nn.Linear(self.img_fdim + self.gcn_dim, self.out_dim)
                self.action_fea = nn.Linear(self.img_fdim + self.gcn_dim, self.out_dim)
                self.object_fea = nn.Linear(self.img_fdim + self.gcn_dim, self.out_dim)

                init.kaiming_uniform_(self.instrument_fea.weight)
                init.kaiming_uniform_(self.action_fea.weight)
                init.kaiming_uniform_(self.object_fea.weight)

                self.triplet_classifier = nn.Linear(self.out_dim * 3, label_class_size)
                # init.kaiming_uniform_(self.triplet_classifier.weight)

            if label_name == 'instrument':
                self.instrument_classifier = nn.Linear(self.out_dim, label_class_size)
                init.kaiming_uniform_(self.instrument_classifier.weight)
            if label_name == 'action':
                self.action_classifier = nn.Linear(self.out_dim, label_class_size)
                init.kaiming_uniform_(self.action_classifier.weight)
            if label_name == 'object':
                self.object_classifier = nn.Linear(self.out_dim, label_class_size)
                init.kaiming_uniform_(self.object_classifier.weight)

        # config activation function
        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'lrelu':
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError
        print("  [MURPHYNet] :: use activation function: >> ", activation, " << !!!")

        self.self_weight = self_weight
        self.num_triplets = num_triplets
        self.num_step = num_step
        self.num_task = num_task

        self.num_instrument = num_instrument
        self.num_action = num_action
        self.num_object = num_object

        print(" [MURPHYNet] :: use layerNorm with GCONV")
        self.rgcn_lnorm1 = nn.LayerNorm(self.img_fdim)

        # each graph conv stores the relational weights in its W, b terms
        self.rgcn = nn.ModuleDict()
        # in: b x n x d  out: b x n x d'
        self.rgcn['entry'] = GraphConv(input_dim=self.img_fdim, output_dim=self.gcn_dim, bias=True, type='xavier')
        for key, val in relations.items():
            self.rgcn[str(key)] = GraphConv(input_dim=self.img_fdim, output_dim=self.gcn_dim, bias=True,
                                            type='xavier')

        self.rgcn_lnorm2 = nn.LayerNorm(self.gcn_dim)
        self.rgcn2 = nn.ModuleDict()
        # in: b x n x d  out: b x n x d'
        self.rgcn2['entry'] = GraphConv(input_dim=self.gcn_dim, output_dim=self.gcn_dim, bias=True,
                                        type='xavier')
        self.rgcn2['idle'] = GraphConv(input_dim=self.gcn_dim, output_dim=self.gcn_dim, bias=True,
                                       type='xavier')
        for key, val in relations.items():
            self.rgcn2[str(key)] = GraphConv(input_dim=self.gcn_dim, output_dim=self.gcn_dim, bias=True,
                                             type='xavier')
        print("  [MURPHYNet] :: use a 2-LAYERED R-GCN module for relation purification! ")
        print("  [MURPHYNet] :: GCN input img_fdim = {:d}, gcn embedding dim = {:d}".format(self.img_fdim,
                                                                                            self.gcn_dim))
        if 'hrca' in self.type:
            print("  [MURPHYNet] :: use HiearchicalRelationCrossAttention !")
            if self.use_rlls_rc_mode:
                print("  [MURPHYNet] :: use constraints for RLLS-w-RC mode !")
            else:
                print("  [MURPHYNet] :: use constraints for RLLS mode !")
            self.hrca = HRCAModule(use_rlls_rc=self.use_rlls_rc_mode,
                                   step_channel=1,
                                   task_channel=1,
                                   triplet_channel=1,
                                   hidden_dim=self.hrca_channels, class_types=relations,
                                   use_prior_knowledge=self.use_prior_knowledge)

    def load_pretrained_backbone(self, path: str):
        print("   [backbone] :: (warnings) load pretrained backbone parameters !")
        backbone_param = torch.load(path)
        unexcepted, missing = self.backbone.load_state_dict(backbone_param['state_dict'], strict=False)
        print("    (unexpected param) =  ", len(unexcepted), ", (missing) = ", len(missing))
        new_state_dict = {}
        for k, v in backbone_param['state_dict'].items():
            name = k.replace('backbone.backbone.', '')  # for pretrained param on RLLS
            if name in self.backbone.state_dict().keys():
                new_state_dict[name] = v
            else:
                name = k.replace('backbone.', '')  # for checkpoint on RLLS (update all)
                if name in self.backbone.state_dict().keys():
                    new_state_dict[name] = v
        unexcepted, missing = self.backbone.load_state_dict(new_state_dict, strict=False)
        print("    [polished] :: (unexpected param) =  ", len(unexcepted), ", (missing) = ", len(missing))

    def compose_batch_type_similarity(self, labels: torch.Tensor, num_class: int):
        # get batch-wise label
        oh_label = F.one_hot(labels, num_classes=num_class).to(torch.float32)
        # transpose to BxB
        return torch.mm(oh_label, oh_label.transpose(1, 0))

    def compose_cosine_batch_similarity(self, features: torch.Tensor):
        if len(features.shape) != 2:
            raise RuntimeError
        b, d = features.shape
        # compute cosine similarity between features
        sim = 9e-15 * torch.ones([b, b], dtype=features.dtype)
        for i in range(b):
            for j in range(b):
                sim[i, j] = F.cosine_similarity(features[i], features[j], dim=0)

        # normalization by rows
        sim = F.softmax(sim, dim=0)
        sim = sim.to(features.device)
        return sim

    def normalize_adj(self, adj: torch.Tensor):
        # here we can enable/disable adj mat
        if self.enable_adj_normalize is False:
            return adj
        degree = torch.sum(adj, dim=1)
        degree = torch.pow(degree, -0.5)
        degree = torch.nan_to_num(degree, nan=0.0)  # add numerical sanity mask
        degree = torch.diag(degree)
        adjhat = torch.mm(torch.mm(degree, adj), degree)
        return adjhat

    def forward(self, inputs: dict):
        # feature extraction
        if len(inputs['img'].shape) > 4:
            B, T, C, H, W = inputs['img'].shape
        else:
            B, C, H, W = inputs['img'].shape
            T = 1
        x = inputs['img'].view(-1, C, H, W)
        x = self.backbone.forward(x)

        if 'lstm' in self.backbone_name:
            x = x.view(-1, T, self.input_dim)
            self.lstm.flatten_parameters()
            y, _ = self.lstm(x)
            img_tf = y.contiguous().view(-1, self.img_fdim)
            if T != 1:
                img_tf = img_tf.view(B, T, self.img_fdim)[:, -1, :]
        else:
            img_tf = self.projection(x.squeeze(-1).squeeze(-1))

        res = {'step': None, 'task': None, 'triplet': None,
               'instrument': None, 'action': None, 'object': None}

        # generate batch adjacent matrix
        corr_fea_mat = self.compose_cosine_batch_similarity(img_tf).detach()
        corr_aff_mats = {}
        if self.disable_batch_type_sim:
            corr_aff_mats['idle'] = self.normalize_adj(corr_fea_mat)
            if self.step_classifier is not None:
                corr_aff_mats['step'] = self.normalize_adj(corr_fea_mat)
            if self.step_classifier is not None:
                corr_aff_mats['task'] = self.normalize_adj(corr_fea_mat)
            if self.step_classifier is not None:
                corr_aff_mats['triplet'] = self.normalize_adj(corr_fea_mat)
            if self.step_classifier is not None:
                corr_aff_mats['instrument'] = self.normalize_adj(corr_fea_mat)
            if self.step_classifier is not None:
                corr_aff_mats['action'] = self.normalize_adj(corr_fea_mat)
            if self.step_classifier is not None:
                corr_aff_mats['object'] = self.normalize_adj(corr_fea_mat)
        else:
            corr_aff_mats['idle'] = self.compose_batch_type_similarity(inputs['idle'], self.num_step)
            corr_aff_mats['idle'] = self.normalize_adj(corr_aff_mats['idle'] * corr_fea_mat)

            if self.step_classifier is not None:
                corr_aff_mats['step'] = self.compose_batch_type_similarity(inputs['step'], self.num_step)
                corr_aff_mats['step'] = self.normalize_adj(corr_aff_mats['step'] * corr_fea_mat)
            if self.task_classifier is not None:
                corr_aff_mats['task'] = self.compose_batch_type_similarity(inputs['task'], self.num_task)
                corr_aff_mats['task'] = self.normalize_adj(corr_aff_mats['task'] * corr_fea_mat)
            if self.triplet_classifier is not None:
                corr_aff_mats['triplet'] = self.compose_batch_type_similarity(inputs['triplet'], self.num_triplets)
                corr_aff_mats['triplet'] = self.normalize_adj(corr_aff_mats['triplet'] * corr_fea_mat)
            if self.instrument_classifier is not None:
                corr_aff_mats['instrument'] = self.compose_batch_type_similarity(inputs['instrument'], self.num_instrument)
                corr_aff_mats['instrument'] = self.normalize_adj(corr_aff_mats['instrument'] * corr_fea_mat)
            if self.action_classifier is not None:
                corr_aff_mats['action'] = self.compose_batch_type_similarity(inputs['action'], self.num_action)
                corr_aff_mats['action'] = self.normalize_adj(corr_aff_mats['action'] * corr_fea_mat)
            if self.object_classifier is not None:
                corr_aff_mats['object'] = self.compose_batch_type_similarity(inputs['object'], self.num_object)
                corr_aff_mats['object'] = self.normalize_adj(corr_aff_mats['object'] * corr_fea_mat)

        img_tf1 = self.rgcn_lnorm1(img_tf)
        h = self.rgcn['entry'](img_tf1, self.self_weight)
        for i, adj in corr_aff_mats.items():
            if i not in self.rgcn.keys():
                continue
            h = h + self.rgcn[i](img_tf1, adj)
        h = self.relu(h)

        h = self.rgcn_lnorm2(h)
        h2 = self.rgcn2['entry'](h, self.self_weight)
        for i, adj in corr_aff_mats.items():
            if i not in self.rgcn2.keys():
                continue
            h2 = h2 + self.rgcn2[i](h, adj)
        h = self.relu(h2)

        h = torch.cat([img_tf, h], dim=1)

        ys: torch.Tensor = self.step_classifier(h)
        res['step'] = ys

        yt: torch.Tensor = self.task_classifier(h)
        res['task'] = yt

        fi = self.relu(self.instrument_fea(h))
        fa = self.relu(self.action_fea(h))
        fo = self.relu(self.object_fea(h))
        hh = torch.cat((fi, fa, fo), dim=1)
        ytr: torch.Tensor = self.triplet_classifier(hh)
        res['triplet'] = ytr

        yi = self.instrument_classifier(fi)
        res['instrument'] = yi

        ya = self.action_classifier(fa)
        res['action'] = ya

        yo = self.object_classifier(fo)
        res['object'] = yo

        latent_probs = dict(step=ys, task=yt, triplet=ytr)
        hrca_atts = self.hrca(latent_probs)
        res['task'] = res['task'] + hrca_atts['task_step'].squeeze(-1)
        res['triplet'] = res['triplet'] + hrca_atts['triplet_step'].squeeze(-1)
        res['triplet'] = res['triplet'] + hrca_atts['triplet_task'].squeeze(-1)

        return res
