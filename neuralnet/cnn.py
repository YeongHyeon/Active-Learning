import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class Neuralnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.who_am_i = "CNN"

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.k_size = kwargs['k_size']
        self.filters = kwargs['filters']

        # self.center = 1
        self.register_buffer("center", torch.zeros(1, self.num_class))
        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.ngpu = kwargs['ngpu']
        self.device = kwargs['device']

        self.params = {}
        self.params['warmup'] = WarmupConv(self.dim_c, self.filters[0], self.k_size, stride=1, name="warmup").to(self.device)

        for idx_filter, _ in enumerate(self.filters[:-1]):
            self.params['clf_%d_1' %(idx_filter+1)] = ConvBlock(self.filters[idx_filter], self.filters[idx_filter]*2, \
                self.k_size, stride=1, name="conv%d_1" %(idx_filter+1)).to(self.device)
            self.params['clf_%d_2' %(idx_filter+1)] = ConvBlock(self.filters[idx_filter]*2, self.filters[idx_filter]*2, \
                self.k_size, stride=2, name="conv%d_2" %(idx_filter+1)).to(self.device)
            self.params['act_%d' %(idx_filter+1)] = Knowledge(self.filters[idx_filter]*2, 128, \
                name="knowledge%d" %(idx_filter+1)).to(self.device)

        self.params['clf_fin'] = Classifier(self.filters[-1], self.num_class, name='classifier').to(self.device)
        self.params['act_fin'] = Classifier(128*len(self.filters[:-1]), 1, name='classifier').to(self.device)

        self.list_pkey = self.params.keys()
        self.total = []
        for idx_pkey, name_pkey in enumerate(self.list_pkey):
            print(idx_pkey, name_pkey)
            self.total.append(self.params[name_pkey])
        self.modules = nn.ModuleList(self.total)


    def forward(self, x, verbose=False):

        feat_act = []
        x_b1 = x
        for idx_pkey, name_pkey in enumerate(self.list_pkey):
            if(verbose): print(name_pkey, x.shape)
            if(name_pkey.startswith('act')):
                x_b2 = x_b1
                if(name_pkey.endswith('fin')):
                    feat_act = torch.cat(feat_act, axis=1)
                    if(verbose): print(' feat:', feat_act.shape)
                    x_b2 = self.params[name_pkey](feat_act)
                else:
                    try: x_b2 = torch.mean(x_b2, axis=(2, 3))
                    except: pass
                    x_b2 = self.params[name_pkey](x_b2)
                    feat_act.append(x_b2)
                if(verbose): print(' out:', x_b2.shape)
            else:
                if(name_pkey.endswith('fin')):
                    x_b1 = self.params[name_pkey](torch.mean(x_b1, axis=(2, 3)))
                else:
                    x_b1 = self.params[name_pkey](x_b1)
                if(verbose): print(' out:', x_b1.shape)

        y_hat = x_b1
        l_hat = torch.mean(x_b2, axis=1)

        return {'y_hat':y_hat, 'l_hat':l_hat}

    def loss_target_pred(self, y, y_hat):

        loss_ce = nn.CrossEntropyLoss(reduce=False)
        opt_b = loss_ce(y_hat, target=y)
        opt = torch.mean(opt_b)

        return {'opt_y': opt, 'opt_y_b': opt_b}

    def loss_loss_pred(self, l, l_hat, pos_margin=1):
        
        l_i, l_j = l.chunk(2, dim=0)
        l_hat_i, l_hat_j = l_hat.chunk(2, dim=0)

        scaling = torch.where(l_i > l_j, torch.ones_like(l_i), -torch.ones_like(l_i))
        opt_b = torch.maximum(torch.zeros_like(l_i), -scaling*(l_hat_i - l_hat_j) + pos_margin)
        opt = torch.mean(opt_b)

        return {'opt_l': opt}

    def update_center(self, y_hat):

        batch_center = torch.sum(y_hat, dim=0, keepdim=True)
        batch_center = batch_center / y_hat.size(0)

        self.center = self.center * 0.9 + batch_center * (1 - 0.9)

class WarmupConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, name=""):
        super().__init__()
        self.warmup = nn.Sequential()
        self.warmup.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
        self.warmup.add_module("%s_act" %(name), nn.ReLU())

    def forward(self, x):

        out = self.warmup(x)
        return out

class Knowledge(nn.Module):

    def __init__(self, in_channels, out_channels, name=""):
        super().__init__()
        self.clf = nn.Sequential()
        self.clf.add_module("%s_lin0" %(name), nn.Linear(in_channels, out_channels))
        self.clf.add_module("%s_act0" %(name), nn.ReLU())

    def forward(self, x):

        return self.clf(x)

class Classifier(nn.Module):

    def __init__(self, in_channels, out_channels, name=""):
        super().__init__()
        self.clf = nn.Sequential()
        self.clf.add_module("%s_lin0" %(name), nn.Linear(in_channels, int(in_channels*2)))
        self.clf.add_module("%s_act0" %(name), nn.ReLU())
        self.clf.add_module("%s_lin1" %(name), nn.Linear(int(in_channels*2), out_channels))

    def forward(self, x):

        return self.clf(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True, name=""):
        super().__init__()

        self.out_channels = out_channels

        self.conv_ = nn.Sequential()
        self.conv_.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, \
            kernel_size, stride, padding=kernel_size//2)) 
        self.conv_.add_module("%s_act" %(name), nn.ReLU())

    def forward(self, x):
        return self.conv_(x)
