import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation,sample,a1,a2,a3,a4,num_layers_num,p):
        super(PrePrompt, self).__init__()
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()

        self.weighted_feature=weighted_feature(a1,a2,a3)

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.dgiprompt = DGIprompt(n_in, n_h, activation)
        self.graphcledgeprompt = GraphCLprompt(n_in, n_h, activation)
        self.lpprompt = Lpprompt(n_in, n_h)
        self.sample = torch.tensor(sample,dtype=int).cuda()
        # print("sample",self.sample)


        # self.dffprompt = weighted_prompt(2)
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()

    def forward(self, seq1, seq2, seq3, seq4, seq5, seq6, adj, aug_adj1edge, aug_adj2edge, aug_adj1mask, aug_adj2mask,
                sparse, msk, samp_bias1, samp_bias2,
                lbl):
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        
        
        logits4 = self.dgiprompt(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits5 = self.graphcledgeprompt(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits6 = self.lpprompt(self.gcn,seq1,adj,sparse)
        # print("logits1=",logits1)
        # print("logits2=",logits2)
        # print("logits3=",logits3)
        # print("logitssize=",logits3.shape)
        # print("logits1=",logits1)
        # print("logits1size=",logits1.shape)
        # print("lbl",lbl)

        # print("lblsize",lbl.shape)

        logits11 = logits1 + self.a4*logits4
        logits22 = logits2 + self.a4*logits5
        logits33 = logits3 + self.a4*logits6

        # logits11 = self.dffprompt(logits1,logits4)
        # logits22 = self.dffprompt(logits2,logits5)
        # logits33 = self.dffprompt(logits3,logits6)

        dgiloss = self.loss(logits11, lbl)
        graphcledgeloss = self.loss(logits22, lbl)
        lploss = compareloss(logits33,self.sample,temperature=1.5)
        lploss.requires_grad_(True)
        
        # print("promptdgi",self.dgi.prompt)
        # print("gcn",self.gcn.fc.weight)
        # print("promptLP",self.lp.prompt)


        # print("dgiloss",dgiloss)
        # print("graphcl",graphcledgeloss)
        # print("lploss",'{:.8f}'.format(lploss)) 

        # print("a1=", self.a1, "a2=", self.a2,"a3=",self.a3)
        # ret =self.weighted_feature(dgiloss,graphcledgeloss,lploss)
        ret = self.a1 * dgiloss + self.a2 * graphcledgeloss + self.a3 * lploss


        # ret2 = self.a1 * dgilossprompt + self.a2 * graphcledgelossprompt + self.a3 * lplosspropmt
        
        # ret = ret1 +self.a4*ret2

        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


def mygather(feature, index):
    # print("index",index)
    # print("indexsize",index.shape)  
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    # print("featuresize",feature.shape)
    # print("index",index)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):
    # print("feature",feature)
    # print("tuple",tuples)
    # feature=feature.cpu()
    # tuples = tuples.cpu()
    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling disconnected tuples")
    for i in tqdm.trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)


class weighted_feature(nn.Module):
    def __init__(self,a1,a2,a3):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,3), requires_grad=True)
        self.reset_parameters(a1,a2,a3)
    def reset_parameters(self,a1,a2,a3):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(a1)
        self.weight[0][1].data.fill_(a2)
        self.weight[0][2].data.fill_(a3)
    def forward(self, graph_embedding1,graph_embedding2,graph_embedding3):
        print("weight",self.weight)
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2 + self.weight[0][2] * graph_embedding3
        return graph_embedding
    

class downprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3,a4,ft_in, nb_classes, feature, labels):
        super(downprompt, self).__init__()
        # self.prompt1 = prompt1
        # self.prompt2 = prompt2
        # self.prompt3 = prompt3
        self.downprompt = downstreamprompt(ft_in)


        self.nb_classes = nb_classes
        self.labels = labels


        self.a4 = a4
        self.leakyrelu = nn.ELU()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        # self.prompt = prompt3
        # self.prompt = prompt3
        # self.a = nn.Parameter(torch.FloatTensor(1, 3), requires_grad=True).cuda()
        # self.reset_parameters()
        self.nodelabelprompt = weighted_prompt(3)


        self.dffprompt = weighted_feature(2)

        feature = feature.squeeze().cuda()

        self.aveemb0 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb1 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb2 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb3 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb4 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb5 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb6 = torch.FloatTensor(ft_in, ).cuda()


        self.one = torch.ones(1,ft_in).cuda()
        self.ave = torch.FloatTensor(nb_classes,ft_in).cuda()
   
    def forward(self,seq,seq1,train=0):
        # promptweight = torch.FloatTensor(1,3).cuda()
        # promptweight[0][0] = 0.3
        # promptweight[0][1] = 0.3
        # promptweight[0][2] = 0.3
        # print(self.a)

        weight = self.leakyrelu(self.nodelabelprompt(self.prompt))
        weight = self.one + weight
        # weight = torch.mm(promptweight,self.prompt)
        # # print("weight",self.a.weight)
        rawret1 = weight * seq
        rawret2 = self.downprompt(seq)
        rawret4 = seq1
        # rawret3 = rawret1 + rawret2
        rawret3 = self.dffprompt(rawret1 ,rawret2)
        # # print("a4",self.a4,"a5",self.a5)

        rawret =rawret3 +self.a4 * rawret4

        # rawret = seq
        rawret = rawret.cuda()
        # rawret = torch.stack((rawret,rawret,rawret,rawret,rawret,rawret))
        if train == 1:
            self.ave = averageemb(labels=self.labels, rawret=rawret,nb_class=self.nb_classes)
                # if self.labels[x].item() == 6:
                #     self.aveemb6 = rawret[x]
        # self.ave = weight * self.ave
        # print("rawretsize",rawret.size())

        ret = torch.FloatTensor(seq.shape[0],self.nb_classes).cuda()
        # print("avesize",self.ave.size(),"ave",self.ave)
        # print("rawret=", rawret[1])
        # print("aveemb", self.ave)
        for x in range(0,seq.shape[0]):
            ret[x][0] = torch.cosine_similarity(rawret[x], self.ave[0], dim=0)
            ret[x][1] = torch.cosine_similarity(rawret[x], self.ave[1], dim=0)
            ret[x][2] = torch.cosine_similarity(rawret[x], self.ave[2], dim=0)
            ret[x][3] = torch.cosine_similarity(rawret[x], self.ave[3], dim=0)
            ret[x][4] = torch.cosine_similarity(rawret[x], self.ave[4], dim=0)
            ret[x][5] = torch.cosine_similarity(rawret[x], self.ave[5], dim=0)
            if self.nb_classes == 7:
                ret[x][6] = torch.cosine_similarity(rawret[x], self.ave[6], dim=0)

        ret = F.softmax(ret, dim=1)

        # ret = torch.argmax(ret, dim=1)
        # print('ret=', ret)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def averageemb(labels,rawret,nb_class):
    retlabel = torch.FloatTensor(nb_class,int(rawret.shape[0]/nb_class),int(rawret.shape[1])).cuda()
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    # print("labels",labels)
    for x in range(0,rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2]= rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
        if labels[x].item() == 3:
            retlabel[3][cnt4] = rawret[x]
            cnt4 = cnt4 + 1
        if labels[x].item() == 4:
            retlabel[4][cnt5] = rawret[x]
            cnt5 = cnt5 + 1
        if labels[x].item() == 5:
            retlabel[5][cnt6] = rawret[x]
            cnt6 = cnt6 + 1
        if labels[x].item() == 6:
            retlabel[6][cnt7] = rawret[x]
            cnt7 = cnt7 + 1
    retlabel = torch.mean(retlabel,dim=1)
    return retlabel

class weighted_prompt(nn.Module):
    def __init__(self,weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0.5)
        self.weight[0][1].data.fill_(0.4)
        self.weight[0][2].data.fill_(0.3)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding=torch.mm(self.weight,graph_embedding)
        return graph_embedding
    
class weighted_feature(nn.Module):
    def __init__(self,weightednum):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)
    def forward(self, graph_embedding1,graph_embedding2):
        # print("weight",self.weight)
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding=self.weight * graph_embedding
        return graph_embedding

class featureprompt(nn.Module):
    def __init__(self,prompt1,prompt2,prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)
    def forward(self,feature):
        # print("prompt",self.weightprompt.weight)
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=None, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        # print("act",type(self.act))
        # print("fc",self.fc.weight)
        # print("fc",self.fc.weight.shape)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, input, sparse=True):
        # print("input",input)
        seq = input[0].cuda()
        adj = input[1].cuda()
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        # print("out",out)
        # print("act",self.act)

        return self.act(out)
        # return out.type(torch.float)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layers_num,dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)

        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self, seq, adj,sparse,LP=False):
        graph_output = torch.squeeze(seq,dim=0)
        graph_len = adj
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        xs = []
        for i in range(self.num_layers_num):
            # print("i",i)
            input=(graph_output,adj)
            graph_output = self.convs[i](input)
            # print("graphout1",graph_output)
            # print("graphout1",graph_output.shape)
            if LP:
                # print("graphout1",graph_output.shape)
                graph_output = self.bns[i](graph_output)
                # print("graphout2",graph_output.shape)
                graph_output = self.dropout(graph_output)
            # print("graphout2",graph_output)
            # print("graphout2",graph_output.shape)
            xs.append(graph_output)
            # print("Xs",xs)
        # xpool= []
        # for x in xs:
        #     graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
        #     graph_embedding = torch.sum(graph_embedding, dim=1)
        #     xpool.append(graph_embedding)
        # x = torch.cat(xpool, -1).unsqueeze(dim=0)
        return graph_output.unsqueeze(dim=0)
    
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = gcn(seq1, adj, sparse)


        # print("h_1",h_1.shape)

        h_3 = h_1 * self.prompt

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = gcn(seq2, adj, sparse)

        h_4 = h_2 * self.prompt

        ret = self.disc(c, h_3, h_4
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

class DGIprompt(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGIprompt, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        
        
        seq1 = seq1 * self.prompt
        h_1 = gcn(seq1, adj, sparse)


        # print("h_1",h_1.shape)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        seq2 = seq2 * self.prompt
        h_2 = gcn(seq2, adj, sparse)


        ret = self.disc(c, h_1, h_2
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

class GraphCL(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GraphCL, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1,n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type):

        h_0 = gcn(seq1, adj, sparse)

        h_00 = h_0 * self.prompt
        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse)
            h_3 = gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse)
            h_3 = gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse)
            h_3 = gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        h_11 = h_1 * self.prompt
        h_33 = h_3 * self.prompt

        c_1 = self.read(h_11, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_33, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse)

        h_22 = h_2 * self.prompt

        ret1 = self.disc(c_1, h_00, h_22, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_00, h_22, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

class GraphCLprompt(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GraphCLprompt, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1,n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type):

        seq1 = seq1 * self.prompt
        seq2 = seq2 * self.prompt
        seq3 = seq3 * self.prompt
        seq4 = seq4 * self.prompt

        h_0 = gcn(seq1, adj, sparse)

        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse)
            h_3 = gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse)
            h_3 = gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse)
            h_3 = gcn(seq4, aug_adj2, sparse)

        else:
            assert False


        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse)


        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)
        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

class LogReg(nn.Module):

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,sparse):
        h_1 = gcn(seq,adj,sparse,True)
        # ret = h_1
        ret = h_1 * self.prompt
        # ret = h_1 
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

class Lpprompt(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lpprompt, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,sparse):
        
        seq = seq * self.prompt
        h_1 = gcn(seq,adj,sparse,True)
        ret = h_1
        # ret = h_1 * self.prompt
        # ret = h_1 
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)