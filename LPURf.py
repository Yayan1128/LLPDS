import torch
import torch.nn as nn

from torch.nn import functional as F
from module_list import label_onehot
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class Learningnet(nn.Module):
    def __init__(self,input_feature_dim, feature_dim):
        super(Learningnet, self).__init__()
       
        assert input_feature_dim == feature_dim,"Should match when residual mode is on ({} != {})".format(input_feature_dim,feature_dim)


        self.learningfeat = nn.Sequential(  
            nn.Conv2d(input_feature_dim, feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(feature_dim)
        )
        self.relu = nn.ReLU(inplace=True)


        initialize_weights(self)

    def forward(self,x):
        output = x + self.learningfeat(x)
        output = self.relu(output)

        return output



class LPUR(nn.Module):
    def __init__(self, device,token_size, input_feature_dim, feature_dim, momentum, temperature, gumbel_rectification,co):
        super(LPUR, self).__init__()
      
        self.token_size = token_size  
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.initial_momentum = momentum
        self.temperature = temperature
        self.co=co
        self.output = nn.Sequential(  
            nn.Conv2d(feature_dim*2, input_feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(input_feature_dim),
            nn.ReLU(inplace=True),
        )
    
        self.learningnet = Learningnet(input_feature_dim, feature_dim)


        self.tok_cls = torch.tensor([x for x in range(self.token_size)]).to(device)
        
        self.clsfier = nn.Linear(in_features=self.feature_dim, out_features=self.token_size, bias=True)


        self.celoss = nn.CrossEntropyLoss(ignore_index=255)
        self.gumbel_rectification = gumbel_rectification

        self.learningTF = lambda x: x.clone()

        self.t_items = F.normalize(torch.rand((token_size, feature_dim), dtype=torch.float),
                                   dim=1).cuda()  # Initialize the memory items
        initialize_weights(self)



    def cue(self, query,tokens):
        bs, h, w, d = query.size()
        r, d = tokens.size()

        cue = torch.matmul(query, torch.t(tokens))  # b X h X w X m

        cue = cue.view(bs * h * w, r)  # (b X h X w) X m

        if self.gumbel_rectification:
         
            cue_tokens = F.gumbel_softmax(cue, dim=1) 
        else:
        
            cue_tokens = F.softmax(cue, dim=1)  

        return cue_tokens

    def forward(self, query,mask=None,token_rectification=False,token_learning=False,learning_detach = True):

       
        # rectification
        if token_rectification:
            updated_query = self.rectification(query)
            return updated_query
        # learning
        if token_learning:
            learningloss = self.learning(query, mask,learning_detach)
         
            return  learningloss

    
    def learning(self, input, mask,learning_detach = True):

        tempmask = mask.clone().detach()#-1
     
        mask_all= (mask.clone().detach().unsqueeze(1) >= 0).float()
       
        query = input.clone()
        if not learning_detach:
            query = self.learningTF(query) 
     
        momentum = self.momentum
       
        query = self.learningnet(query)
       
        query = F.normalize(query, dim=1)
        
        batch_size, dims, h, w = query.size()

        
        query = query.view(batch_size, dims, -1)
        
        tempmask=label_onehot(tempmask, self.token_size)
        tempmask = tempmask * mask_all#2, 2,360, 640]
     
        tempmask = F.interpolate(tempmask.contiguous().type(torch.float32), [h,w], mode='bilinear', align_corners=True).permute(0,2,3,1).contiguous()

        tempmask = tempmask.view(batch_size, -1, self.token_size)
        denominator = tempmask.sum(1).unsqueeze(dim=1)
        nominator = torch.matmul(query, tempmask)

      

        cur_feat_norm_t = query.transpose(2,1) # b,n,c
        sim = torch.matmul(cur_feat_norm_t, nominator) * 2.0 # b,n,2
        sim = sim.softmax(1)
        weight=sim*tempmask
        proto_local=torch.matmul(query, weight)
        nominator_local = torch.t(proto_local.sum(0))  # batchwise sum,2,n
    
        nominator = torch.t(nominator.sum(0))  # batchwise sum
        denominator = denominator.sum(0)  # batchwise sum
        denominator = denominator.squeeze()

        updated_tokens = self.t_items.clone().detach() 
        for slot in range(self.token_size):
            if denominator[slot] != 0: 
                updated_tokens[slot] = momentum * self.t_items[slot] + (
                            (1 - momentum) * ((nominator[slot] / denominator[slot]+self.co*nominator_local[slot] / denominator[slot])/(1+self.co)))  # memory momentum update

        updated_tokens = F.normalize(updated_tokens, dim=1)  # normalize

      
        div_loss = self.diversityloss(updated_tokens)

        cls_loss = self.classification_loss(updated_tokens)

        learning_loss = [div_loss, cls_loss]

        if learning_detach: # 
            self.t_items = updated_tokens.detach()
            return learning_loss
        else:
            self.t_items = updated_tokens
            return learning_loss

    def classification_loss(self, tok):

        score = self.clsfier(tok)
        return self.celoss(score, self.tok_cls)

    def diversityloss(self, tok):

        cos_sim = torch.matmul(tok,torch.t(tok))
        margin = 0 
        cos_sim_pos = cos_sim-margin
        cos_sim_pos[cos_sim_pos<0]=0
        loss = (torch.sum(cos_sim_pos)-torch.trace(cos_sim_pos))/(self.token_size*(self.token_size-1))
        return loss


    def rectification(self, query):

        query0 = F.normalize(query.clone(), dim=1)
        query = query0.permute(0, 2, 3, 1).contiguous()  # b X h X w X d
        batch_size, h, w, dims = query.size()  # b X h X w X d

 

        softmax_cue_tokens = self.cue(query, self.t_items)
        query_reshape = query.view(batch_size * h * w, dims)
        concat_tokens = torch.matmul(softmax_cue_tokens, self.t_items)  # (b X h X w) X d
    
        updated_query = torch.cat((query_reshape, concat_tokens), dim=1)  # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2).contiguous()

        updated_query = self.output(updated_query)

        return updated_query


