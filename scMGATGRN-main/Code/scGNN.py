import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
import math


class scMGATGRN(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,output_dim,num_head1,num_head2,
                 alpha,device,type,reduction,num_nodes):
        super(scMGATGRN, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.num_nodes=num_nodes

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim


        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,num_nodes,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)

        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,num_nodes,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        self.tf_linear1 = nn.Linear(hidden2_dim,output_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,output_dim)



        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)




    def encode(self,x,adj):
        if self.reduction =='concate':

            x = torch.cat([att(x, adj,1)for att in self.ConvLayer1], dim=1)
            x = F.elu(x)


        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj,1) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)

        else:
            raise TypeError


        out = torch.mean(torch.stack([att(x, adj,2) for att in self.ConvLayer2]),dim=0)
        out=F.elu(out)

        return out


    def decode(self,tf_embed,target_embed):

        if self.type =='dot':

            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)


            return prob

        elif self.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)

            return prob

        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)

            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))


    def forward(self,x,adj,train_sample):

        embed= self.encode(x,adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.elu(tf_embed)
        tf_embed = F.dropout(tf_embed,p=0.01)
        target_embed = self.target_linear1(embed)
        target_embed = F.elu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        self.tf_ouput = tf_embed
        self.target_output = target_embed


        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output



class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,nums,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.num=nums


        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))
        self.a2 = nn.Parameter(torch.zeros(size=( 2*self.output_dim, 1)))
        self.W = nn.Parameter(torch.FloatTensor(self.output_dim, self.output_dim))
        self.Q = nn.Parameter(torch.FloatTensor(self.output_dim, 1))



        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight2.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.Q.data, gain=1.414)


    def _prepare_attentional_mechanism_input(self, x,x2):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        Wh3 = torch.matmul(x2, self.a2[:self.output_dim, :])
        Wh4 = torch.matmul(x2, self.a2[self.output_dim:, :])
        e = torch.exp(-torch.square(Wh1 - Wh2.T)/1e-0)
        e2=F.leaky_relu(Wh3 + Wh4.T,negative_slope=self.alpha)

        return e,e2



    def forward(self,x,adj,layer):

        h=torch.matmul(x,self.weight)
        h2=torch.matmul(x, self.weight2)
        e,e2 = self._prepare_attentional_mechanism_input(h,h2)

        zero_vec = -9e15 * torch.ones_like(e)
        N = torch.ones_like(adj.to_dense())

        A = adj.to_dense()
        I=torch.eye(A.shape[0])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        I=I.to(device)
        A=A+I

        B = torch.matmul(A, N) + torch.matmul(N, A) - torch.matmul(A, A)
        C = torch.div(torch.matmul(A, A), B)
        attention = torch.where(adj.to_dense()>0, e, zero_vec)
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)
        jaccard_C =torch.where(A > 0,N,C)
        attention2 = torch.where(  jaccard_C > 0.2, e2, zero_vec)


        attention = F.softmax(attention, dim=1)
        attention2 = F.softmax(attention2, dim=1)
        
        attention = F.dropout(attention, training=self.training)
        attention2 = F.dropout(attention2, training=self.training)
        h_pass = torch.matmul(attention, h)
        h_pass2=torch.matmul(attention2,h2)
        output_data = h_pass
        output_data2=h_pass2

        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data2 = F.leaky_relu(output_data2,negative_slope=self.alpha)


        output_data = F.normalize(output_data,p=2,dim=1)
        output_data2 = F.normalize(output_data2, p=2, dim=1)
        w1 = torch.mean(torch.matmul(torch.tanh(torch.matmul( output_data, self.W)), self.Q), dim=0)
        w2 = torch.mean(torch.matmul(torch.tanh(torch.matmul( output_data2, self.W)), self.Q), dim=0)
        w = torch.cat([w1, w2], dim=0)
        w = F.softmax(w, dim=0)
        output_data = w[0] * output_data + w[1] * output_data2




        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data












