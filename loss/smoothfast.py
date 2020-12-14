import torch


class PatchMemory(object):
   # error_integral={}
    #error_deg={}
    #tmp={}

    def __init__(self, momentum=0.1, num=1):

        self.name = []
        self.AGENT = []
        self.error = [] ##########new add
        self.last = [] ##########new add
        self.error_integral={}
        self.error_deg={}
        self.agent={}
        self.alpha = 0.2
        self.num = num
        self.integral = 0.5 ############new add
        self.deg = 0.1############new add

    def get_soft_label(self, path, feat_list,epoch=1):

        feat = torch.stack(feat_list, dim=0)
        feat = feat[:, ::self.num, :]
        position = []
        soft_label = []
      #  self.momentum = 0.5-(0.5-0.1)*(epoch+1)*0.01  ##############new add learning rate
        #print(self.momentum)


        # update the agent
        '''
        error_integral = ExemplarMemory.error_integral
        error_deg = ExemplarMemory.error_deg
        agent = ExemplarMemory.tmp
        '''
        for j,p in enumerate(path):

            current_soft_feat = feat[:, j, :].detach()
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()
            key  = p
            if key not in self.name:
                
                self.error_integral[key.item()] = 0
                self.error_deg[key.item()] = 0
                self.agent[key.item()]= current_soft_feat

                self.name.append(key)
                '''
                self.agent.append(current_soft_feat)
                self.error.append(0)###################new add
                self.last.append(0)###################new add
                '''
                ind = self.name.index(key)
                position.append(ind)
                


            else:
                              
                error_add = self.error_integral[key.item()]
                error_last = self.error_deg[key.item()]
                #tmp = agent[key.item()]
       


           # error_add = self.error.pop(ind)##############################new add error control
                error_add = error_add+current_soft_feat-self.agent[key.item()]##############################new add error control
                self.agent[key.item()]= self.agent[key.item()]*(1-self.alpha ) + self.alpha *current_soft_feat +self.integral*(error_add)+self.deg*(current_soft_feat-self.agent[key.item()]-error_last)
                self.error_integral[key.item()] = error_add
                self.error_deg[key.item()] = current_soft_feat-self.agent[key.item()]
                soft_label.append(self.agent[key.item()])
                ind = self.name.index(key)
                position.append(ind)
                
           ''' 
                ind = self.name.index(key)
                tmp = self.agent.pop(ind)
                error_last = self.last.pop(ind)###################new add

                

                error_add = self.error.pop(ind)##############################new add error control
                error_add = error_add+current_soft_feat-tmp##############################new add error control
                
               # print(error_add)##############################new add error control


                tmp = tmp*(1-self.momentum) + self.momentum*current_soft_feat +self.integral*(error_add)+self.deg*(current_soft_feat-tmp-error_last) ############new add
                self.agent.insert(ind, tmp)
                self.error.insert(ind, error_add)######new add
                self.last.insert(ind, current_soft_feat-tmp)######new add
                position.append(ind)

        if len(position) != 0:
            position = torch.tensor(position).cuda()

        agent = torch.stack(self.agent, dim=1).cuda()
        return agent, position
        
'''

        soft_label = torch.stack(soft_label, dim=0).cuda()
        if len(position) != 0:
            position = torch.tensor(position).cuda()
        return soft_label, position


class SmoothingForImage(object):
    def __init__(self, momentum=0.1, num=1):

        self.map = dict()
        self.momentum = momentum
        self.num = num


    def get_soft_label(self, path, feature):

        feature = torch.cat(feature, dim=1)
        soft_label = []

        for j,p in enumerate(path):

            current_soft_feat = feature[j*self.num:(j+1)*self.num, :].detach().mean(dim=0)
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()

            key  = p
            if key not in self.map:
                self.map.setdefault(key, current_soft_feat)
                soft_label.append(self.map[key])
            else:
                self.map[key] = self.map[key]*(1-self.momentum) + self.momentum*current_soft_feat
                soft_label.append(self.map[key])
        soft_label = torch.stack(soft_label, dim=0).cuda()
        return soft_label



