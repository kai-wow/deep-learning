import torch


# x = torch.randn(4,5)
# y = torch.randn(4,5)
# print(x)
# print(y)

# # 1. max of entire tensor (torch.max(input) → Tensor)
# m = torch.max(x)
# print(m)
# # 2. max along a dimension (torch.max(input, dim, keepdim=False, *, out=None) → (Tensor, LongTensor))
# m, idx = torch.max(x,dim=0,keepdim=True)
# print(m)
# print(idx)
# # 2-5
# p = (m,idx)
# torch.max(x,0,False,out=p)
# print(p[0])
# print(p[1])

# t = torch.max(x,y)
# print(t)

import torch
import torch.utils.data 
class ExampleDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.data = "abcdefghijklmnopqrstuvwxyz"
  
  def __getitem__(self,idx): # if the index is idx, what will be the data?
    return self.data[idx]
  
  def __len__(self): # What is the length of the dataset
    return len(self.data)

dataset1 = ExampleDataset() # create the dataset
dataloader = torch.utils.data.DataLoader(dataset=dataset1, shuffle=True, batch_size=2) # shuffle 洗牌，是否打乱顺序
for datapoint in dataloader:
    print(datapoint)