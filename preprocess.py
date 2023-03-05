# x = [{'width': 1024.0, 'height': 768.0}, {'width': 1024.0, 'height': 536.0}, {'width': 932.0, 'height': 740.0}, {'width': 748.0, 'height': 508.0}, {'width': 784.0, 'height': 584.0}, {'width': 792.0, 'height': 538.0}, {'width': 0, 'height': 0}]
# print(len(x))
# max_size = []
# for label in x:
#     max_size.append([label['width'] / 1024, label['height'] / 768])

# print(max_size)

# import torch
# n = torch.tensor(max_size) * torch.tensor([1024, 768])
# m = torch.round(n).type(torch.int64)
# print(m)
import torch
# t4d = torch.empty(3, 2, 2)
# padding = (1, 1, 1, 0)
# print(t4d.shape)
# t4d = torch.nn.functional.pad(t4d[:], padding, mode='constant', value=-1)
# print(t4d)

# x = [{'width': 1024.0, 'height': 768.0}, {'width': 300.79133858267716, 'height': 357.6122047244094}, {'width': 506.3996062992126, 'height': 420.6692913385827}, {'width': 360.4724409448819, 'height': 250.75984251968504}, {'width': 342.13385826771656, 'height': 116.46358267716535}, {'width': 359.5698818897638, 'height': 112.49606299212599}, {'width': 0.0, 'height': 0.0}]
# print(len(x))
# max_size = []
# for label in x:
#     max_size.append([label['width'] / 1024, label['height'] / 768])

# print(max_size)

# x = [[0.3520238681102362, 0.3265102116141732], [0.4945308655265748, 0.5477464730971129], [0.29374154158464566, 0.46564089156824146], [0.33411509596456695, 0.15164528994422571], [0.35114246278297245, 0.14647924868766404]]

concat_tensor = torch.empty(0, 3, 4)

# Concatenate a tensor of size 2x3x4
tensor1 = torch.randn(2, 3, 4)
concat_tensor = torch.cat([concat_tensor, tensor1], dim=0)

# Concatenate another tensor of size 1x3x4
tensor2 = torch.randn(1, 3, 4)
concat_tensor = torch.cat([concat_tensor, tensor2], dim=0)

# Final concatenated tensor of size 3x3x4
print(concat_tensor)
print(concat_tensor.size())
print(concat_tensor.shape)