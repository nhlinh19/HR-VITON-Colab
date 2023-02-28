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

def get_background_value(image_tensor):
    # Take the mean of each color channel
    background_value = torch.mean(image_tensor, dim=(0,1,2))
    
    return background_value

# Create a 3x4x4 image tensor with a uniform background value of 10
image_tensor = torch.ones((3, 4, 4)) * 10

# Set some pixels to different values to simulate an image
image_tensor[:, 1:3, 1:3] = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

# Get the background value of the image
background_value = get_background_value(image_tensor)

# Print the image tensor and the background value
print("Image tensor:\n", image_tensor)
print("Background value:\n", background_value)
