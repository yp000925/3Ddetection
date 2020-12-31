import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
from collections import OrderedDict

print('Loading model..')
net = RetinaNet()
checkpoint = torch.load('./checkpoint/checkpoint.pt',map_location=torch.device('cpu'))
if torch.cuda.is_available():
    net.load_state_dict(checkpoint['net'])
else:
    cuda_state_dict = checkpoint['net']
    new_state_dict = OrderedDict([(key.split('module.')[-1],cuda_state_dict[key]) for key in cuda_state_dict])
    net.load_state_dict(new_state_dict)
net.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('./data/hologram/0.jpg')

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, requires_grad=False)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (1024,1024))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()