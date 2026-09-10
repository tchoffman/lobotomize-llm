"""Pre-train a small DCGAN on MNIST so the talk's latent-walk demo loads instantly."""
import torch, torch.nn as nn, warnings; warnings.filterwarnings("ignore")
from torchvision import datasets, transforms

DEV="mps" if torch.backends.mps.is_available() else "cpu"
Z=32
class G(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(
            nn.ConvTranspose2d(Z,256,7,1,0,bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,1,4,2,1,bias=False), nn.Tanh())
    def forward(s,z): return s.net(z.view(-1,Z,1,1))
class D(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(
            nn.Conv2d(1,64,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,1,7,1,0))
    def forward(s,x): return s.net(x).view(-1)

tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
dl=torch.utils.data.DataLoader(datasets.MNIST("data",train=True,download=True,transform=tf),
                               batch_size=128,shuffle=True,drop_last=True)
g,dd=G().to(DEV),D().to(DEV)
og=torch.optim.Adam(g.parameters(),2e-4,betas=(0.5,0.999))
od=torch.optim.Adam(dd.parameters(),2e-4,betas=(0.5,0.999))
bce=nn.BCEWithLogitsLoss()
for ep in range(20):
    for x,_ in dl:
        x=x.to(DEV); b=x.size(0)
        z=torch.randn(b,Z,device=DEV); fake=g(z)
        od.zero_grad()
        ld=bce(dd(x),torch.ones(b,device=DEV))+bce(dd(fake.detach()),torch.zeros(b,device=DEV))
        ld.backward(); od.step()
        og.zero_grad()
        lg=bce(dd(fake),torch.ones(b,device=DEV)); lg.backward(); og.step()
    print(f"epoch {ep+1}/20  D={ld.item():.3f} G={lg.item():.3f}",flush=True)
torch.save({"state":g.state_dict(),"z_dim":Z},"models/mnist_dcgan_g.pt")
print("saved models/mnist_dcgan_g.pt")
