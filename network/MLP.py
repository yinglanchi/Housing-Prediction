import torch.nn as nn
import torch

from sklearn.metrics import r2_score


class BaseModel(nn.Module):
    def __init__(self, input_size, cfg):
        super(BaseModel, self).__init__()
        
        self.device = cfg["device"]
        self.net = MLP(input_size, cfg)
        self.cfg = cfg
        self.loss_weights = cfg["loss_weights"]

        self.l2_loss = nn.MSELoss()
        
    def summarize_losses(self, loss_dict):
        total_loss = 0
        for key, item in self.loss_weights.items():
            if key in loss_dict:
                total_loss += loss_dict[key] * item
        loss_dict['total_loss'] = total_loss
        self.loss_dict = loss_dict
        
    def set_data(self, data):
        self.feed_dict = {}
        for key, item in data.items():
            if key in [""]:
                continue
            item = item.float().to(self.device)
            self.feed_dict[key] = item
            
    def compute_loss(self):
        feed_dict = self.feed_dict
        pred_dict = self.pred_dict
        loss_dict = {}
        
        pred_prices = pred_dict["prices"]
        gt_prices = feed_dict["prices"]
        
        loss_dict["L2_prices"] = self.l2_loss(pred_prices, gt_prices)
        loss_dict["L1_reg"] = self.l1_reg()
        loss_dict["L2_reg"] = self.l2_reg()
        
        loss_dict["prices_r2_score"] = r2_score(gt_prices.cpu().detach().numpy(), pred_prices.cpu().detach().numpy())
        
        self.summarize_losses(loss_dict)
        
    def l1_reg(self):
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
        return l1_reg

    def l2_reg(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return l2_reg
    
    def test(self):
        self.loss_dict = {}
        with torch.no_grad():
            self.pred_dict = self.net(self.feed_dict)
            self.compute_loss()
            
    def update(self):
        self.pred_dict = self.net(self.feed_dict)
        self.compute_loss()
        self.loss_dict['total_loss'].backward()
            

class MLP(nn.Module):
    def __init__(self, input_size, cfg):
        super(MLP, self).__init__()
        
        self.cfg = cfg
        layers = cfg["mlp_layers"]
        
        self.MLP = nn.Sequential()
        
        self.MLP.add_module(f"Linear0", nn.Linear(input_size, layers[0]))
        
        for i in range(1, len(layers)):
            self.MLP.add_module(f"ReLU{i - 1}", nn.ReLU())
            self.MLP.add_module(f"Linear{i}", nn.Linear(layers[i - 1], layers[i]))
            
    def forward(self, x):
        return {"prices": self.MLP(x["features"])}
