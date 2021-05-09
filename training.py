import Levenshtein
import torch
import matplotlib.pyplot as plt
from session import Session
from tqdm.notebook import tqdm
from utils import *

# tf and af scheduling function
def thred_sched(e, thred, delta, init=0.9, minval=0.6):
    val = init 
    if e > thred:
        val -= delta * (e - thred)
    return max(val, minval)


#  Session scheduler that takes no argument
class LRSched_0arg:
    def __init__(self, factory):
        self.factory = factory

    def __call__(self, session):
        self.sched = self.factory(session)
        return self

    def step(self, epoch, acc, loss):
        self.sched.step()


#  LAS training session
class LASSession(Session):
    def init(self, tf_sched=None, af_sched=None, **kwargs):
        self.train_loss_history = {}
        self.val_loss_history = {}
        self.train_dist_history = {}
        self.val_dist_history = {}
        self.tf_sched = tf_sched or (lambda e: 0.0)
        self.af_sched = af_sched or (lambda e: 0.0)

    def get_checkpoint(self):
        return {"train_loss_history": self.train_loss_history,
                "val_loss_history": self.val_loss_history,
                "train_dist_history": self.train_dist_history,
                "val_dist_history": self.val_dist_history,
                }

    def put_checkpoint(self, ckpt_dict):
        self.train_loss_history = ckpt_dict.get('train_loss_history', {})
        self.val_loss_history = ckpt_dict.get('val_loss_history', {})
        self.train_dist_history = ckpt_dict.get('train_dist_history', {})
        self.val_dist_history = ckpt_dict.get('val_dist_history', {})
    
    def run_epoch(self, epoch, end_epoch):
        tf = self.tf_sched(epoch)
        af = self.af_sched(epoch)
        print(f'\nEpoch {epoch}/{end_epoch} tf={tf} af={af}')
        train_dist, train_loss = self.train_model(tf=tf, af=af, prompt=f"Epoch {self.epoch}/{end_epoch}")
        self.train_loss_history[epoch] = train_loss
        self.train_dist_history[epoch] = train_dist
        plot_attention(self.model.get_attention())
        print(f'Train Dist: {train_dist:.2f} Train Loss: {train_loss:.2f}')
        val_dist, val_loss = self.eval_model()
        self.val_dist_history[self.epoch] = val_dist
        self.val_loss_history[self.epoch] = val_loss
        plot_attention(self.model.get_attention())
        print(f'Val Dist: {val_dist:.2f} Val Loss: {val_loss:.2f}')
        return val_dist, val_loss

    def train_model(self, data=None, criterion=None, prompt=None, tf=0.9, af=0.0):
        data = data or self.train_data
        optimizer = self.optim
        criterion = criterion or self.criterion
        total_loss = 0.0
        total_distance = 0.0
        self.model.train()
        t = tqdm(data)
        if prompt: t.set_description(prompt)
        for x, y, xlens, ylens in t:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(self.use_amp):
                predictions = self.model(x, xlens, y, tf, af)
                y_mask = torch.arange(y.shape[1])[None,:] >= ylens[:, None]
                y_mask = y_mask.to(device)
                loss = criterion(predictions.transpose(1, 2), y) # Take sequence as dimension
                loss.masked_fill_(y_mask, 0)
                loss = loss.sum(dim=1).mean() # Sum over sequence, mean over batch
            decoded = batch_decode(predictions)
            truths = index_to_transcripts(y)
            for i in range(len(decoded)):
                d = Levenshtein.distance(decoded[i], truths[i])
                total_distance += d
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()
            t.set_postfix({"loss": f"{loss.item():.2f}"})
            del x, y, predictions, y_mask
        avg_loss = total_loss / len(data)
        avg_dist = total_distance / len(data.dataset)
        torch.cuda.empty_cache()
        return avg_dist, avg_loss

    def eval_model(self, data=None, criterion=None, verbose=0.0):
        data = data or self.val_data
        criterion = criterion or self.criterion
        self.model.eval()
        total_loss = 0.0
        total_distance = 0
        for x, y, xlens, ylens in data:
            x, y = x.to(device), y.to(device)
            with autocast(self.use_amp):
                predictions = self.model(x, xlens, y)
                y_mask = (torch.arange(y.shape[1])[None,:] >= ylens[:, None]).to(device)
                loss = criterion(predictions.transpose(1, 2), y) # Take sequence as dimension
                loss.masked_fill_(y_mask, 0)
                loss = loss.sum(dim=1).mean()
            total_loss += loss.item()
            decoded = batch_decode(predictions)
            truths = index_to_transcripts(y)
            for i in range(len(decoded)):
                d = Levenshtein.distance(decoded[i], truths[i])
                total_distance += d
                if np.random.random() < verbose:
                    print(decoded[i])
                    print(truths[i])
                    print(d)
            del x, y, predictions
        avg_loss = total_loss / len(data)
        avg_dist = total_distance / len(data.dataset)
        torch.cuda.empty_cache()
        return avg_dist, avg_loss

    def plot_history(self):
        loss = [self.train_loss_history.get(e, None) for e in range(1, self.epoch)]
        dist = [self.val_dist_history.get(e, None) for e in range(1, self.epoch)]
        plt.plot(loss, label='loss')
        plt.plot(dist, label='dist')
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.show()

    def plot_loss(self):
        train = [self.train_loss_history.get(e, None) for e in range(1, self.epoch)]
        test = [self.val_loss_history.get(e, None) for e in range(1, self.epoch)]
        plt.plot(train, label='train')
        plt.plot(test, label='test')
        plt.title("Transcript Generation Cross-Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.show()
    
    def plot_dist(self):
        train = [self.train_dist_history.get(e, None) for e in range(1, self.epoch)]
        test = [self.val_dist_history.get(e, None) for e in range(1, self.epoch)]
        plt.plot(train, label='train')
        plt.plot(test, label='test')
        plt.title("Transcript Generation Levenshtein Distance")
        plt.xlabel('Epoch')
        plt.ylabel('Levenshtein Distance')
        plt.legend()
        plt.show()
