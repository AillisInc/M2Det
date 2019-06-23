import torch

def save_snapshots(epoch, model, optimizer, scheduler, path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, path)

def load_snapshots_to_model(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

def load_epoch(path):
    checkpoint = torch.load(path)
    return checkpoint['epoch']
