import torch

def save_snapshots(epoch, model, optimizer, scheduler, path):
    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model
    torch.save({
        'epoch': epoch + 1,
        'model': module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, path)

def load_snapshots_to_model(path, model, optimizer, scheduler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

def load_epoch(path):
    checkpoint = torch.load(path)
    return checkpoint['epoch']
