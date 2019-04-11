import os
import torch

__all__ = ['restore']

def restore(snapshot_dir, model):
    restore_dir = snapshot_dir
    filelist = os.listdir(restore_dir)
    filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('.pth.tar')]
    if len(filelist) > 0:
        filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
        snapshot = os.path.join(restore_dir, filelist[0])
    else:
        snapshot = ''

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(snapshot))
        except KeyError:
            raise Exception("Loading pre-trained values failed.")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(snapshot))
        
        
def restore_test(args, model, group):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(group, args.num_folds))
    filename='step_%d.pth.tar'%(args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s'%(snapshot))


