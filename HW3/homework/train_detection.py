import argparse, time, torch, torch.nn as nn, torch.optim as optim
from models import load_model

def import_drive_loader():
    for name in ["drive_dataset","road_dataset"]:
        try:
            mod=__import__(name)
            if hasattr(mod,"load_data"): return mod.load_data
        except Exception: continue
    raise ImportError("Could not find drive_dataset.py or road_dataset.py")

def iou_from_logits(logits,y_true,num_classes=3):
    preds=logits.argmax(1);ious=[]
    for c in range(num_classes):
        pred,truth=preds==c,y_true==c
        inter=(pred&truth).sum().float(); union=(pred|truth).sum().float().clamp(min=1)
        ious.append((inter/union).item())
    return sum(ious)/num_classes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="drive_data"); ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--bs",type=int,default=8); ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--save",default="detector.pt"); ap.add_argument("--lambda_depth",type=float,default=1.0)
    args=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_data=import_drive_loader()
    train_loader=load_data(args.data,split="train",batch_size=args.bs)
    val_loader=load_data(args.data,split="val",batch_size=args.bs,shuffle=False)

    model=load_model("detector",in_channels=3,num_seg_classes=3).to(device)
    ce=nn.CrossEntropyLoss(); l1=nn.L1Loss(); opt=optim.Adam(model.parameters(),lr=args.lr)
    best_miou=0.0

    for epoch in range(1,args.epochs+1):
        model.train(); t0=time.time(); tr_loss=tr_miou=0
        for batch in train_loader:
            x=batch["image"].to(device); seg=batch["track"].long().to(device); dep=batch["depth"].float().to(device)
            opt.zero_grad(); seg_logits,dep_pred=model(x)
            loss=ce(seg_logits,seg)+args.lambda_depth*l1(dep_pred,dep)
            loss.backward(); opt.step()
            tr_loss+=loss.item(); tr_miou+=iou_from_logits(seg_logits,seg)
        tr_loss/=len(train_loader); tr_miou/=len(train_loader)

        model.eval(); val_miou=val_l1=0
        with torch.no_grad():
            for batch in val_loader:
                x=batch["image"].to(device); seg=batch["track"].long().to(device); dep=batch["depth"].float().to(device)
                seg_logits,dep_pred=model(x)
                val_miou+=iou_from_logits(seg_logits,seg); val_l1+=l1(dep_pred,dep).item()
        val_miou/=len(val_loader); val_l1/=len(val_loader)
        print(f"Epoch {epoch:02d}: loss={tr_loss:.3f}, train_mIoU={tr_miou:.3f}, val_mIoU={val_miou:.3f}, val_L1={val_l1:.3f}, time={time.time()-t0:.1f}s")
        if val_miou>best_miou:
            best_miou=val_miou; torch.save(model.state_dict(),args.save)
            print(f"Saved best model (val_mIoU={val_miou:.3f})")
    print(f"Best val_mIoU={best_miou:.3f}")

if __name__=="__main__": main()
