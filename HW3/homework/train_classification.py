import argparse, time, torch, torch.nn as nn, torch.optim as optim
from homework.models import load_model
from classification_dataset import load_data

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="classification_data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug", type=str, default="hflip")
    ap.add_argument("--save", type=str, default="classifier.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_data(args.data, "train", args.aug, args.bs)
    val_loader   = load_data(args.data, "val", "basic", args.bs, shuffle=False)

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val = 0.0

    for epoch in range(1, args.epochs+1):
        model.train(); t0 = time.time()
        tr_loss=tr_acc=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            logits=model(x)
            loss=criterion(logits,y)
            loss.backward(); opt.step()
            tr_loss+=loss.item(); tr_acc+=accuracy(logits,y)
        tr_loss/=len(train_loader); tr_acc/=len(train_loader)

        model.eval(); val_acc=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                val_acc+=accuracy(model(x),y)
        val_acc/=len(val_loader)
        print(f"Epoch {epoch:02d}: loss={tr_loss:.3f}, train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}, time={time.time()-t0:.1f}s")
        if val_acc>best_val:
            best_val=val_acc; torch.save(model.state_dict(), args.save)
            print(f"Saved best model (val_acc={val_acc:.3f})")
    print(f"Best val_acc={best_val:.3f}")

if __name__=="__main__": main()
