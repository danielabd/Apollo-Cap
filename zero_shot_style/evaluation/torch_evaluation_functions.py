#https://torchmetrics.readthedocs.io/_/downloads/en/stable/pdf/
from torchmetrics import BLEUScore, ROUGEScore,
from torchmetrics import

def main():
    preds = ['the cat is on the mat']
    target = [['there is a cat on the mat', 'a cat is on the mat']]
    metric = BLEUScore()
    metric(preds, target)
    Rouge
    print('finish')
if __name__=='__main__':
    main()