import torch.nn as nn
import torch


def test_model(model, dataloader, logging):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    # Iterate over data.
    for i, (inputs, labels) in enumerate(dataloader):

        with torch.no_grad():
        # wrap them in Variable
            inputs = inputs.cuda()
            labels = labels.cuda()
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects / len(dataloader)

    logging.info('test Acc: {:.4f}'.format(epoch_acc))