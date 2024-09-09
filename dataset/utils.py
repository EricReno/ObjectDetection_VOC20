import torch

class CollateFunc(object):
    def __call__(self, batch):
        images = []
        targets = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)

        return images, targets