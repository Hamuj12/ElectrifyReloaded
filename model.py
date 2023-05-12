import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class ResistorDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir  # Add this line
        self.coco = COCO(annotation_file)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Filter out images without annotations
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

    def get_boxes(self, anns):
        boxes = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
        return torch.as_tensor(boxes, dtype=torch.float32)


    def __getitem__(self, index):
        # Get the image ID and load the image
        img_id = self.ids[index]
        img = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img['file_name'])
        image = Image.open(path).convert("RGB")

        # Get the annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(ann_ids)

        # Create masks for each annotation
        masks = []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if len(masks) > 1:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.array(masks)

        # Apply transformations
        if self.transforms is not None:
            image, masks = self.transforms(image, masks)

        # Convert masks to target format
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Create target dictionary
        target = {}
        target["boxes"] = self.get_boxes(anns)
        target["labels"] = torch.ones((len(anns),), dtype=torch.int64)
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)

        return image, target


    def __len__(self):
        return len(self.ids)
    
def get_transforms():
    def transform(img, masks):
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = img_transform(img)
        masks = np.stack(masks, axis=0)
        return img, masks

    return transform

train_dataset = ResistorDataset("ElectrifyReloaded.v2-model1.coco-segmentation/train", "ElectrifyReloaded.v2-model1.coco-segmentation/train/_annotations.coco.json", get_transforms())
val_dataset = ResistorDataset("ElectrifyReloaded.v2-model1.coco-segmentation/valid", "ElectrifyReloaded.v2-model1.coco-segmentation/valid/_annotations.coco.json", get_transforms())

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

def get_model_instance_segmentation(num_classes):
    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the box and mask predictors with new ones
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_model(model, dataloaders, device, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)
        running_loss = 0.0
        num_iter = 0
        
        for images, targets in dataloaders['train']:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            num_iter += 1
        
        average_loss = running_loss / num_iter
        print(f"Average Loss: {average_loss:.4f}")

    return model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a dictionary containing both DataLoaders
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
model = get_model_instance_segmentation(num_classes=2)
model.load_state_dict(torch.load("trained_resistor_detection_model.pt"))
model.to(device)
model.eval()

import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_polygons(output, threshold=0.5):
    masks = output['masks'].cpu().detach().numpy()
    polygons = []
    for mask in masks:
        mask = mask[0]
        mask = (mask > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                polygons.append(contour)
    return polygons

from torchvision.transforms import ToTensor

input_image = Image.open("img2.jpg").convert("RGB")
input_tensor = ToTensor()(input_image)
input_tensor = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

def draw_polygons(image, polygons):
    image_copy = image.copy()
    for polygon in polygons:
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_copy, [pts], True, (0, 255, 0), 2)
    return image_copy

polygons = get_polygons(output[0])
result_image = draw_polygons(np.array(input_image), polygons)

plt.imshow(result_image)
plt.axis("off")
plt.show()