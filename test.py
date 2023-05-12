import os
import sys
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tkinter import Tk, filedialog
import cv2
from pycocotools.coco import COCO
import torchvision
from torchvision.transforms import ToTensor

# Import your custom functions
# from your_module import get_model_instance_segmentation, get_polygons, draw_polygons
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

def get_polygons(output, threshold):
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

def draw_polygons(image, polygons):
    image_copy = image.copy()
    for polygon in polygons:
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_copy, [pts], True, (0, 255, 0), 2)
    return image_copy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load("trained_resistor_detection_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Set up the file explorer
    root = Tk()
    root.withdraw()

    num = 0
    while True:
        try:
            num += 1
            print("Select an image to infer:")
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])

            if not file_path:
                print("No image selected, exiting.")
                sys.exit(0)

            image = Image.open(file_path).convert("RGB")
            input_tensor = ToTensor()(image)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)

            polygons = get_polygons(outputs[0], 0.5)
            result_image = draw_polygons(np.array(image), polygons)

            # Save the resulting image
            result_image = Image.fromarray(result_image)
            # save into the folder above the image
            path = "C:/Users/hmuj2/Documents/ElectrifyReloaded"
            result_image.save(os.path.join(path, f"{num}_result.png"))

            print(f"Image with polygons saved as {os.path.splitext(file_path)[0]}_result.png")
        except KeyboardInterrupt:
            print("Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    main()
