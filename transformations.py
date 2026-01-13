from torchvision import transforms
import torchvision.transforms.functional as TF

def contrast_transform(factor: float):
    return transforms.Lambda(lambda img: TF.adjust_contrast(img, factor))

def transformation(transformation, image_size, mean, std):
    match transformation:
        case "none":
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # deterministic resize to final_size
                transforms.ToTensor(),                        # 0..1
                transforms.Normalize(mean=mean, std=std)     # encoder expectation
            ])

        case "zoomed":
            # zoom in by a zoom_factor (>1). We'll first resize up, then center-crop to final_size.
            zoom_factor = 1.3333333333  # 1 / 0.75, same visual zoom as your earlier 0.75 scale
            up_size = int(round(image_size * zoom_factor))
            transform = transforms.Compose([
                transforms.Resize((up_size, up_size)),       # enlarge
                transforms.CenterCrop(image_size),           # crop center -> zoomed view
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        case "greyscale":
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),  # keep 3 channels for encoder
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        case "contrast":
            # deterministic contrast multiplier: choose a multiplier e.g. 1.25 (25% stronger)
            contrast_factor = 1.25
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                contrast_transform(contrast_factor),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        case "lowresolution":
            downscale_factor = 0.15  # keep 25% of the resolution, adjust as needed
            down_size = int(round(image_size * downscale_factor))

            transform = transforms.Compose([
                transforms.Resize((down_size, down_size)),       # downsample
                transforms.Resize((image_size, image_size)),     # upscale back to normal size
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        case "flipped":
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        # segment anything

        case _:
            raise ValueError(f"Unknown transform: {transform}")
    
    return transform

def multi_transformation(transform_list, transform_data):
    image_size, mean, std = transform_data
    transforms_combined = []

    for transformation in transform_list:
        transformation = transformation.strip()

        match transformation:
            case "zoomed":
                zoom_factor = 1.3333333333
                up_size = int(round(image_size * zoom_factor))
                transforms_combined.extend([
                    transforms.Resize((up_size, up_size)),
                    transforms.CenterCrop(image_size)
                ])

            case "greyscale":
                transforms_combined.extend([
                    transforms.Resize((image_size, image_size)),
                    transforms.Grayscale(num_output_channels=3)
                ])

            case "contrast":
                contrast_factor = 1.25
                transforms_combined.extend([
                    transforms.Resize((image_size, image_size)),
                    contrast_transform(contrast_factor)
                ])

            case "lowresolution":
                downscale_factor = 0.15
                down_size = int(round(image_size * downscale_factor))
                transforms_combined.extend([
                    transforms.Resize((down_size, down_size)),
                    transforms.Resize((image_size, image_size))
                ])

            case "flipped":
                transforms_combined.extend([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=1.0)
                ])

            case "none":
                transforms_combined.extend([
                    transforms.Resize((image_size, image_size))
                ])

            case _:
                raise ValueError(f"Unknown transform: {transformation}")

    transforms_combined.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(transforms_combined)