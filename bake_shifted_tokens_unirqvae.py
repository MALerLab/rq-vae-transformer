import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import yaml
from omegaconf import OmegaConf

import torchvision.transforms as transforms

from rqvae.models.rqvae.rqvae import RQVAE
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model



if __name__ == "__main__":
  model_name = "unirqvae_f16_c1024_k4"
  config_path = list((Path("logs")/ model_name).rglob("config.yaml"))[0]
  config = OmegaConf.load(config_path)
  config = load_config(config_path)
  config.arch = augment_arch_defaults(config.arch)
  
  model, _ = create_model(config.arch)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.pt"))[0]
  model.load_state_dict(torch.load(ckpt_path)["state_dict"])
  model.cuda().eval()
  
  torch.set_grad_enabled(False)
  
  image_path_list = list(Path("/home/sake/userdata/latent_score_dataset_resize/").rglob("*/*/*/images/crop_resized/*.png"))
  filtered_pathlist = []
  for p in image_path_list:
    if p.parents[4].stem in ["0-2", "8-0", "8-2", "30-0"]:
      filtered_pathlist.append(p)
  image_path_list = filtered_pathlist
  
  totensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5], [0.5])
  
  for image_path in tqdm(image_path_list):
    save_path = (image_path.parent.parent.parent / "image_tokens" / (model_name) / "shifted" / image_path.stem).with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")

    # Filter
    width, height = image.size
    if height < 70 or height > 390 or height > width:
      print(f"Skipping {image_path} due to invalid dimensions: {width}x{height}")
      continue

    image = totensor(image)
    image = normalize(image)
    
    # pixel shifting to get 8*4 shifted tokens for a single image
    x_y_shifted_tokens = []
    for i in range(8):
      x_shifted_img = image[...,i:image.shape[-1]-7+i]
      y_shifted_imgs = []
      for j in range(4):
        y_shifted_imgs.append(torch.nn.functional.pad(x_shifted_img[:, j:x_shifted_img.shape[-2]-4+j], (0, 0, 4-j, j), mode='replicate'))
      y_shifted_imgs = torch.stack(y_shifted_imgs)
      out = model.get_codes(y_shifted_imgs.cuda())
      x_y_shifted_tokens.append(out.squeeze(0))
    x_y_shifted_tokens = torch.stack(x_y_shifted_tokens)
    torch.save(x_y_shifted_tokens, str(save_path))