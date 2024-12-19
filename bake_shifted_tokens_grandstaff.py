import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
import PIL
import yaml
from omegaconf import OmegaConf

import torchvision.transforms as transforms

from rqvae.models.rqvae.rqvae import RQVAE
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model



if __name__ == "__main__":
  # model_name = "sqocremavq_16_240_gray_unfit"
  model_name = "gsrqvae_f16_c1024_k4_unshared_128p_gray_unfit"
  config_path = list((Path("logs")/ model_name).rglob("config.yaml"))[0]
  config = OmegaConf.load(config_path)
  config = load_config(config_path)
  config.arch = augment_arch_defaults(config.arch)
  
  model, _ = create_model(config.arch)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.pt"))[0]
  model.load_state_dict(torch.load(ckpt_path)["state_dict"])
  model.cuda().eval()
  
  torch.set_grad_enabled(False)
  
  # image_path_list = list(Path("/home/sake/userdata/latent_score_dataset/string_quartet/segments").rglob("*/*/images/flattened_resampled/240_gray/*.png"))
  image_path_list = list(Path("/home/sake/userdata/olimpic_dataset/grandstaff-lmx").rglob("**/*_128.jpg"))
  filtered_pathlist = []
  for p in image_path_list:
    if "distorted" in p.name:
      continue
    elif p.name.startswith("."):
      continue
    else:
      filtered_pathlist.append(p)
  image_path_list = filtered_pathlist
  
  totensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5], [0.5])
  
  for image_path in tqdm(image_path_list):
    save_path = image_path.parent / (image_path.stem + f":{model_name}_shifted.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # if Path(str(save_path) + ":0_0.pt").exists():
    #   continue
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")
    # aspect_ratio = image.width / image.height
    # new_width = int(128 * aspect_ratio)
    # image = image.resize((new_width, 128))
    # resized_image_path = image_path.with_stem(image_path.stem + "_128")
    # image.save(resized_image_path)
    image = totensor(image)
    image = normalize(image)
    
    x_y_shifted_tokens = []
    for i in range(16):
      x_shifted_img = image[...,i:image.shape[-1]-15+i]
      y_shifted_imgs = []
      for j in range(4):
        y_shifted_imgs.append(torch.nn.functional.pad(x_shifted_img[:, j:x_shifted_img.shape[-2]-4+j], (0, 0, 4-j, j), mode='replicate'))
      y_shifted_imgs = torch.stack(y_shifted_imgs)
      out = model.get_codes(y_shifted_imgs.cuda())
      x_y_shifted_tokens.append(out.squeeze(0))
    x_y_shifted_tokens = torch.stack(x_y_shifted_tokens)
    torch.save(x_y_shifted_tokens, str(save_path))
    # break