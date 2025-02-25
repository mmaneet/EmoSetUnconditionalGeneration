# EmoSetUnconditionalGeneration
1. Set up Python venv.
2. Install diffuser module from source. 

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

3. Install Requirements.txt (bring file to project & run pip install requirements.txt)
4. add the EmoSet2.py file to site-packages/torchvision/datasets
5. navigate to diffusers\examples\unconditional_image_generation & replace the “train unconditional.py” file with the custom one in this repo
6. Install EmoSet from https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?rlkey=7f3oyjkr6zyndf0gau7t140rv&e=1&dl=0
7. Replace the .json files in the EmoSet folder with the ones in this repository
8. change your terminal directory to <path_from_root>\diffusers\examples\unconditional_image_generation\
9. run 

```bash
accelerate config default
```

10. run 

the following command: accelerate launch train_unconditional.py --data_root="path/to/EmoSet/Directory/On/Your/Laptop” --output_dir="EmoSet-unconditional-train" --mixed_precision="fp16”

```bash
accelerate launch train_unconditional.py --data_root="<path/to/EmoSet/Directory/On/Your/Laptop>" --output_dir="EmoSet-unconditional-train" --mixed_precision="fp16"
```
# Training
use the jupyter notebook file in this repo to test inference capabilities.
