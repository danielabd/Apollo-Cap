# Apollo-Cap
Pytorch implementation of [Apollo: Zero-Shot Multimodal Reasoning with Multiple Experts](todo put here link arxiv)

## Stylized Image Caption Generation
<!--- - Comparing APOLLO-CAP-PD, CapDec, and ZeroCap+IPM approaches across positive, negative, humorous, and romantic styles:---> 
### Examples from SentiCap and Flickrstyle10k datasets:
![](git_images/Apollo_examples_r.png)
<!--- - Comparing all approaches with a focus on positive image captions:--->
### Ablation study of our approach for positive caption
![](git_images/all_approaches_cake.png)
## Audio-Aware Image Caption Generation
### APOLLO-CAP-P caption examples for images and audio clips featuring children’s laughter:
![](git_images/audio_apollo.png)

## Set up environment:
```bash
$ pip install requirements.txt
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
The code was tested successfully on Intel Xeon with NVIDIA RTX 2080 Ti and CUDA 11.4.

## Usage
To execute the following commands, go to `zero_shot_style/`.
### To generate stylized image caption:
#### Apollo-Cap:
##### positive:
python run.py --style positive --ce_scale 1.96 --clip_scale 2.19 --text_style_scale 9.68 --sentiment_temperature 0.001 --use_img_path ./example_images/000000276434.jpg
##### negative:
python run.py --style negative --ce_scale 2.855 --clip_scale 5.036 --text_style_scale 11.9 --sentiment_temperature 0.001 --use_img_path ./example_images/000000155617.jpg
##### humor:
python run.py --style humor --ce_scale 0.6604141408776456 --clip_scale 1 --text_style_scale 2.9876837003652907 --sentiment_temperature  0.001 --use_img_path ./example_images/2211593099_4a4f1c85d2.jpg
##### romantic:
python run.py --style romantic --ce_scale 0.7097647446401579 --clip_scale 1 --text_style_scale 4.332869432646197 --sentiment_temperature  0.001 --use_img_path ./example_images/1579287915_4257c54451.jpg

#### Apollo-Cap-P:
##### positive:
python run.py --style positive --mul_clip_style --ce_scale 4 --clip_scale 8 --text_style_scale 0 --sentiment_temperature 0.01 --use_img_path ./example_images/000000274455.jpg
##### negative:
python run.py --style negative --mul_clip_style --ce_scale 0.6209475551271303 --clip_scale 2 --text_style_scale 0 --sentiment_temperature 0.08555964306820746 --use_img_path ./example_images/000000217303.jpg
##### humor:
python run.py --style humor --mul_clip_style --ce_scale 0.4173438996507689 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.05089738868653932 --use_img_path ./example_images/311267421_e204e643cf.jpg
##### romantic:
python run.py --style romantic --mul_clip_style --ce_scale 0.5 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.05364761206623257 --use_img_path ./example_images/3457315666_b943111dec.jpg

#### Apollo-Cap-PD:
##### positive
python run.py --style positive --update_ViT --mul_clip_style --ce_scale 0.2214432225421577 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.1430339855494212 --num_iterations_clip_style 1 --use_img_path ./example_images/000000276434.jpg
##### negative
python run.py --style negative --update_ViT --mul_clip_style --ce_scale 0.6070550610590508 --clip_scale 2 --text_style_scale 0 --sentiment_temperature 0.17425402664880124 --num_iterations_clip_style 1 --use_img_path ./example_images/000000077954.jpg
##### humor:
python run.py --style humor --update_ViT --mul_clip_style --ce_scale 0.3426740175716766 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.05655316717625009 --num_iterations_clip_style 1 --use_img_path ./example_images/940973925_a2e6d7951c.jpg
##### romantic:
python run.py --style romantic --update_ViT --mul_clip_style --ce_scale 0.3735 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.0672676323972359 --num_iterations_clip_style 1 --use_img_path ./example_images/1489286545_8df476fa26.jpg

### To generate audio-aware image caption:
#### Apollo-Cap-P:
python run.py --mul_clip_style --ce_scale 4 --clip_scale 8 --text_style_scale 0 --sentiment_temperature 0.01  --use_audio_model --audio_path ./child_laughing.wav --audio_sampling_rate 24000 --use_img_path ./example_images/000000155617.jpg


## Citation
If you use our work for your research, please cite us. <!--- TODO: put bib tex ---> 

