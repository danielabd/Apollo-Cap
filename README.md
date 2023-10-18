# Apollo-Cap
Pytorch implementation of [Apollo: Zero-Shot Multimodal Reasoning with Multiple Experts](todo put here link arxiv)

## Stylized Image Caption Generation
<!--- - Comparing APOLLO-CAP-PD, CapDec, and ZeroCap+IPM approaches across positive, negative, humorous, and romantic styles:---> 
### Examples from Senticap and Flickrstyle10k datasets:
![](git_images/Apollo_examples_r.png)
<!--- - Comparing all approaches with a focus on positive image captions:--->
### Ablation study of our approach
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
### To generate stylized image caption:
#### Apollo-Cap:
##### positive:
python run.py --style positive --ce_scale 1.96 --clip_scale 2.19 --text_style_scale 9.68 --sentiment_temperature 0.001 --use_img_path <image_path>
##### negative:
python run.py --style negative --ce_scale 2.855 --clip_scale 5.036 --text_style_scale 11.9 --sentiment_temperature 0.001 --use_img_path <image_path>
##### humor:
python run.py --style humor --ce_scale 0.6604141408776456 --clip_scale 1 --text_style_scale 2.9876837003652907 --sentiment_temperature  0.001 --use_img_path <image_path>
##### romantic:
python run.py --style romantic --ce_scale 0.7097647446401579 --clip_scale 1 --text_style_scale 4.332869432646197 --sentiment_temperature  0.001 --use_img_path <image_path>

#### Apollo-Cap-P:
##### positive:
python run.py --style positive --mul_clip_style --ce_scale 4 --clip_scale 8 --text_style_scale 0 --sentiment_temperature 0.01 --use_img_path <image_path>
##### negative:
python run.py --style negative --mul_clip_style --ce_scale 0.6209475551271303 --clip_scale 2 --text_style_scale 0 --sentiment_temperature 0.08555964306820746 --use_img_path <image_path>
##### humor:
python run.py --style humor --mul_clip_style --ce_scale 0.4173438996507689 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.05089738868653932 --use_img_path <image_path>
##### romantic:
python run.py --style romantic --mul_clip_style --ce_scale 0.5 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.05364761206623257 --use_img_path <image_path>

#### Apollo-Cap-PD:
##### positive
python run.py --style positive --update_ViT --mul_clip_style --ce_scale 0.2214432225421577 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.1430339855494212 --num_iterations_clip_style 1 --use_img_path <image_path>
##### negative
python run.py --style negative --update_ViT --mul_clip_style --ce_scale 0.6070550610590508 --clip_scale 2 --text_style_scale 0 --sentiment_temperature 0.17425402664880124 --num_iterations_clip_style 1 --use_img_path <image_path>
##### humor:
python run.py --style humor --update_ViT --mul_clip_style --ce_scale 0.3426740175716766 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.05655316717625009 --num_iterations_clip_style 1 --use_img_path <image_path>
##### romantic:
python run.py --style romantic --update_ViT --mul_clip_style --ce_scale 0.3735 --clip_scale 1 --text_style_scale 0 --sentiment_temperature 0.0672676323972359 --num_iterations_clip_style 1 --use_img_path <image_path>

### To generate audio-aware image caption:
#### Apollo-Cap-PD:
python run.py --mul_clip_style --ce_scale 4 --clip_scale 8 --text_style_scale 0 --sentiment_temperature 0.01  --use_audio_model --use_img_path <image_path> --audio_path ~/data/for_audio/argumentwav.wav --audio_sampling_rate 24000 --use_img_path <image_path>


## Citation
If you use our work for your research, please cite us. <!--- TODO: put bib tex ---> 

