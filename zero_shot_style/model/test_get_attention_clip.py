import os.path

#source code of CLIP by openAI to use and change the attentions:
#   https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L195


from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
'''
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)
'''
imgs_dir = os.path.join(os.path.join(os.path.expanduser('~'),'data','love_images'))
pos_imgs_dir = os.path.join(os.path.join(os.path.expanduser('~'),'data','pos_images'))
neg_imgs_dir = os.path.join(os.path.join(os.path.expanduser('~'),'data','neg_images'))
#image_paths = [os.path.join(imgs_dir,'3.jpeg'),os.path.join(imgs_dir,'5.jpeg')]
results_dir = os.path.join(os.path.join(os.path.expanduser('~'),'results'))

imgs_dir = pos_imgs_dir
image_paths = [os.path.join(imgs_dir,x) for x in os.listdir(imgs_dir) if x[0]!='.' and "attention" not in x and os.path.isfile(os.path.join(imgs_dir,x))]
imgs_names = [x.split('/')[-1].split('.')[0] for x in image_paths]
images = [Image.open(x) for x in image_paths]

pos_imgs_names = imgs_names
pos_images = images

imgs_dir = neg_imgs_dir
image_paths = [os.path.join(imgs_dir,x) for x in os.listdir(imgs_dir) if x[0]!='.' and "attention" not in x and os.path.isfile(os.path.join(imgs_dir,x))]
imgs_names = [str(int(x.split('/')[-1].split('.')[0])+10) for x in image_paths]
images = [Image.open(x) for x in image_paths]

neg_imgs_names = imgs_names
neg_images = images

images = []
images.extend(pos_images)
images.extend(neg_images)
imgs_names = []
imgs_names.extend(pos_imgs_names)
imgs_names.extend(neg_imgs_names)
inputs = processor(
    text=["Love"], images=images, return_tensors="pt", padding=True
)
#outputs_0 = model(**inputs)
outputs = model(**inputs, output_attentions = True)
vision_model_output = outputs.vision_model_output
attentions = vision_model_output["attentions"]


if not os.path.exists(os.path.join(results_dir, 'attentions')):
    os.mkdir(os.path.join(results_dir, 'attentions'))
for layer_i,attention in enumerate(attentions):
    if not os.path.exists(os.path.join(results_dir,'attentions','layer_'+str(layer_i))):
        os.mkdir(os.path.join(results_dir, 'attentions','layer_'+str(layer_i)))
    for head_i in range(attention.shape[1]):
        fig = plt.figure()
        fig.suptitle("Attention for pos imgs (top) and neg imgs (bottom)", fontsize=15)
        for sample_i in range(attention.shape[0]):
            #if not os.path.exists(os.path.join(results_dir, 'attentions', 'layer_'+str(layer_i),'head_'+str(head_i))):
            #    os.mkdir(os.path.join(results_dir, 'attentions', 'layer_'+str(layer_i),'head_'+str(head_i)))
            attention_head = attention[sample_i,head_i,:,:]
            attention_head_np = attention_head.cpu().data.numpy()
            fig.add_subplot(2, len(pos_imgs_names),
                              sample_i+1)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(attention_head_np)
            #tgt_fig_path = os.path.join(imgs_dir, 'attentions', "attentions.png")
            #plt.savefig(os.path.join(results_dir, 'attentions', 'layer_'+str(layer_i),'head_'+str(head_i),imgs_names[sample_i]+"_attentions.png"))
        plt.savefig(os.path.join(results_dir, 'attentions', 'layer_' + str(layer_i), 'head_' + str(head_i)))

plt.suptitle("Attentnos. X=heads, Y=samples", fontsize=15)
plt.xlabel("Heads")
plt.ylabel("batch")
tgt_fig_path = os.path.join(imgs_dir,"attentions.png")
plt.savefig(tgt_fig_path)
print(f"Saved figure to {tgt_fig_path}")
plt.gca().axes.get_yaxis().set_visible(False)

plt.show()
'''
ax1 = plt.subplot(attentions[0].shape[0], attentions[0].shape[1], sample_i*attentions[0].shape[1]+head_i+1)
plt.matshow(attention_head_np)
plt.title(f"i={imgs_name[sample_i]}, h={head_i}")
plt.show(block=False)
plt.hold()
'''


#
# plt.subplot(211)
# plt.imshow(np.random.random((100, 100)))
# plt.subplot(212)
# plt.imshow(np.random.random((100, 100)))



'''

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)

            inputs = processor(images=image, return_tensors="pt")

            image_features = model.get_image_features(**inputs)

            #transformers.FlaxCLIPVisionModel
            #image_fts_with_attentions = [self.clip.get_image_features(x,output_attentions=True) for x in clip_imgs] # Tuple of jnp.ndarray (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
            #Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


            #todo: cntinue too investigate it
'''

plt.savefig(imgs_dir+"attentions.png")
# for idx, cl in enumerate(np.unique(y)):
'''
plt.scatter(attention_head[:])
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show(block=False)
'''
attention_head_np = attention_head.cpu().data.numpy()
plt.matshow(attention_head_np)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities