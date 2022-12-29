import os.path

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
#image_paths = [os.path.join(imgs_dir,'3.jpeg'),os.path.join(imgs_dir,'5.jpeg')]
image_paths = [os.path.join(imgs_dir,x) for x in os.listdir(imgs_dir) if x[0]!='.' and "attention" not in x]
imgs_name = [x for x in os.listdir(imgs_dir) if x[0]!='.' and "attention" not in x]

images = [Image.open(x) for x in image_paths]

inputs = processor(
    text=["Love"], images=images, return_tensors="pt", padding=True
)
#outputs_0 = model(**inputs)
outputs = model(**inputs, output_attentions = True)
vision_model_output = outputs.vision_model_output
attentions = vision_model_output["attentions"]
fig = plt.figure()
fig.suptitle("Main Title", fontsize=15)

for layer_i,attention in enumerate(attentions):
    for sample_i in range(attention.shape[0]):
        for head_i in range(attention.shape[1]):
            attention_head = attention[sample_i,head_i,:,:]
            attention_head_np = attention_head.cpu().data.numpy()
            fig.add_subplot(attentions[0].shape[0], attentions[0].shape[1],
                              sample_i * attentions[0].shape[1] + head_i + 1)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(attention_head_np)

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



plt.subplot(211)
plt.imshow(np.random.random((100, 100)))
plt.subplot(212)
plt.imshow(np.random.random((100, 100)))



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