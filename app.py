import gradio as gr
import requests, json
import torch

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



def get_completion(raw_image): 

    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")
    
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
    
    

gr.close_all()
demo = gr.Interface(fn=get_completion,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model, Made by Abdul Samad",
                    allow_flagging="never",
                   examples=["rottweiler_puppy_dog_background_4.jpg", "rottweiler_dog_wedding_dresses.jpg", "cheetah_animal_predator_525169.jpg",
                            "woman_cheetah_animal_human.jpg","man_stands_near_wild.jpg","calidris_alba_bird_nature.jpg"]
                   )



demo.launch()