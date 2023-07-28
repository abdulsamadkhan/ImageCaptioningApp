import gradio as gr
import requests, json
import os
import io
import IPython.display
import base64 
import torch

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")





#def greet(name):
#    return "Hello " + name +os.environ['HF_TOKENS']


#demo = gr.Interface(fn=greet, inputs="text", outputs="text")

#demo.launch()


#gr.close_all()
#gr.Textbox(os.environ['HF_TOKENS'])

#Image-to-text endpoint
def get_completion(image): 
    raw_image = Image.open(image).convert('RGB')

    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")
    
    out = model.generate(**inputs)
    return json.loads(processor.decode(out[0], skip_special_tokens=True))
    
    
    #    headers = {
#      "Authorization": f"Bearer {os.environ['HF_TOKENS']}",
#      "Content-Type": "application/json"
#    }
#    data = { "inputs": inputs }
#    if parameters is not None:
#        data.update({"parameters": parameters})
#    response = requests.request("POST",
#                                ENDPOINT_URL,
#                                headers=headers,
#                                data=json.dumps(data))
#    return json.loads(response.content.decode("utf-8"))




gr.close_all()
demo = gr.Interface(fn=get_completion,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never")



#demo = gr.Interface(fn=captioner,
 #                   inputs=[gr.Image(label="Upload image", type="pil")],
 #//                   outputs=[gr.Textbox(label="Caption")],
 # //                  title="Image Captioning with BLIP",
 #  //                 description="Caption any image using the BLIP model",
 #   //                allow_flagging="never",
 #    //               examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

demo.launch()