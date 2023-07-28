import gradio as gr
import requests, json
import os
import io
import IPython.display
from PIL import Image
import base64 

#def greet(name):
#    return "Hello " + name +os.environ['HF_TOKENS']


#demo = gr.Interface(fn=greet, inputs="text", outputs="text")

#demo.launch()


#gr.close_all()
#gr.Textbox(os.environ['HF_TOKENS'])

#Image-to-text endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL="http://internal-aws-prod-internal-revproxy-alb-11660607.us-west-1.elb.amazonaws.com/rev-proxy/huggingface/itt"): 
    headers = {
      "Authorization": f"Bearer {os.environ['HF_TOKENS']}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))


demo = gr.Interface(
    fn=get_completion,
    inputs=gr.inputs.Textbox(),
    outputs="text"
)

#image_url = "https://free-images.com/sm/9596/dog_animal_greyhound_983023.jpg"
#demo = gr.get_completion(image_url)

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)
    return result[0]['generated_text']

#gr.close_all()
#demo = gr.Interface(fn=captioner,
#                    inputs=[gr.Image(label="Upload image", type="pil")],
#                    outputs=[gr.Textbox(label="Caption")],
#                    title="Image Captioning with BLIP",
#                    description="Caption any image using the BLIP model",
#                    allow_flagging="never")



#demo = gr.Interface(fn=captioner,
 #                   inputs=[gr.Image(label="Upload image", type="pil")],
 #//                   outputs=[gr.Textbox(label="Caption")],
 # //                  title="Image Captioning with BLIP",
 #  //                 description="Caption any image using the BLIP model",
 #   //                allow_flagging="never",
 #    //               examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

#demo.launch()