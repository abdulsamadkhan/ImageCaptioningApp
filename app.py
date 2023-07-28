import gradio as gr
import requests, json
import os
import io
#import IPython.display
#from PIL import Image
#import base64 



#Image-to-text endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL="http://internal-aws-prod-internal-revproxy-alb-11660607.us-west-1.elb.amazonaws.com/rev-proxy/huggingface/itt"]): 
    headers = {
      "Authorization": f"Bearer {HF_KEY}",
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


#image_url = "https://free-images.com/sm/9596/dog_animal_greyhound_983023.jpg"
#display(IPython.display.Image(url=image_url))
#get_completion(image_url)

def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()