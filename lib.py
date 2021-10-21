import json
from datetime import datetime 
import pandas as pd
from flatland.utils.rendertools import RenderTool
from PIL import Image
import imageio
import os
import shutil

class VideoRecorder:
    def __init__(self):
        self.writer = imageio.get_writer('test.mp4', fps=5)
        shutil.rmtree('tmp', ignore_errors=True)
        os.mkdir('tmp')
    def add_image(self, image):
        path = 'tmp/' + str(datetime.now().timestamp()) + '.png'
        Image.fromarray(image).save(path)
        self.writer.append_data(imageio.imread(path))
    def save(self):
        self.writer.close()
        shutil.rmtree('tmp')
        

def screenshot(env):
    render_tool = RenderTool(env, gl="PILSVG")
    render_tool.render_env()
    path = str(datetime.now().timestamp()) + '.png'
    Image.fromarray(render_tool.get_image()).save(path)


def save_json(observations):
    data = pd.Series(observations).to_json(orient='values')
    with open('next_observations-'+str(datetime.now().timestamp())+'.json', 'w') as fp:
        json.dump(data, fp)