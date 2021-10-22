import json
from datetime import datetime
import pandas as pd
from flatland.utils.rendertools import RenderTool
from PIL import Image
import imageio
import os
import shutil


class Logger:
    def __init__(self, env):
        shutil.rmtree('tmp', ignore_errors=True)
        os.mkdir('tmp')
        
        self.env = env
        self.video_recorder = VideoRecorder()
        self.render_tool = RenderTool(env, gl="PILSVG")

    def log(self, observations, rewards):
        self.render_tool.render_env()
        self.video_recorder.add_image(self.render_tool.get_image())

        self.save_json('observations', observations)
        self.save_json('rewards', rewards)

    def save_json(self, name, data):
        if not os.path.isdir('tmp/' + name):
            os.mkdir('tmp/' + name)
        data = pd.Series(data).to_json(orient='values')
        with open('tmp/'+name+'/data-'+str(datetime.now().timestamp())+'.json', 'w') as fp:
            json.dump(data, fp)

    def save(self):
        self.video_recorder.save()


class VideoRecorder:
    def __init__(self):
        self.writer = imageio.get_writer('tmp/test.mp4', fps=5)

        os.mkdir('tmp/images')

    def add_image(self, image):
        path = 'tmp/images/' + str(datetime.now().timestamp()) + '.png'
        Image.fromarray(image).save(path)
        self.writer.append_data(imageio.imread(path))

    def save(self):
        shutil.rmtree('tmp/images')


def screenshot(env):
    render_tool = RenderTool(env, gl="PILSVG")
    render_tool.render_env()
    path = str(datetime.now().timestamp()) + '.png'
    Image.fromarray(render_tool.get_image()).save(path)


def save_json(observations):
    data = pd.Series(observations).to_json(orient='values')
    with open('next_observations-'+str(datetime.now().timestamp())+'.json', 'w') as fp:
        json.dump(data, fp)
