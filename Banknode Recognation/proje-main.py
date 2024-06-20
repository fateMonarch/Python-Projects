# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

model = load_model('proje-modeli.keras')
sys.setrecursionlimit(1000)
global kontrol
kontrol = 0

def predict_image_class(model, frame):
    img_array = frame.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)
    return decoded_predictions[0], decoded_predictions[1], decoded_predictions[2]

class FF(App):
    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        layout = Builder.load_file('proje-design.kv')
        return layout
               
    def on_start(self):
        global kontrol
        self.root.ids.loop.text = ''
        self.root.ids.snap.text = 'Snap'
        
        if kontrol == 0:
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def on_stop(self):
        global kontrol
        kontrol = 1
        self.capture.release()

        Clock.unschedule(self.update_frame)
        Window.close()
    
    def on_request_close(self, *args):
        global kontrol
        kontrol = 1
        self.capture.release()

        Clock.unschedule(self.update_frame)
        Window.close()
        
    def snap_button_pressed(self):
        self.root.ids.loop.disabled = False
        self.root.ids.snap.disabled = True
            
        self.root.ids.loop.background_color = 0.5, 0.5, 0.5, 1
        self.root.ids.snap.background_color = 0, 0, 0, 0
        self.root.ids.text.background_color = 1, 1, 1, 1
        
        self.root.ids.loop.color = 0.5, 0.5, 0.5, 1
        self.root.ids.snap.color = 0, 0, 0, 0
        self.root.ids.text.color = 0, 0, 0, 1
        self.root.ids.tick_or_cross.color = 1, 1, 1, 1
            
        self.root.ids.loop.text_color = 1, 1, 1, 1
        self.root.ids.snap.text_color = 0, 0, 0, 0
        
        self.root.ids.loop.text = 'Take Another Picture?'
        self.root.ids.snap.text = ''
        
        self.root.ids.frame.texture, self.root.ids.tick_or_cross.texture, self.root.ids.text.text, kontrol = self.capture_frame()
            
        return self.root.ids.loop.disabled, self.root.ids.snap.disabled, self.root.ids.loop.background_color, kontrol, 
        self.root.ids.snap.background_color, self.root.ids.loop.color, self.root.ids.snap.color, self.root.ids.text.color, 
        self.root.ids.loop.text_color, self.root.ids.snap.text_color, self.root.ids.tick_or_cross.color, self.root.ids.loop.text, 
        self.root.ids.frame.texture, self.root.ids.tick_or_cross.texture, self.root.ids.text.text, self.root.ids.snap.text,
        self.root.ids.text.background_color
            
    def loop_button_pressed(self):
        global kontrol
        kontrol = 0
        if kontrol == 0:
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
            
        self.root.ids.loop.disabled = True
        self.root.ids.snap.disabled = False
        
        self.root.ids.loop.background_color = 0, 0, 0, 0
        self.root.ids.snap.background_color = 0.5, 0.5, 0.5, 1
        self.root.ids.text.background_color = 0, 0, 0, 0
        
        self.root.ids.loop.color = 0, 0, 0, 0
        self.root.ids.snap.color = 0.5, 0.5, 0.5, 1
        self.root.ids.text.color = 0, 0, 0, 0
        self.root.ids.tick_or_cross.color = 0, 0, 0, 0

        self.root.ids.loop.text_color = 0, 0, 0, 0
        self.root.ids.snap.text_color = 1, 1, 1, 1
        
        self.root.ids.loop.text = ''
        self.root.ids.snap.text = 'Snap'
        
        return self.root.ids.loop.disabled, self.root.ids.snap.disabled, self.root.ids.loop.background_color, kontrol,  
        self.root.ids.snap.background_color, self.root.ids.loop.color, self.root.ids.snap.color, self.root.ids.text.color, 
        self.root.ids.loop.text_color, self.root.ids.snap.text_color, self.root.ids.tick_or_cross.color, self.root.ids.snap.text, 
        self.root.ids.loop.text, self.root.ids.text.background_color
                   
    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.root.ids.frame.texture = image_texture
            return self.root.ids.frame.texture
    
        return self.root.ids.frame.texture

    def capture_frame(self):
       ret, frame = self.capture.read()
       
       buf1 = cv2.flip(frame, 0)
       buf = buf1.tostring()
       
       image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
       image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
       self.root.ids.frame.texture = image_texture

       global kontrol
       kontrol = 1
       self.capture.release()
       if ret:
           threshold = 0.5
           gri_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
           pixels = gri_frame.flatten()
           average_pixel_value = np.mean(pixels)
           threshold_value = int(average_pixel_value * 0.5)
           _, thresholded_frame1 = cv2.threshold(gri_frame, threshold_value, 255, cv2.THRESH_BINARY)

           referans1 = cv2.imread('C:/Users/muham/.spyder-py3/TURK.png')
           gri_referans1 = cv2.cvtColor(referans1, cv2.COLOR_BGR2GRAY)
           _, thresholded_referans1 = cv2.threshold(gri_referans1, 128, 255, cv2.THRESH_BINARY)
           
           result1 = cv2.matchTemplate(thresholded_frame1, thresholded_referans1, cv2.TM_CCOEFF_NORMED)
           loc1 = np.where(result1 >= threshold)
           
           referans2 = cv2.imread('C:/Users/muham/.spyder-py3/DOLLARS.png')
           gri_referans2 = cv2.cvtColor(referans2, cv2.COLOR_BGR2GRAY)
           _, thresholded_referans2 = cv2.threshold(gri_referans2, 128, 255, cv2.THRESH_BINARY)
           
           result2 = cv2.matchTemplate(thresholded_frame1, thresholded_referans2, cv2.TM_CCOEFF_NORMED)
           loc2 = np.where(result2 >= threshold)   
           
           referans3 = cv2.imread('C:/Users/muham/.spyder-py3/EURO.png')
           gri_referans3 = cv2.cvtColor(referans3, cv2.COLOR_BGR2GRAY)
           _, thresholded_referans3 = cv2.threshold(gri_referans3, 128, 255, cv2.THRESH_BINARY)
           
           result3 = cv2.matchTemplate(thresholded_frame1, thresholded_referans3, cv2.TM_CCOEFF_NORMED)
           loc3 = np.where(result3 >= threshold)  
           
           tick = cv2.imread('C:/Users/muham/.spyder-py3/tick.jpg')
           
           buf3 = cv2.flip(tick, 0)
           buf5 = buf3.tostring()
 
           cross = cv2.imread('C:/Users/muham/.spyder-py3/cross.jpg')
           
           buf6 = cv2.flip(cross, 0)
           buf8 = buf6.tostring()
           
           if len(loc1[0]) >= 1:
               brightened_frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=80)
               _, thresholded_frame = cv2.threshold(brightened_frame, 208, 255, cv2.THRESH_BINARY)
               
               referans6 = cv2.imread('C:/Users/muham/.spyder-py3/ATAM.png')
               gri_referans6 = cv2.cvtColor(referans6, cv2.COLOR_BGR2GRAY)
               _, thresholded_referans6 = cv2.threshold(gri_referans6, 128, 255, cv2.THRESH_BINARY)
               
               result6 = cv2.matchTemplate(thresholded_frame, thresholded_referans6, cv2.TM_CCOEFF_NORMED)
               loc6 = np.where(result6 >= threshold)
               
               if len(loc6[0]) >= 1:
                   predicted_class1, predicted_class2, predicted_class3 = predict_image_class(model, frame)
                   if predicted_class1 == 'Original' or predicted_class1 == 'Rotated Original':
                       self.root.ids.text.text = 'That banknode is not fake and that is a ' + predicted_class1 + predicted_class2 + predicted_class3 + '.'
                       image_texture = Texture.create(size=(tick.shape[1], tick.shape[0]), colorfmt='bgr')
                       image_texture.blit_buffer(buf5, colorfmt='bgr', bufferfmt='ubyte')
                       self.root.ids.tick_or_cross.texture = image_texture
                       
                   else:
                       self.root.ids.text.text = 'That banknode is not fake and that is a ' + predicted_class1 + predicted_class2 + predicted_class3 + '.'
                       image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
                       image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
                       self.root.ids.tick_or_cross.texture = image_texture
                       
               else:
                   self.root.ids.text.text = 'That banknode is fake!'
                   image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
                   image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
                   self.root.ids.tick_or_cross.texture = image_texture
           
           elif len(loc2[0]) >= 1:
               brightened_frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=80)
               _, thresholded_frame = cv2.threshold(brightened_frame, 208, 255, cv2.THRESH_BINARY)
               
               referans7 = cv2.imread('C:/Users/muham/.spyder-py3/BENJAMIN.jpg')
               gri_referans7 = cv2.cvtColor(referans7, cv2.COLOR_BGR2GRAY)
               _, thresholded_referans7 = cv2.threshold(gri_referans7, 128, 255, cv2.THRESH_BINARY)
               
               result7 = cv2.matchTemplate(thresholded_frame, thresholded_referans7, cv2.TM_CCOEFF_NORMED)
               loc7 = np.where(result7 >= threshold)
               if len(loc7[0]) >= 1:
                   predicted_class1, predicted_class2, predicted_class3 = predict_image_class(model, frame)
                   if predicted_class1 == 'Original' or predicted_class1 == 'Rotated Original':
                       self.root.ids.text.text = 'That banknode is not fake and that is a ' + predicted_class1 + predicted_class2 + predicted_class3 + '.'
                       image_texture = Texture.create(size=(tick.shape[1], tick.shape[0]), colorfmt='bgr')
                       image_texture.blit_buffer(buf5, colorfmt='bgr', bufferfmt='ubyte')
                       self.root.ids.tick_or_cross.texture = image_texture
                       
                   else:
                       self.root.ids.text.text = 'That banknode is not fake and that is a ' + predicted_class1 + predicted_class2 + predicted_class3 + '.'
                       image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
                       image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
                       self.root.ids.tick_or_cross.texture = image_texture
                       
               else:
                   self.root.ids.text.text = 'That banknode is fake!'
                   image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
                   image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
                   self.root.ids.tick_or_cross.texture = image_texture
          
           elif len(loc3[0]) >= 1:
                brightened_frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=80)
                _, thresholded_frame = cv2.threshold(brightened_frame, 208, 255, cv2.THRESH_BINARY)
                
                referans8 = cv2.imread('C:/Users/muham/.spyder-py3/BEYAZ.png')
                gri_referans8 = cv2.cvtColor(referans8, cv2.COLOR_BGR2GRAY)
                _, thresholded_referans8 = cv2.threshold(gri_referans8, 128, 255, cv2.THRESH_BINARY)
                
                if thresholded_frame != thresholded_referans8:
                    predicted_class1, predicted_class2, predicted_class3 = predict_image_class(model, frame)
                    if predicted_class1 == 'Original' or predicted_class1 == 'Rotated Original':
                        self.root.ids.text.text = 'That banknode is not fake and that is a ' + predicted_class1 + predicted_class2 + predicted_class3 + '.'
                        image_texture = Texture.create(size=(tick.shape[1], tick.shape[0]), colorfmt='bgr')
                        image_texture.blit_buffer(buf5, colorfmt='bgr', bufferfmt='ubyte')
                        self.root.ids.tick_or_cross.texture = image_texture
                        
                    else:
                        self.root.ids.text.text = 'That banknode is not fake and that is a ' + predicted_class1 + predicted_class2 + predicted_class3 + '.'
                        image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
                        image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
                        self.root.ids.tick_or_cross.texture = image_texture
                        
                else:
                    self.root.ids.text.text = 'That banknode is fake!'
                    image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
                    image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
                    self.root.ids.tick_or_cross.texture = image_texture
                
           else:
               self.root.ids.text.text = 'There is not any banknode in the picture!'  
               image_texture = Texture.create(size=(cross.shape[1], cross.shape[0]), colorfmt='bgr')
               image_texture.blit_buffer(buf8, colorfmt='bgr', bufferfmt='ubyte')
               self.root.ids.tick_or_cross.texture = image_texture
                 
       return self.root.ids.frame.texture, self.root.ids.tick_or_cross.texture, self.root.ids.text.text, kontrol

if __name__ == '__main__':
    FF().run()
