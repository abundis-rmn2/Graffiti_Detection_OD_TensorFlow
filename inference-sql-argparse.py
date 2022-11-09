import time
import argparse
import mysql.connector
import json

import requests
import io
import os
import numpy as np
import glob
import json

from six import BytesIO
from PIL import Image

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import ftplib

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key: value[0, :num_detections].numpy()
                 for key, value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
      output_dict['detection_masks'], output_dict['detection_boxes'],
      image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict

labelmap_path = 'labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
print("Load labelmap")
tf.keras.backend.clear_session()
model = tf.saved_model.load('inference_graph/saved_model')
print("Load inference graph")

global_time = time.time()
parser = argparse.ArgumentParser(description='Paso de parámetros')
parser.add_argument("-MUID", dest="p_MUID", help="MUID to fetch")
params = parser.parse_args()

c = open("config.json")
config = json.load(c)
#MUID = 'asoter_1_hashtagTop_9_cec6fcb9'
MUID = params.p_MUID

dir_exist = os.path.exists("./exported_images/" + MUID)

def directory_exists(dir,ftp):
    filelist = []
    ftp.retrlines('LIST',filelist.append)
    for f in filelist:
        if f.split()[-1] == dir and f.upper().startswith('D'):
            return True
    return False

def DataUpload(local_dir, target_dir):
    ftp_server = ftplib.FTP(config["FTP"]["hostname"],config["FTP"]["username"],config["FTP"]["password"])
    ftp_server.encoding = "utf-8"
    #ftp_server.login()
    ftp_server.cwd('/media/exported_images')
    if directory_exists(target_dir, ftp_server) is False:  # (or negate, whatever you prefer for readability)
        print(target_dir)
        ftp_server.mkd(target_dir)
    ftp_server.cwd(target_dir)
    # https://stackoverflow.com/questions/67520579/uploading-a-files-in-a-folder-to-ftp-using-python-ftplib
    print("Uploading exported batch")
    toFTP = os.listdir(local_dir)
    for filename in toFTP:
        if filename not in ftp_server.nlst():
            print("Uploading: ")
            with open(os.path.join(local_dir, filename), 'rb') as file:  # Here I open the file using it's  full path
                ftp_server.storbinary(f'STOR {filename}', file)  # Here I store the file in the FTP using only it's name as I intended
            print(filename)
        else:
            print("File already exist")
    ftp_server.quit()

if not dir_exist:
    #os.makedirs(user_dir, 0o777)
    os.makedirs("./exported_images/" + MUID, 0o777)
    print("The dir was created")
else:
    print("The dir already exist")

try:
    cnx = mysql.connector.connect(user=config["SQL"]["username"],
                                  password=config["SQL"]["password"],
                                  host=config["SQL"]["hostname"],
                                  database=config["SQL"]["database"],
                                  )
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
else:
    print("Looking for caption in MUID:", MUID)
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM data_media WHERE MUID IN ('%s') " % (MUID))
    posts = cursor.fetchall()
    print("MUID found :", len(posts))
    asset_url = ''
    img_format = ''

    for post in posts:
        inference_dict = []
        print("Image URL:")
        print(post)
        if post[6] == 1:
            asset_url_jpg = "http://data.abundis.com.mx/media/" + post[14] + "/" + post[1] + "_" + post[3] + ".jpg"
            asset_url_webp = "http://data.abundis.com.mx/media/" + post[14] + "/" + post[1] + "_" + post[3] + ".webp"
            r_webp = requests.head(asset_url_webp)
            r_jpg = requests.head(asset_url_jpg)
            if r_webp.headers['Content-Type'] == 'image/webp':
                asset_url = asset_url_webp
                img_format= "webp"
            elif r_jpg.headers['Content-Type'] == 'image/jpeg':
                asset_url = asset_url_jpg
                img_format = "jpg"

        img_data = requests.get(asset_url).content
        if img_format == 'webp':
            with open('./downloaded_images/'+post[4]+'.webp', 'wb') as handler:
                handler.write(img_data)
            filename = post[4]+'_exported.webp'
            image_np = load_image_into_numpy_array('./downloaded_images/'+post[4]+'.webp')
            output_dict = run_inference_for_single_image(model, image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            im = Image.fromarray(image_np)
            im.save('./exported_images/' +MUID+ '/' + filename)

            print("File inferences", filename)
            print("with at least 0.5 of score")
            for d_class, d_score in zip(output_dict['detection_classes'][:5], output_dict['detection_scores'][:5]):
                if d_score > 0.5:
                    d_class_name = category_index[d_class]['name']
                    print('{0} with score {1}'.format(d_class_name, d_score))
                    inference_dict.append( ( d_class_name, float(d_score) ) )

        elif img_format == 'jpg':
            with open('./downloaded_images/'+post[4]+'.jpg', 'wb') as handler:
                handler.write(img_data)
            filename = post[4] + '_exported.jpg'
            image_np = load_image_into_numpy_array('./downloaded_images/' + post[4] + '.jpg')
            output_dict = run_inference_for_single_image(model, image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            im = Image.fromarray(image_np)
            im.save('./exported_images/' +MUID+ '/' + filename)

            print("File inferences", filename)
            print("with at least 0.5 of score")
            c = 1
            for d_class, d_score in zip(output_dict['detection_classes'][:5], output_dict['detection_scores'][:5]):
                if d_score > 0.5:

                    d_class_name = category_index[d_class]['name']
                    print('{0} with score {1}'.format(d_class_name, d_score))
                    inference_dict.append( ( d_class_name, float(d_score) ) )

        print("Inference to JSON and then SQL")
        inference_json = json.dumps(inference_dict)
        print(inference_json)
        cnx.reconnect()
        innercursor = cnx.cursor()
        sql_inference = "UPDATE data_media SET inference_custom = %s WHERE id = %s"
        val = (inference_json, post[0])
        innercursor.execute(sql_inference, val)
        cnx.commit()

        print(innercursor.rowcount, "registros afectado/s")


    DataUpload('./exported_images/' +MUID+ '/', MUID)
