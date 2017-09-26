"""sudoku.py

An attempt to recognize numbers from an image of a sudoku grid
in which I learn that convnets trained on MNIST generalize terribly.

"""

import sys
import argparse

import tensorflow as tf
import numpy as np

FLAGS = None

def main(_):

    with open('./data/clean.jpg', 'rb') as imgfile:
        imgdata = imgfile.read()
        image = tf.image.rgb_to_grayscale(tf.image.decode_jpeg(imgdata))
        # image_bw = image // 128
        image_bw = image

    # do a bunch of processing to get patches containing numbers (or not)
    # and try to make input more MNISTish to increase accuracy
    imgs = tf.placeholder(tf.int32, shape=[81, None, None, 1])
    crop_img = lambda img: tf.image.central_crop(img, .9)
    cropped_imgs = tf.map_fn(crop_img, imgs)
    resized_imgs = tf.image.resize_area(cropped_imgs, [28, 28])[..., 0] / 256
    processed_imgs = tf.maximum(-resized_imgs + .8, 0) * 2
    input_imgs = tf.reshape(processed_imgs, [-1, 784])

    with tf.Session() as sess:
        # load trained MNIST digit recognizer
        new_saver = tf.train.import_meta_graph('./model/mnist.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
        x = tf.get_default_graph().get_tensor_by_name('input:0')
        dropout = tf.get_default_graph().get_tensor_by_name('dropout:0')
        y_net = tf.get_default_graph().get_tensor_by_name('output:0')

        # patch with number finding done in numpy because would be pain in the ass in tensorflow
        img_arr = sess.run(image_bw)
        w = img_arr.shape[0]
        h = img_arr.shape[1]
        num_patches = []
        for i in range(9):
            for j in range(9):
                # patch params
                px, py = int(w*(i+.5))//9 + 2, int(h*(j+.5))//9 + 2
                pw, ph = w//22, h//22
                patch = img_arr[px-pw : px+pw, py-ph : py+ph]
                num_patches.append(patch)
        num_patches = np.array(num_patches)

        # clean_imgs = sess.run(processed_imgs, {imgs: num_patches})
        input_imgs = sess.run(input_imgs, {imgs: num_patches})
        clean_imgs = np.reshape(input_imgs, [-1, 28, 28])

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # fig.add_subplot(1,1,1)
        # plt.imshow(clean_imgs[75])
        # plt.show()

        # result = y_net.eval(feed_dict={x: input_imgs[75][None], dropout: 1.0})
        # print(result)
        print(tf.reshape(tf.argmax(y_net, axis=1), [9, 9]).eval(
            feed_dict={x: input_imgs, dropout: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
