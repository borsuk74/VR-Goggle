# Snippet from https://www.tensorflow.org/alpha/tutorials/load_data/tf_records
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes_feature(image):
    """Returns a bytes_list from an image tensor."""

    if image.dtype != tf.uint8:
        # `tf.io.encode_jpeg``requires tf.unit8 input images, with values between
        # 0 and 255. We do the conversion with the following function, if needed:
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

    # We convert the image tensor back into a byte list...
    image_string = tf.io.encode_jpeg(image, quality=90)

    # ... and then into a Feature:
    return _bytes_feature(image_string)


def convert_sample_to_example(sample):
    """ Convert image + label sample into Example for serialization."""

    # We convert our elements into serialized features:
    features = tf.train.Features(feature={
        'image': _image_as_bytes_feature(sample['image']),
        'label': _int64_feature(sample['label']),
    })

    # Then we wrap them into an Example:
    example = tf.train.Example(features=features)
    return example


def convert_example_to_sample(example):
    """ Parse the serialized example into an image + feature sample."""

    # We create a dictionary describing the features.
    features_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    # We pass the example and descriptive dict to TF parsing function:
    sample = tf.io.parse_single_example(example, features_description)

    # Finally, we decode the sample's JPEG-encoded image string into an actual image:
    sample['image'] = tf.io.decode_jpeg(sample['image'])

    return sample


def write_record():
    # Read image raw data, which will be embedded in the record file later.
    image_string = open('data/stone.jpg', 'rb').read()

    # Manually set the label to 0. This should be set according to your situation.
    label = 0

    # For each sample there are two features: image raw data, and label. Wrap them in a single dict.
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    # Create a `example` from the feature dict.
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Write the serialized example to a record file.
    with tf.io.TFRecordWriter('data/records/stone.tfrecords') as writer:
        writer.write(tf_example.SerializeToString())


def read_record():
    # Use dataset API to import date directly from TFRecord file.
    raw_image_dataset = tf.data.TFRecordDataset('data/records/stone.tfrecords')

    # Create a dictionary describing the features.
    # The key of the dict should be the same with the key in writing function.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    # Define the parse function to extract a single example as a dict.
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    # If there are more than one example, use a for loop to read them out.
    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        label = image_features['label'].numpy()



def main():
    write_record()
    read_record()

if __name__ == '__main__':
    main()