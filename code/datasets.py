import os
import random
import codecs
from torchtext import data

pad_id = 1
class DB(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
        
    def __init__(self, datadir, text_field, label_field, label_num, sect='train', examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            datadir: The path to the data directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            label_num: The total number of labels.
            type: train | val | test
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def remove_unk(s, *args):
            return s - 1
        label_field.postprocessing = data.Pipeline(remove_unk)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []
            with codecs.open(os.path.join(datadir, 'topicclass_%s.txt' % sect), errors='ignore') as f:
                while True: 
                    line = f.readline().strip()
                    if not line:
                        break
                    text = line.split('|||')[1].strip()
                    label = line.split('|||')[0].strip()
                    examples.append(data.Example.fromlist([text, label], fields))
        super(DB, self).__init__(examples, fields, **kwargs)
