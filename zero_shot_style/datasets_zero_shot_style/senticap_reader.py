import json
import os
import sys
import argparse

class SenticapSentence(object):
    """
    Stores details about a sentence.

    @ivar tokens: A tokenized version of the sentence with punctuation removed and
            words made lower case.
    @ivar word_sentiment: Indicates which words are part of an Adjective Noun
            Pair with sentiment; 1 iff the word is part of an ANP with sentiment.
    @ivar sentiment_polarity: Does this sentence express positive or negative sentiment.
    @ivar raw_sentence: The caption without any processing; taken directly from MTURK.
    """

    NEGATIVE_SENTIMENT = 0
    POSITIVE_SENTIMENT = 1

    def __init__(self):
        self.tokens = []
        self.word_sentiment = []
        self.sentiment_polarity = []
        self.raw_sentence = []

    def setTokens(self, tokens):
        assert isinstance(tokens, list)
        for tok in tokens:
            assert isinstance(tok, str) or isinstance(tok, unicode)

        self.tokens = tokens

    def setWordSentiment(self, word_sentiment):
        assert isinstance(word_sentiment, list)

        self.word_sentiment = [int(s) for s in word_sentiment]

    def setSentimentPolarity(self, sentiment_polarity):
        assert sentiment_polarity in [self.NEGATIVE_SENTIMENT, self.POSITIVE_SENTIMENT]
        
        self.sentiment_polarity = sentiment_polarity

    def setRawSentence(self, raw_sentence):
        assert isinstance(raw_sentence, str) or isinstance(raw_sentence, unicode)

        self.raw_sentence = raw_sentence

    def getTokens(self):
        return self.tokens

    def getWordSentiment(self):
        return self.word_sentiment

    def getSentimentPolarity(self):
        return self.sentiment_polarity

    def getRawsentence(self):
        return self.raw_sentence

class SenticapImage(object):
    """
    Stores details about a sentence.

    @ivar filename: The filename of the image in the MSCOCO dataset
    @ivar imgid: A unique but arbritrary number assigned to each image.
    @ivar sentences: A list of sentences corresponding to this image of 
            type `SenticapSentence`.
    @ivar split: Indicates if this is part of the TEST, TRAIN or VAL split.

    """
    TEST_SPLIT = 0
    TRAIN_SPLIT = 1
    VAL_SPLIT = 2


    def __init__(self):
        self.filename = ""
        self.imgid = None
        self.sentences = [] #stylized
        self.factual_sentences = [] #factual
        self.split = None
        self.imgpath = ''

    def set_factual_sentences(self, factual_sentences):
        self.factual_sentences = factual_sentences

    def set_imgpath(self,path):
        self.imgpath = path

    def set_imgpath_by_dir_path(self, imgs_folder,split,imgs_folder2017):
        '''
        if split not in self.filename:
            type_dir = self.filename.split('_')[1].split('2014')[0]
            target_img_path = os.path.join(imgs_folder2017, type_dir + '2017', self.filename.split('_')[-1])
        else:
            target_img_path = os.path.join(imgs_folder, split + '2014', self.filename)
            if os.path.isfile(target_img_path):
                self.imgpath = target_img_path
                return 1  # success to find correct path
        return 3  # failed to find fullpath to the image
        '''
        target_img_path = []
        target_img_path.append(os.path.join(imgs_folder2017, 'train2014', self.filename.split('_')[-1]))
        target_img_path.append(os.path.join(imgs_folder2017, 'val2014', self.filename.split('_')[-1]))
        target_img_path.append(os.path.join(imgs_folder2017, 'test2014', self.filename.split('_')[-1]))
        target_img_path.append(os.path.join(imgs_folder2017, 'train2017', self.filename.split('_')[-1]))
        target_img_path.append(os.path.join(imgs_folder2017, 'val2017', self.filename.split('_')[-1]))
        target_img_path.append(os.path.join(imgs_folder2017, 'test2017', self.filename.split('_')[-1]))
        for candidate_path in target_img_path:
            if os.path.isfile(candidate_path): #not sure about this line works well
                self.imgpath = candidate_path
                # print(f"self.filename: {self.filename}")
                # print(f"self.filename.split('_')[-1]): {self.filename.split('_')[-1]}")
                # print(f"split: {split}")
                # print(f"self.imgpath: {self.imgpath}")
                # type_dir = self.filename.split('_')[1].split('2014')[0]
                # if split not in self.filename:
                #     if split not in self.imgpath:
                #         if type_dir not in self.imgpath:
                #             print("check")
                return 2 #full path from 2017 dir
        #for folder in os.listdir(imgs_folder):
        #    for cur_file in os.listdir(os.path.join(imgs_folder,folder)):
        #        if cur_file==self.filename:
        #            self.imgpath = os.path.join(os.listdir(os.path.join(imgs_folder,folder)),self.filename)
        return 3 # didn't find path

    def setFilename(self, filename):
        assert isinstance(filename, str) or isinstance(filename, unicode)
        self.filename = filename

    def setImgID(self, imgid):
        self.imgid = imgid
    
    def addSentence(self, sentence):
        assert isinstance(sentence, SenticapSentence)
        self.sentences.append(sentence)

    def setSplit(self, split):
        assert split in [self.TEST_SPLIT, self.TRAIN_SPLIT, self.VAL_SPLIT]
        self.split = split

    def getFilename(self):
        return self.filename

    def getImgID(self):
        return self.imgid

    def getSentences(self):
        return self.sentences

    def getSplit(self):
        return self.split

    def get_imgpath(self):
        return self.imgpath

    def get_factual_sentences(self):
        return self.factual_sentences

class SenticapReader(object):
    """Handles the reading of the senticap dataset.
    Has functions to write examples to a simple csv format,
    and to count the number of examples.
    """

    images = []

    def __init__(self, filename,imgs_folder,imgs_folder2017):
        """
        Initializer that reads a senticap json file
        
        @param filename: the file path of the json file
        """
        self.imgids = []
        self.readJson(filename,imgs_folder,imgs_folder2017)
        self.filename = filename


    def get_images(self):
        return self.images


    def get_img_ids(self):
        return self.imgids


    def readJson(self, filename,imgs_folder,imgs_folder2017):
        """
        Read a senticap json file and load it into `SenticapImage` and
        `SenticapSentence` classes. The result is saved in `self.images`.
        
        @param filename: the file path of the json file
        """
        print("Starting to create senticap dataset...")
        data = json.load(open(filename, "r"))
        imgs_from_2017 = []
        wrong_path = []
        number_of_captions = 0
        number_of_neg_captions = 0
        number_of_pos_captions = 0
        splits = {"train": 0, "val": 0, "test": 0}
        captions_splits = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}}
        for image in data["images"]:
            splits[image["split"]] += 1
            self.imgids.append(image["imgid"])
            #create the SenticapImage entry
            im = SenticapImage()
            im.setFilename(image["filename"])
            if image["split"] not in ["train","val","test"]:
                print("check split!")
            if image["split"] == "train":
                im.setSplit(im.TRAIN_SPLIT)
            elif image["split"] == "test":
                im.setSplit(im.TEST_SPLIT)
            elif image["split"] == "val":
                im.setSplit(im.VAL_SPLIT)
            im.setImgID(image["imgid"])

            cur_captions_splits = im.set_imgpath_by_dir_path(imgs_folder,image["split"],imgs_folder2017)
            #for this image create all the SenticapSentence entries
            for sent in image["sentences"]:
                number_of_captions += 1
                se = SenticapSentence()
                se.setTokens(sent["tokens"])
                se.setWordSentiment(sent["word_sentiment"])
                if sent["sentiment"] == 0:
                    se.setSentimentPolarity(se.NEGATIVE_SENTIMENT)
                    number_of_neg_captions += 1
                    captions_splits[image["split"]]["neg"] += 1
                else:
                    se.setSentimentPolarity(se.POSITIVE_SENTIMENT)
                    number_of_pos_captions += 1
                    captions_splits[image["split"]]["pos"] += 1
                se.setRawSentence(sent["raw"])
                im.addSentence(se)

            self.images.append(im)

        print(f"{len(imgs_from_2017)} images from 2017.")
        print(f"{len(wrong_path)} were not taken in account because wrong image path.")
        print(f"There are {len(self.images)} good images for data.")
        print(f"Total number of captions:{number_of_captions}")
        print(f"Total number of neg captions:{number_of_neg_captions}")
        print(f"Total number of pos captions:{number_of_pos_captions}")
        print(f"Total number of train images:{splits['train']}, with {captions_splits['train']['pos']} positive captions and {captions_splits['train']['neg']} negative captions")
        print(f"Total number of val images:{splits['val']}, with {captions_splits['val']['pos']} positive captions and {captions_splits['val']['neg']} negative captions")
        print(f"Total number of test images:{splits['test']}, with {captions_splits['test']['pos']} positive captions and {captions_splits['test']['neg']} negative captions")

    def writeCSV(self, output_filename, train=True, test=True, val=True, pos=True, neg=True):
        """
        Write a CSV file from the examples matching the filter criteria. The
        columns of the csv are (filename, is_positive_sentiment, caption).
        where:  
            - B{filename:} is the filename of the MSCOCO image
            - B{is_positive_sentiment:} is 1 if the sentence expresses
                positive sentiment 0 if the sentence expresses
                negative sentiment 
            - B{caption:} is the tokenized, lowercase,
                punctuation removed sentence joined with space
                characters

        @param output_filename: path of csv to write
        @param test: include testing examples
        @param val: include validation examples
        @param pos: include positive sentiment examples
        @param neg: include negative sentiment examples
        """
        fout = open(output_filename, "w")
        fout.write("filename,is_positive_sentiment,caption\n")
        for im in self.images:
            if im.getSplit() == im.TEST_SPLIT and not test:
                continue
            if im.getSplit() == im.TRAIN_SPLIT and not train:
                continue
            if im.getSplit() == im.VAL_SPLIT and not val:
                continue
            sentences = im.getSentences()
            for sent in sentences:
                if sent.getSentimentPolarity() == sent.NEGATIVE_SENTIMENT and not neg:
                    continue
                if sent.getSentimentPolarity() == sent.POSITIVE_SENTIMENT and not pos:
                    continue
                fout.write('%s,%d,"%s"\n' % (im.getFilename(), 
                        sent.getSentimentPolarity()==sent.POSITIVE_SENTIMENT, 
                        ' '.join(sent.getTokens())))
        fout.close()

    def countExamples(self, train=True, test=True, val=True, pos=True, neg=True):
        """
        Count the number of examples matching the filter criteria

        @param train: include training examples
        @param test: include testing examples
        @param val: include validation examples
        @param pos: include positive sentiment examples
        @param neg: include negative sentiment examples
        @return: a tuple giving the number of images with sentences and the
                total number of sentences
        @rtype: `tuple(int, int)`
        """
        num_sentence = 0
        num_image_with_sentence = 0
        for im in self.images:
            if im.getSplit() == im.TEST_SPLIT and not test:
                continue
            if im.getSplit() == im.TRAIN_SPLIT and not train:
                continue
            if im.getSplit() == im.VAL_SPLIT and not val:
                continue

            image_has_sentence = False
            sentences = im.getSentences()
            for sent in sentences:
                if sent.getSentimentPolarity() == sent.NEGATIVE_SENTIMENT and not neg:
                    continue
                if sent.getSentimentPolarity() == sent.POSITIVE_SENTIMENT and not pos:
                    continue
                num_sentence += 1
                image_has_sentence = True
            if image_has_sentence:
                num_image_with_sentence += 1

        return (num_image_with_sentence, num_sentence)


def main():
    #handle arguments
    ap = argparse.ArgumentParser()    
    ap.add_argument("--filename", "-f", default='data/senticap_dataset.json',
            help = "Path to the senticap json")
    ap.add_argument("--imgs_folder",  default='../../source/coco/images',
            help = "Path to the senticap json")
    ap.add_argument("--csv_output", "-o", help = "Where to write the csv file.")
    ap.add_argument("--train", action="store_true", help = "Include the training examples")
    ap.add_argument("--test", action="store_true", help = "Include the testing examples")
    ap.add_argument("--val", action="store_true", help = "Include the validation examples")
    ap.add_argument("--pos", action="store_true", 
            help = "Include the positive sentiment examples")
    ap.add_argument("--neg", action="store_true",
            help = "Include the negative sentiment examples")
    args = ap.parse_args()

    args.train = 20

    args.val = 20

    args.test = 20
    args.pos = 20
    args.neg = 20
    sr = SenticapReader(args.filename,args.imgs_folder)
    if args.csv_output:
        sr.writeCSV(args.csv_output, train=args.train, test=args.test, val=args.val)
    else:
        count = sr.countExamples(train=args.train, test=args.test, val=args.val,
                pos=args.pos, neg=args.neg)
        print ("Input Filename:", args.filename)
        print("Filters:")
        if args.train:
            print("Train")
        if args.test:
            print("Test")
        if args.val:
            print("Val")
        if args.pos:
            print("Positive")
        if args.neg:
            print("Negative")
        print("\n")
        print(f"Number of images: {count}\nNumber of Sentences: {count}")


if __name__ == "__main__":
    main()
