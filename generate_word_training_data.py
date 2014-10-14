# Max Jaderberg 16/5/14
# Generates training data using WordRenderer
import sys
import os
from titan_utils import is_cluster, get_task_id, crange
from word_renderer import WordRenderer, FontState, FileCorpus, TrainingCharsColourState, SVTFillImageState, wait_key, NgramCorpus
from scipy.io import savemat
import Image
import numpy as n
import tarfile

SETTINGS = {
    #####################################
    'SVT': {
        'corpus_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1/svt_lex_lower.txt",
                      "/mnt/sharedscratch/users/max/nips2014/svt1/svt_lex_lower.txt"],
        'fontstate':{
            'font_list': ["/Users/jaderberg/Data/TextSpotting/googlefonts/fontlist_good_8.5.14.txt",
                      "/mnt/sharedscratch/users/max/nips2014/googlefonts/fontlist_good_8.5.14.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/Users/jaderberg/Data/TextSpotting/mjchars/nips_training.mat",
                             "/mnt/sharedscratch/users/max/nips2014/nips_training.mat"],
        'fillimstate': {
            'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1",
                         "/mnt/sharedscratch/users/max/nips2014/svt1"],
            'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat",
                         "/mnt/sharedscratch/users/max/nips2014/svt1/SVT-train.mat"],
        }
    },
    #####################################
    'ICDAR': {
        'corpus_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest/nipslex.txt",
                      "/mnt/sharedscratch/users/max/nips2014/icdar2003/nipslex.txt"],
        'fontstate':{
            'font_list': ["/Users/jaderberg/Data/TextSpotting/googlefonts/fontlist_good_8.5.14.txt",
                      "/mnt/sharedscratch/users/max/nips2014/googlefonts/fontlist_good_8.5.14.txt"],
            'random_caps': 1,
        },
        'trainingchars_fn': ["/Users/jaderberg/Data/TextSpotting/mjchars/nips_training.mat",
                             "/mnt/sharedscratch/users/max/nips2014/nips_training.mat"],
        'fillimstate': {
            'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest",
                         "/mnt/sharedscratch/users/max/nips2014/icdar2003"],
            'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest/ICDAR2003_words_test.mat",
                         "/mnt/sharedscratch/users/max/nips2014/icdar2003/ICDAR2003_words_test.mat"],
        }
    },
    #####################################
    '50kDICT': {
        'corpus_fn': ["/Users/jaderberg/Data/TextSpotting/nips2014/lex50k.txt",
                      "/mnt/sharedscratch/users/max/nips2014/lex50k.txt"],
        'fontstate':{
            'font_list': ["/Users/jaderberg/Data/TextSpotting/googlefonts/fontlist_good_8.5.14.txt",
                      "/mnt/sharedscratch/users/max/nips2014/googlefonts/fontlist_good_8.5.14.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/Users/jaderberg/Data/TextSpotting/mjchars/nips_training.mat",
                             "/mnt/sharedscratch/users/max/nips2014/nips_training.mat"],
        'fillimstate': [
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1",
                             "/mnt/sharedscratch/users/max/nips2014/svt1"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat",
                             "/mnt/sharedscratch/users/max/nips2014/svt1/SVT-train.mat"],
            },
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest/ICDAR2003_words_test.mat",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003/ICDAR2003_words_test.mat"],
            }
        ]
    },
    #####################################
    '90kDICT': {
        'corpus_fn': ["/Users/jaderberg/Data/TextSpotting/nips2014/lex50k_expanded.txt",
                      "/mnt/sharedscratch/users/max/nips2014/lex50k_expanded.txt"],
        'fontstate':{
            'font_list': ["/Users/jaderberg/Data/TextSpotting/googlefonts/fontlist_good_8.5.14.txt",
                      "/mnt/sharedscratch/users/max/nips2014/googlefonts/fontlist_good_8.5.14.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/Users/jaderberg/Data/TextSpotting/mjchars/nips_training.mat",
                             "/mnt/sharedscratch/users/max/nips2014/nips_training.mat"],
        'fillimstate': [
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1",
                             "/mnt/sharedscratch/users/max/nips2014/svt1"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat",
                             "/mnt/sharedscratch/users/max/nips2014/svt1/SVT-train.mat"],
            },
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest/ICDAR2003_words_test.mat",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003/ICDAR2003_words_test.mat"],
            }
        ]
    },
    #####################################
    '10kNGRAM': {
        'ngram_mode': True,
        'corpus_fn': ["/Users/jaderberg/Data/TextSpotting/nips2014/ngram-encode10k_90k",
                      "/mnt/sharedscratch/users/max/nips2014/ngram-encode10k_90k"],
        'corpus_class': NgramCorpus,
        'substrings': 0.3,
        'fontstate':{
            'font_list': ["/Users/jaderberg/Data/TextSpotting/googlefonts/fontlist_good_8.5.14.txt",
                      "/mnt/sharedscratch/users/max/nips2014/googlefonts/fontlist_good_8.5.14.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/Users/jaderberg/Data/TextSpotting/mjchars/nips_training.mat",
                             "/mnt/sharedscratch/users/max/nips2014/nips_training.mat"],
        'fillimstate': [
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1",
                             "/mnt/sharedscratch/users/max/nips2014/svt1"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat",
                             "/mnt/sharedscratch/users/max/nips2014/svt1/SVT-train.mat"],
            },
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest/ICDAR2003_words_test.mat",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003/ICDAR2003_words_test.mat"],
            }
        ]
    },

    #####################################
    '90kDICTsubstr': {
        'ngram_mode': False,
        'corpus_fn': ["/Users/jaderberg/Data/TextSpotting/nips2014/lex50k_expanded.txt",
                      "/mnt/sharedscratch/users/max/nips2014/lex50k_expanded.txt"],
        'substrings': 0.3,
        'fontstate':{
            'font_list': ["/Users/jaderberg/Data/TextSpotting/googlefonts/fontlist_good_8.5.14.txt",
                      "/mnt/sharedscratch/users/max/nips2014/googlefonts/fontlist_good_8.5.14.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/Users/jaderberg/Data/TextSpotting/mjchars/nips_training.mat",
                             "/mnt/sharedscratch/users/max/nips2014/nips_training.mat"],
        'fillimstate': [
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1",
                             "/mnt/sharedscratch/users/max/nips2014/svt1"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat",
                             "/mnt/sharedscratch/users/max/nips2014/svt1/SVT-train.mat"],
            },
            {
                'data_dir': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003"],
                'gtmat_fn': ["/Users/jaderberg/Data/TextSpotting/DataDump/SceneTrialTest/ICDAR2003_words_test.mat",
                             "/mnt/sharedscratch/users/max/nips2014/icdar2003/ICDAR2003_words_test.mat"],
            }
        ]
    },
}

SAVE_GT = False

suffix = ""
NUM_TO_GENERATE = 10000
NUM_PER_FOLDER = 500
OUT_BASE = ["/Users/jaderberg/Data/TextSpotting/mjsynth/", "/mnt/sharedscratch/users/max/nips2014/mjsynth/"]
SAMPLE_HEIGHT = 32
QUALITY = [80, 10]

TARITUP = True
BATCHTAR = False
DELETE_AFTER_TAR = True

if __name__ == "__main__":

    iscluster = int(is_cluster())

    dataset = sys.argv[1]
    out_dir = os.path.join(OUT_BASE[iscluster], "%s%dpx%s" % (dataset, SAMPLE_HEIGHT, suffix))

    print 'Generating training data for %s, %dpx height to %s' % (dataset, SAMPLE_HEIGHT, out_dir)

    settings = SETTINGS[dataset]

    ngram_mode = settings.get('ngram_mode', False)

    # init providers
    try:
        corp_class = settings['corpus_class']
    except KeyError:
        corp_class = FileCorpus
    corpus = corp_class(settings['corpus_fn'][iscluster])
    fontstate = FontState(font_list=settings['fontstate']['font_list'][iscluster])
    fontstate.random_caps = settings['fontstate']['random_caps']
    colourstate = TrainingCharsColourState(settings['trainingchars_fn'][iscluster])
    if not isinstance(settings['fillimstate'], list):
        fillimstate = SVTFillImageState(settings['fillimstate']['data_dir'][iscluster], settings['fillimstate']['gtmat_fn'][iscluster])
    else:
        # its a list of different fillimstates to combine
        states = []
        for i, fs in enumerate(settings['fillimstate']):
            s = SVTFillImageState(fs['data_dir'][iscluster], fs['gtmat_fn'][iscluster])
            # move datadir to imlist
            s.IMLIST = [os.path.join(s.DATA_DIR, l) for l in s.IMLIST]
            states.append(s)
        fillimstate = states.pop()
        for fs in states:
            fillimstate.IMLIST.extend(fs.IMLIST)

    # take substrings
    try:
        substr_crop = settings['substrings']
    except KeyError:
        substr_crop = -1

    # init renderer
    sz = (800,200)
    WR = WordRenderer(sz=sz, corpus=corpus, fontstate=fontstate, colourstate=colourstate, fillimstate=fillimstate)

    # create out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    task_id = get_task_id()
    task_folder = os.path.join(out_dir, "%d" % task_id)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    if TARITUP and not BATCHTAR:
        tar = tarfile.open("%s.tar" % task_folder, 'w')

    # save the lexicon classes
    if task_id == 1:
        if not ngram_mode:
            savemat(os.path.join(out_dir, "lexicon.mat"),  {'lexicon': corpus.corpus_list})
        else:
            savemat(os.path.join(out_dir, "lexicon.mat"),  {'lexicon': corpus.words, 'ngramidx': corpus.idx, 'ngramcount': corpus.values})


    folder_id = 1
    num_in_folder = 1
    for i in crange(range(0, NUM_TO_GENERATE)):
        # gen sample
        try:
            data = WR.generate_sample(outheight=SAMPLE_HEIGHT, random_crop=True, substring_crop=substr_crop)
        except Exception:
            print "\tERROR"
            continue
        if data is None:
            print "\tcould not generate good sample"
            continue

        folder = os.path.join(task_folder, "%d" % folder_id)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if not ngram_mode:
            fnstart = "%d_%s_%d" % (num_in_folder, data['text'], data['label'])
        else:
            fnstart = "%d_%s_%d" % (num_in_folder, data['text'], data['label']['word_label'])

        # save with random compression
        quality = min(100, max(0, int(QUALITY[1]*n.random.randn() + QUALITY[0])))
        img = Image.fromarray(data['image'])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        imfn = os.path.join(folder, fnstart + ".jpg")
        img.save(imfn, 'JPEG', quality=quality)

        if ngram_mode:
            # save the ngramidxs
            ngramfn = os.path.join(folder, fnstart + '.ngram')
            f = open(ngramfn, 'w')
            for k, ng in enumerate(data['label']['ngram_labels']):
                f.write('%d %d\n' % (ng, data['label']['ngram_counts'][k]))
            f.close()

        if SAVE_GT:
            # save char groundtruth
            gt = {
                'text': data['text'],
                'label': data['label'],
                'chars': data['chars'],
            }
            matfn = os.path.join(folder, fnstart + ".mat")
            savemat(matfn, gt)

        if TARITUP and not BATCHTAR:
            # add image
            tar.add(imfn, arcname=os.path.join(str(folder_id), fnstart + ".jpg"))
            if ngram_mode:
                tar.add(ngramfn, arcname=os.path.join(str(folder_id), fnstart + ".ngram"))
            if SAVE_GT:
                # add mat file
                tar.add(matfn, arcname=os.path.join(str(folder_id), fnstart + ".mat"))
            if DELETE_AFTER_TAR:
                # remove them
                os.remove(imfn)
                if ngram_mode:
                    os.remove(ngramfn)
                if SAVE_GT:
                    os.remove(matfn)

        #print "\tsaved to", os.path.join(folder, fnstart + ".jpg")

        if num_in_folder > NUM_PER_FOLDER:
            num_in_folder = 1
            folder_id += 1
        else:
            num_in_folder += 1

    if TARITUP and BATCHTAR:
        print "Tar-ing data up..."
        cmd = 'tar -cf %s.tar %s' % (task_folder, task_folder)
        print "\t", cmd
        os.system(cmd)
        if DELETE_AFTER_TAR:
            cmd = 'rm -rf %s' % task_folder
            print '\t', cmd
            os.system(cmd)

    if TARITUP and not BATCHTAR:
        tar.close()

    print "FINISHED!"


