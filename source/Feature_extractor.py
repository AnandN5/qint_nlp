import utils
import re


def punt_features(tokens, i):
    features = {
        'next-word-capitalized': tokens[i + 1][0].isupper(),
        'prev-word': tokens[i - 1][0],
        'punctuation': tokens[i],
        'prev-word-is-one-char': len(tokens[i - 1]) == 1,
        'next-word-is-newline': tokens[i + 1] == '\n'
    }
    return features


def word_tag_features(untagd_sents, index, history):
    """
     untagged_sents sentence: [w1, w2, ...], index: the index of the word
    """
    try:
        return {
            'word': untagd_sents[index],
            'is_first': index == 0,
            'is_last': index == len(untagd_sents) - 1,
            'is_capitalized': untagd_sents[index][0].upper() == untagd_sents[index][0],
            'is_all_caps': untagd_sents[index].upper() == untagd_sents[index],
            'is_all_lower': untagd_sents[index].lower() == untagd_sents[index],
            'prefix-1': untagd_sents[index][0],
            'prefix-2': untagd_sents[index][:2],
            'prefix-3': untagd_sents[index][:3],
            'suffix-1': untagd_sents[index][-1],
            'suffix-2': untagd_sents[index][-2:],
            'suffix-3': untagd_sents[index][-3:],
            'prev_word': '' if index == 0 else untagd_sents[index - 1],
            'prev_tag': '' if index == 0 else history[index - 1],
            'next_word': '' if index == len(untagd_sents) - 1 else untagd_sents[index + 1],
            'has_hyphen': '-' in untagd_sents[index],
            'is_numeric': untagd_sents[index].isdigit(),
            'capitals_inside': untagd_sents[index][1:].lower() != untagd_sents[index][1:]
        }
    except Exception as e:
        raise(e)


def chunk_features(sent_tags, i, history):
    tag = sent_tags[i]
    if i == 0:
        prevtag = "<START>"
    else:
        prevtag = sent_tags[i - 1]
    if i == len(sent_tags) - 1:
        nexttag = "<END>"
    else:
        nexttag = sent_tags[i + 1]

    features = {
        'tag': tag,
        'is-first': True if i == 0 else False,
        'is-last': True if i == len(sent_tags) - 1 else False,
        'tag-1': '' if i == 0 else sent_tags[i - 1],
        'tag-2': '' if i <= 1 else sent_tags[i - 2],
        'iob-1': '' if i == 0 else history[i - 1],
        'iob-2': '' if i <= 1 else history[i - 2],
        'tag+1': '' if i == len(sent_tags) - 1 else sent_tags[i + 1],
        'tag+2': '' if i >= len(sent_tags) - 2 else sent_tags[i + 2],
        'prevtag+tag': '%s+%s' % (prevtag, tag),
        'tag+nexttag': '%s+%s' % (tag, nexttag),
        'tags_since_dt': utils.tags_since_dt(sent_tags, i)
    }
    return features


def primary_key_feature(leaf):
    leaf_item = leaf[0]
    leaf_tag = leaf[1]
    special_characters = re.findall('[-_]+', leaf_item)
    features = {
        'word': leaf_item,
        'tag': leaf_tag,
        'contains_hyphen': '-' in leaf_item,
        'contains_underscore': '_' in leaf_item,
        'no:_special_characters': len(special_characters),
        'special_characters': special_characters
        }
    return features
