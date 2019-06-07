'''This module implements some ML methods to detect users who exhibit signs of depression.'''

import csv

from . import data
from . import logging_helper as lh

CAT_FILE = 'app/resources/allsubreddits.csv'
CATEGORIES = ['Depression', 'NotDepression']

SOURCE_FILE = 'app/resources/RS_2017-10.bz2'

DEPTOPICS = ['add', 'cripplingalcoholism', 'disorder', 'Health',
'HealthProject', 'leaves', 'MenGetRapedToo', 'rapecounseling',
'7CupsofTea', 'addiction', 'ADHD', 'Advice', 'affirmations', 'afterthesilence',
'Agoraphobia', 'AlAnon', 'alcoholicsanonymous', 'alcoholism', 'Anger', 'Antipsychiatry',
'Anxiety', 'Anxietyhelp', 'anxietysuccess', 'anxietysupporters', 'AskDocs',
'aspergers', 'AspiePartners', 'AtheistTwelveSteppers', 'behavior', 'behaviortherapy',
'bingeeating', 'BipolarReddit', 'BipolarSOs', 'BodyAcceptance', 'BPD', 'bulimia',
'CompulsiveSkinPicking', 'dbtselfhelp', 'depression', 'depressionregimens',
'disability', 'distractit', 'domesticviolence', 'downsyndrome', 'DysmorphicDisorder',
'eating_disorders', 'EatingDisorderHope', 'EatingDisorders', 'emetophobia',
'EOOD', 'ForeverAlone', 'fuckeatingdisorders', 'GetMotivated', 'getting_over_it',
'GFD', 'HaveHope', 'HealthAnxiety', 'helpmecope', 'itgetsbetter', 'leaves',
'mentalhealth', 'mentalillness', 'mentalpod', 'mixednuts', 'MMFB', 'MSTsurvivors',
'needadvice', 'Needafriend', 'neurodiversity', 'NoFap', 'nosurf', 'OCD', 'OCPD',
'offmychest', 'OpiatesRecovery', 'PanicParty', 'Phobia', 'PsychiatricFreedom',
'Psychiatry', 'psychology', 'psychopathology', 'psychotherapy', 'psychotic_features',
'psychoticreddit', 'ptsd', 'quittingkratom', 'rape', 'rapecounseling', 'reasonstolive',
'rehabtherapy', 'sad', 'schizoaffective', 'schizophrenia', 'secondary_survivors',
'selfharm', 'SelfHarmCommunity', 'selfhelp', 'siblingsupport', 'slp', 'SMARTRecovery',
'socialanxiety', 'socialskills', 'socialwork', 'socialworkresources', 'specialed',
'StopDipping', 'stopdrinking', 'stopgaming', 'StopSelfHarm', 'stopsmoking',
'StopSelfHarm', 'stopsmoking', 'SuicideWatch', 'survivorsofabuse', 'swami', 'Teetotal',
'TheMixedNuts', 'tOCD', 'Tourettes', 'traumatoolbox', 'Trichsters', 'TwoXADHD',
'uniqueminds', 'whatsbotheringyou'] 


def process_csv(csvfile):
    with open(csvfile) as cfile:
        topicsreader = csv.reader(cfile)
        for row in topicsreader:
            print(', '.join(row))


def prepare_notdepression():
    '''Returns a collection of posts of the category NotDepression.'''

    # Fetch for non-depression topics.
    fetch = data.FetchMultiple(SOURCE_FILE, DEPTOPICS)
    posts = fetch.get_posts()

    return posts


def prepare_depression(n):
    '''Returns a colleciton of posts of the category Depression.'''

    top = data.Top()
    posts = top.get_positives(n)

    return posts 

def prepare_input():

    # TODO: use of random library to extract random posts for the big dataset.
    notdepr = prepare_notdepression()
    depr = prepare_depression(500000)
    print(len(notdepr))
    print(type(notdepr))
    print('-'*10)
    print(len(depr))
    print(type(depr))


def main():

    # Prepare depression.
    # depres = prepare_depression(1000000)
    # print(len(depres))
    prepare_input()

main()