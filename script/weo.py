#-*- coding: utf-8 -*-
'''
transform World Economic Outlook data set from IMF into DDF data model.
'''

import pandas as pd
import numpy as np
import os
import re

# configuration of file paths
source = '../source/WEOApr2015all.xls'  # despite its name, it's actually a tabbed csv file.
output_dir = '../output/'        # output dir


# functions for building DDF
def to_concept_id(s):
    '''convert a string to lowercase alphanumeric + underscore id for concepts'''
    return re.sub(r'[/ -\.]+', '_', s).lower()


def extract_concepts_continuous(data):
    '''extract the continous concepts'''

    # headers for data frame and csv output
    headers_continuous = ['concept', 'concept_type', 'weo_subject_code', 'subject_descriptor',
                      'subject_notes', 'unit', 'scale', 'domain']

    # the continuous concepts in WEO data are represented by a 'WEO subject code'
    concepts_continuous = data['WEO Subject Code'].unique()

    # now build the data frame
    concepts_continuous_df = pd.DataFrame([], columns=headers_continuous)
    concepts_continuous_df['weo_subject_code'] = concepts_continuous
    concepts_continuous_df['concept'] = concepts_continuous_df['weo_subject_code'].apply(to_concept_id)
    concepts_continuous_df['concept_type'] = 'measure'
    # for one weo subject, there is only one unit/scale/subject notes/subject descriptor.
    # so calling unique()[0] to get that.
    concepts_continuous_df['subject_descriptor'] = [
        data[data['WEO Subject Code'] == x]['Subject Descriptor'].unique()[0]
        for x in concepts_continuous_df['weo_subject_code']]
    concepts_continuous_df['subject_notes'] = [
        data[data['WEO Subject Code'] == x]['Subject Notes'].unique()[0]
        for x in concepts_continuous_df['weo_subject_code']]
    concepts_continuous_df['unit'] = [
        data[data['WEO Subject Code'] == x]['Units'].unique()[0]
        for x in concepts_continuous_df['weo_subject_code']]
    concepts_continuous_df['scale'] = [
        data[data['WEO Subject Code'] == x]['Scale'].unique()[0]
        for x in concepts_continuous_df['weo_subject_code']]

    return concepts_continuous_df


def extract_concepts_discrete(data):
    '''extract the continous concepts'''

    # headers for data frame and csv output
    headers_discrete = ['concept', 'name', 'concept_type', 'domain', 'drillups']

    # because the columns contains all datapoints (from '1980' to '2020')
    # so we should remove them from discrete concepts
    discrete = data.columns

    # you may need to change these 2 lines to match the time range
    y1 = discrete.get_loc('1980')
    y2 = discrete.get_loc('2020')

    discrete = np.concatenate([discrete[:y1], discrete[y2+1:], ['year']])

    # now the discrete concepts are:
    # array(['WEO Country Code', 'ISO', 'WEO Subject Code', 'Country',
    #  'Subject Descriptor', 'Subject Notes', 'Units', 'Scale',
    #   'Country/Series-specific Notes', 'Estimates Start After', 'year']

    # As noted by Jasper, we should name the entity domain as singular noun.
    # but we have 'Units' in the columns from WEO data. So I manually set this
    # here ntil I find a better solution.
    discrete[6] = 'Unit'

    # also change some of country related columns for consistency
    discrete[1] = 'Country'
    discrete[3] = 'Country Name'

    # build data frame
    concepts_discrete_df = pd.DataFrame([], columns=headers_discrete)
    concepts_discrete_df['name'] = discrete
    concepts_discrete_df['concept'] = concepts_discrete_df['name'].apply(to_concept_id)

    concepts_discrete_df['concept_type'] = ['string', 'entity_domain', 'string',
                               'string', 'string', 'string', 'entity_domain',
                                'entity_domain', 'string', 'time', 'time'
                               ]

    return concepts_discrete_df


def extract_entities_country(data):
    '''extract entities for countries'''

    # headers for data frame and csv output
    headers_country = ['weo_country_code', 'country_name', 'country']

    # build data frame
    entities_weo_country_code = data['WEO Country Code'].unique()
    entities_iso = data['ISO'].unique()
    entities_country = data['Country'].unique()

    weo_country_df = pd.DataFrame([], columns=headers_country)
    weo_country_df['weo_country_code'] = entities_weo_country_code
    weo_country_df['country_name'] = entities_country
    weo_country_df['country'] = entities_iso
    weo_country_df['country'] = weo_country_df['country'].str.lower()

    return weo_country_df[['country', 'weo_country_code', 'country_name']]


def extract_entities_unit(data):
    ''' extract entities for units'''

    # headers for data frame and csv output
    headers_unit = ['unit', 'name', 'link']

    # build the data frame
    entities_units = data['Units'].unique()
    units_df = pd.DataFrame([], columns=headers_unit)
    units_df['name'] = entities_units
    units_df['unit'] = units_df['name'].apply(to_concept_id)

    return units_df


def extract_entities_scale(data):
    ''' extract entities for scales'''

    # headers for data frame and csv output
    headers_scale = ['scale', 'name', 'link']

    # build data frame
    entities_scale = data['Scale'].unique()

    scales_df = pd.DataFrame([], columns=headers_scale)
    scales_df['name'] = entities_scale

    scales_df['scale'] = scales_df['name'].str.lower()

    scales_df = scales_df.dropna(how='all')

    return scales_df


def extract_datapoints_country_year(data):
    ''' extract datapoints, for each weo subject, by country and year'''

    res = {}

    # loop through all WEO subject to create ddf file for each one.
    for subject in data['WEO Subject Code'].unique():

        headers_datapoints = ['year', 'country', subject.lower()]

        data_subj = data[data['WEO Subject Code'] == subject]

        data_subj = data_subj.set_index('ISO')
        data_subj = data_subj.T['1980':'2020']
        data_subj = data_subj.unstack().reset_index().dropna()

        data_subj = data_subj.iloc[:, [1, 0, 2]]  # rearrange columns
        data_subj.columns = headers_datapoints

        data_subj['country'] = data_subj['country'].apply(to_concept_id)

        res[subject.lower()] = data_subj

    return res


def extract_special_notes(data):
    '''There are special notes for each country/weo subject pair, we should
    extract them too. '''

    # headers for data frame and csv output
    headers_notes = ['country', 'concept', 'country_series_specific_notes', 'estimates_start_after']

    # build data frame
    special_notes = data[['ISO', 'WEO Subject Code', 'Country/Series-specific Notes', 'Estimates Start After']].copy()
    special_notes.columns = headers_notes
    special_notes['concept'] = special_notes['concept'].str.lower()
    special_notes['country'] = special_notes['country'].str.lower()

    return special_notes


if __name__ == '__main__':

    print('reading source file...')
    data = pd.read_csv(source, sep='\t', skip_footer=2, na_values=['n/a', '--'], engine='python')

    print('creating concepts ddf file...')
    concept_continuous = extract_concepts_continuous(data)
    path = os.path.join(output_dir, 'ddf--concepts--continuous.csv')
    concept_continuous.to_csv(path, index=False)

    concept_discrete = extract_concepts_discrete(data)
    path = os.path.join(output_dir, 'ddf--concepts--discrete.csv')
    concept_discrete.to_csv(path, index=False)

    print('creating entities ddf file...')
    entities_country = extract_entities_country(data)
    path = os.path.join(output_dir, 'ddf--entities--country.csv')
    entities_country.to_csv(path, index=False)

    entities_scale = extract_entities_scale(data)
    path = os.path.join(output_dir, 'ddf--entities--scale.csv')
    entities_scale.to_csv(path, index=False)

    entities_unit = extract_entities_unit(data)
    path = os.path.join(output_dir, 'ddf--entities--unit.csv')
    entities_unit.to_csv(path, index=False)

    print('creating data points ddf file...')
    datapoints = extract_datapoints_country_year(data)
    # save each series into csv
    for subj, df in datapoints.items():
        path = os.path.join(output_dir, 'ddf--datapoints--' + subj + '--by--country--year.csv')
        df.to_csv(path, index=False)

    print('creating special notes file...')
    notes = extract_special_notes(data)
    path = os.path.join(output_dir, 'ddf--notes.csv')
    # the float_format is work around for the problem that pandas will output
    # the year like 2013.0 in the csv file.
    notes.to_csv(path, index=False, float_format='%.0f')
