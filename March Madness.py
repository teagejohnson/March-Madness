# IMPORTS
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, root_mean_squared_error, roc_auc_score, r2_score

import xgboost as xgb

import pandas as pd
import numpy as np

import time
import glob
import json
import os


# OPTIONS
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


# CONSTANTS
TEAM_NAME_MAP = {
    "BYU": "Brigham Young",
    "ETSU": "East Tennessee State",
    "Fairleigh Dickinson": 'FDU',
    "Grambling State": "Grambling",
    "Hartford": "Hartford Hawks",
    "Loyola Chicago": "Loyola (IL)",
    "LIU": "Long Island University",
    "LIU Brooklyn": "Long Island University",
    "Louisiana–Lafayette": "Louisiana",
    "LSU": "Louisiana State",
    "MSU": "Michigan State",
    "Ole Miss": "Mississippi",
    "Penn": "Pennsylvania",
    "Pitt": "Pittsburgh",
    "Saint Mary's": "Saint Mary's (CA)",
    "SIU-Edwardsville": "Southern Illinois-Edwardsville",
    "SMU": "Southern Methodist",
    "Southern Miss": "Southern Mississippi",
    "St. Joseph's": "Saint Joseph's",
    "St. Mary's (CA)": "Saint Mary's (CA)",
    "St. Peter's": "Saint Peter's",
    "Texas A&M–Corpus Christi": "Texas A&M-Corpus Christi",
    "UConn": "Connecticut",
    "UCSB": "UC Santa Barbara",
    "UC-Davis": "UC Davis",
    "UC-Irvine": "UC Irvine",
    "UC-San Diego": "San Diego",
    "UMass": "Massachusetts",
    "UMBC": "Maryland-Baltimore County",
    "UNC": "North Carolina",
    "UNLV": "Nevada-Las Vegas",
    "USC": "Southern California",
    "VCU": "Virginia Commonwealth"
}

COLUMNS_BASIC = {
    # 'Rk': '',
    # 'Gtm': '',
    # 'Date': '',
    'Unnamed: 3': 'location',
    'Opp': 'opponent',
    'Type': 'type',
    'Rslt': 'result',
    # 'Tm': '',
    # 'Opp.1': '',
    # 'OT': '',
    # 'FG': '',
    # 'FGA': '',
    # 'FG%': '',
    # '3P': '',
    # '3PA': '',
    '3P%': 'three_point_percentage',
    # '2P': '',
    '2PA': 'two_pointers_attempted',
    '2P%': 'two_point_percentage',
    # 'eFG%': '',
    # 'FT': '',
    # 'FTA': '',
    'FT%': 'free_throw_percentage',
    # 'ORB': '',
    'DRB': 'defensive_rebounds',
    # 'TRB': '',
    # 'AST': '',
    # 'STL': '',
    # 'BLK': '',
    # 'TOV': '',
    'PF': 'fouls',
    'FG.1': 'opponent_field_goals_made',
    'FGA.1': 'opponent_field_goals_attempted',
    # 'FG%.1': '',
    # '3P.1': '',
    '3PA.1': 'opponent_three_pointers_attempted',
    '3P%.1': 'opponent_three_point_percentage',
    # '2P.1': '',
    # '2PA.1': '',
    '2P%.1': 'opponent_two_point_percentage',
    # 'eFG%.1': '',
    'FT.1': 'opponent_free_throws_made',
    'FTA.1': 'opponent_free_throws_attempted',
    'FT%.1': 'opponent_free_throw_percentage',
    'ORB.1': 'opponent_offensive_rebounds',
    # 'DRB.1': '',
    # 'TRB.1': '',
    'AST.1': 'opponent_assists',
    # 'STL.1': '',
    'BLK.1': 'opponent_blocks',
    # 'TOV.1': '',
    'PF.1': 'opponent_fouls'
}

COLUMNS_ADVANCED = {
    # 'Rk': '',
    # 'Gtm': '',
    # 'Date': '',
    # 'Unnamed: 3': '',
    # 'Opp': '',
    # 'Type': '',
    # 'Rslt': '',
    # 'Tm': '',
    # 'Opp.1': '',
    # 'OT': '',
    'ORtg': 'offensive_rating',
    'DRtg': 'opponent_offensive_rating',
    'Pace': 'pace',
    'FTr': 'free_throw_attempt_rate',
    '3PAr': 'three_point_attempt_rate',
    # 'TS%': '',
    'TRB%': 'total_rebound_percentage',
    'AST%': 'assist_percentage',
    # 'STL%': '',
    'BLK%': 'block_percentage',
    # 'eFG%': '',
    'TOV%': 'turnover_percentage',
    'ORB%': 'offensive_rebound_percentage',
    'FT/FGA': 'free_throw_rate',
    # 'eFG%.1': '',
    'TOV%.1': 'opponent_turnover_percentage',
    # 'DRB%': '',
    'FT/FGA.1': 'opponent_free_throw_rate'
}

COLUMNS = [
    'team',
    'opponent',
    'location',
    'type',
    'result',
    'pace',
    'offensive_rating',
    'two_point_percentage',
    'three_point_percentage',
    'free_throw_percentage',
    'assist_percentage',
    'block_percentage',
    'turnover_percentage',
    'offensive_rebound_percentage',
    'total_rebound_percentage',
    'total_foul_percentage',
    'three_point_attempt_rate',
    'free_throw_attempt_rate',
    'free_throw_rate',
    'opponent_offensive_rating',
    'opponent_two_point_percentage',
    'opponent_three_point_percentage',
    'opponent_free_throw_percentage',
    'opponent_assist_percentage',
    'opponent_block_percentage',
    'opponent_turnover_percentage',
    'opponent_offensive_rebound_percentage',
    'opponent_total_rebound_percentage',
    'opponent_total_foul_percentage',
    'opponent_three_point_attempt_rate',
    'opponent_free_throw_attempt_rate',
    'opponent_free_throw_rate'
]

REGRESSION_COLUMNS_X = [
    'seed',
    'opponent_seed',
    'pace',
    'offensive_rating',
    'two_point_percentage',
    'three_point_percentage',
    'free_throw_percentage',
    'assist_percentage',
    'block_percentage',
    'turnover_percentage',
    'offensive_rebound_percentage',
    'total_rebound_percentage',
    'total_foul_percentage',
    'three_point_attempt_rate',
    'free_throw_attempt_rate',
    'free_throw_rate',
    'home_offensive_rating',
    'home_pace',
    'home_two_point_percentage',
    'home_three_point_percentage',
    'home_free_throw_percentage',
    'home_assist_percentage',
    'home_block_percentage',
    'home_turnover_percentage',
    'home_offensive_rebound_percentage',
    'home_total_rebound_percentage',
    'home_total_foul_percentage',
    'home_three_point_attempt_rate',
    'home_free_throw_attempt_rate',
    'home_free_throw_rate',
    'away_offensive_rating',
    'away_pace',
    'away_two_point_percentage',
    'away_three_point_percentage',
    'away_free_throw_percentage',
    'away_assist_percentage',
    'away_block_percentage',
    'away_turnover_percentage',
    'away_offensive_rebound_percentage',
    'away_total_rebound_percentage',
    'away_total_foul_percentage',
    'away_three_point_attempt_rate',
    'away_free_throw_attempt_rate',
    'away_free_throw_rate',
    # 'neutral_offensive_rating',
    # 'neutral_pace',
    # 'neutral_two_point_percentage',
    # 'neutral_three_point_percentage',
    # 'neutral_free_throw_percentage',
    # 'neutral_assist_percentage',
    # 'neutral_block_percentage',
    # 'neutral_turnover_percentage',
    # 'neutral_offensive_rebound_percentage',
    # 'neutral_total_rebound_percentage',
    # 'neutral_total_foul_percentage',
    # 'neutral_three_point_attempt_rate',
    # 'neutral_free_throw_attempt_rate',
    # 'neutral_free_throw_rate',
    'defensive_offensive_rating',
    'defensive_pace',
    'defensive_two_point_percentage',
    'defensive_three_point_percentage',
    'defensive_free_throw_percentage',
    'defensive_assist_percentage',
    'defensive_block_percentage',
    'defensive_turnover_percentage',
    'defensive_offensive_rebound_percentage',
    'defensive_total_rebound_percentage',
    'defensive_total_foul_percentage',
    'defensive_three_point_attempt_rate',
    'defensive_free_throw_attempt_rate',
    'defensive_free_throw_rate',
    'home_defensive_offensive_rating',
    'home_defensive_pace',
    'home_defensive_two_point_percentage',
    'home_defensive_three_point_percentage',
    'home_defensive_free_throw_percentage',
    'home_defensive_assist_percentage',
    'home_defensive_block_percentage',
    'home_defensive_turnover_percentage',
    'home_defensive_offensive_rebound_percentage',
    'home_defensive_total_rebound_percentage',
    'home_defensive_total_foul_percentage',
    'home_defensive_three_point_attempt_rate',
    'home_defensive_free_throw_attempt_rate',
    'home_defensive_free_throw_rate',
    'away_defensive_offensive_rating',
    'away_defensive_pace',
    'away_defensive_two_point_percentage',
    'away_defensive_three_point_percentage',
    'away_defensive_free_throw_percentage',
    'away_defensive_assist_percentage',
    'away_defensive_block_percentage',
    'away_defensive_turnover_percentage',
    'away_defensive_offensive_rebound_percentage',
    'away_defensive_total_rebound_percentage',
    'away_defensive_total_foul_percentage',
    'away_defensive_three_point_attempt_rate',
    'away_defensive_free_throw_attempt_rate',
    'away_defensive_free_throw_rate',
    # 'neutral_defensive_offensive_rating',
    # 'neutral_defensive_pace',
    # 'neutral_defensive_two_point_percentage',
    # 'neutral_defensive_three_point_percentage',
    # 'neutral_defensive_free_throw_percentage',
    # 'neutral_defensive_assist_percentage',
    # 'neutral_defensive_block_percentage',
    # 'neutral_defensive_turnover_percentage',
    # 'neutral_defensive_offensive_rebound_percentage',
    # 'neutral_defensive_total_rebound_percentage',
    # 'neutral_defensive_total_foul_percentage',
    # 'neutral_defensive_three_point_attempt_rate',
    # 'neutral_defensive_free_throw_attempt_rate',
    # 'neutral_defensive_free_throw_rate'
]

REGRESSION_COLUMNS_Y = [
    'offensive_rating_true'
]

CLASSIFICATION_COLUMNS_X = [f'{x}_team_0' for x in REGRESSION_COLUMNS_X] + [f'{x}_team_1' for x in REGRESSION_COLUMNS_X]

CLASSIFICATION_COLUMNS_Y = [
    'winner_true'
    ]


#####

def get_team_game_logs(team_name, team_id, year):
    try:
        url_basic = f'https://www.sports-reference.com/cbb/schools/{team_id}/men/{year}-gamelogs.html'
        url_advanced = f'https://www.sports-reference.com/cbb/schools/{team_id}/men/{year}-gamelogs-advanced.html'

        team_game_logs_basic = pd.read_html(url_basic, header=1)[0][COLUMNS_BASIC.keys()]
        team_game_logs_advanced = pd.read_html(url_advanced, header=1)[0][COLUMNS_ADVANCED.keys()]

        team_game_logs_basic.rename(columns=COLUMNS_BASIC, inplace=True)
        team_game_logs_advanced.rename(columns=COLUMNS_ADVANCED, inplace=True)

        team_game_logs = pd.concat([team_game_logs_basic, team_game_logs_advanced], axis=1)

        team_game_logs = team_game_logs[(team_game_logs['result'] == 'W') | (team_game_logs['result'] == 'L')]

        team_game_logs['team'] = team_name
        team_game_logs['location'] = np.where(team_game_logs['location'] == '@', 'away', np.where(team_game_logs['location'] == 'N', 'neutral' , 'home'))

        team_game_logs = team_game_logs.astype({x: 'float64' for x in team_game_logs.columns if x not in ['team', 'location', 'opponent', 'type', 'result']})

        team_game_logs['total_foul_percentage'] = team_game_logs['fouls'] / (team_game_logs['fouls'] + team_game_logs['opponent_fouls'] + 1e-9)
        
        team_game_logs['opponent_assist_percentage'] = team_game_logs['opponent_assists'] / (team_game_logs['opponent_field_goals_made'] + 1e-9)
        team_game_logs['opponent_block_percentage'] = team_game_logs['opponent_blocks'] / (team_game_logs['two_pointers_attempted'] + 1e-9)
        team_game_logs['opponent_offensive_rebound_percentage'] = team_game_logs['opponent_offensive_rebounds'] / (team_game_logs['opponent_offensive_rebounds'] + team_game_logs['defensive_rebounds'] + 1e-9)
        team_game_logs['opponent_total_rebound_percentage'] = 1 - team_game_logs['total_rebound_percentage']
        team_game_logs['opponent_total_foul_percentage'] = 1 - team_game_logs['total_foul_percentage']
        team_game_logs['opponent_three_point_attempt_rate'] = team_game_logs['opponent_three_pointers_attempted'] / (team_game_logs['opponent_field_goals_attempted'] + 1e-9)
        team_game_logs['opponent_free_throw_attempt_rate'] = team_game_logs['opponent_free_throws_attempted'] / (team_game_logs['opponent_field_goals_attempted'] + 1e-9)

        team_game_logs = team_game_logs[COLUMNS]

    except:
        team_game_logs = pd.DataFrame()

    return team_game_logs


def get_regular_season_data(year=None):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Team Game Logs'

    if year:
        game_logs = pd.read_parquet(f'{directory}/{year}').reset_index(drop=True)
        game_logs['year'] = str(year)

    else:
        game_logs = pd.read_parquet(directory).rename(columns={'dir0': 'year'}).reset_index(drop=True)

    game_logs['opponent'] = game_logs['opponent'].apply(lambda x: TEAM_NAME_MAP[x] if x in TEAM_NAME_MAP else x)

    game_logs_regular_season = game_logs[game_logs['type'].isin(['REG (Conf)', 'REG (Non-Conf)', 'CTOURN'])].reset_index(drop=True)

    stats_regular_season = game_logs_regular_season[[x for x in game_logs_regular_season.columns if x not in ['opponent', 'location', 'type', 'result']]].groupby(['year', 'team'], observed=False).mean().reset_index()
    stats_regular_season_home = game_logs_regular_season[game_logs_regular_season['location'] == 'home'][[x for x in game_logs_regular_season.columns if x not in ['opponent', 'location', 'type', 'result']]].groupby(['year', 'team'], observed=False).mean().reset_index().rename(columns={x: f'home_{x}' for x in stats_regular_season.columns if x not in ['year', 'team']})
    stats_regular_season_away = game_logs_regular_season[game_logs_regular_season['location'] == 'away'][[x for x in game_logs_regular_season.columns if x not in ['opponent', 'location', 'type', 'result']]].groupby(['year', 'team'], observed=False).mean().reset_index().rename(columns={x: f'away_{x}' for x in stats_regular_season.columns if x not in ['year', 'team']})
    stats_regular_season_neutral = game_logs_regular_season[game_logs_regular_season['location'] == 'neutral'][[x for x in game_logs_regular_season.columns if x not in ['opponent', 'location', 'type', 'result']]].groupby(['year', 'team'], observed=False).mean().reset_index().rename(columns={x: f'neutral_{x}' for x in stats_regular_season.columns if x not in ['year', 'team']})

    offense_regular_season = stats_regular_season[[x for x in stats_regular_season.columns if 'opponent' not in x]]
    offense_home_regular_season = stats_regular_season_home[[x for x in stats_regular_season_home.columns if 'opponent' not in x]]
    offense_away_regular_season = stats_regular_season_away[[x for x in stats_regular_season_away.columns if 'opponent' not in x]]
    offense_neutral_regular_season = stats_regular_season_neutral[[x for x in stats_regular_season_neutral.columns if 'opponent' not in x]]

    defense_regular_season = stats_regular_season[[x for x in stats_regular_season.columns if 'opponent' in x or x in ['year', 'team', 'pace']]].rename(columns={'pace': 'opponent_pace'})
    defense_home_regular_season = stats_regular_season_home[[x for x in stats_regular_season_home.columns if 'opponent' in x or x in ['year', 'team', 'home_pace']]].rename(columns={'home_pace': 'home_opponent_pace'})
    defense_away_regular_season = stats_regular_season_away[[x for x in stats_regular_season_away.columns if 'opponent' in x or x in ['year', 'team', 'away_pace']]].rename(columns={'away_pace': 'away_opponent_pace'})
    defense_neutral_regular_season = stats_regular_season_neutral[[x for x in stats_regular_season_neutral.columns if 'opponent' in x or x in ['year', 'team', 'neutral_pace']]].rename(columns={'neutral_pace': 'neutral_opponent_pace'})

    defense_regular_season.columns = defense_regular_season.columns.str.replace('opponent', 'defensive')
    defense_home_regular_season.columns = defense_home_regular_season.columns.str.replace('opponent', 'defensive')
    defense_away_regular_season.columns = defense_away_regular_season.columns.str.replace('opponent', 'defensive')
    defense_neutral_regular_season.columns = defense_neutral_regular_season.columns.str.replace('opponent', 'defensive')

    return offense_regular_season, offense_home_regular_season, offense_away_regular_season, offense_neutral_regular_season, defense_regular_season, defense_home_regular_season, defense_away_regular_season, defense_neutral_regular_season, game_logs


def get_data(test_size=0.25, random_state=None):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Team Game Logs/'
    
    march_madness_seeds = pd.read_csv(f'{cwd}/March Madness/March Madness Seeds.csv')

    march_madness_seeds['team'] = march_madness_seeds['team'].apply(lambda x: TEAM_NAME_MAP[x] if x in TEAM_NAME_MAP else x)    

    offense_regular_season, offense_home_regular_season, offense_away_regular_season, offense_neutral_regular_season, defense_regular_season, defense_home_regular_season, defense_away_regular_season, defense_neutral_regular_season, game_logs = get_regular_season_data(year=None)

    game_logs_march_madness = game_logs[game_logs['type'].isin(['ROUND-64', 'ROUND-32', 'ROUND-16', 'ROUND-8', 'NATIONAL-SEMI', 'NATIONAL-FINAL'])].reset_index(drop=True)

    game_logs_march_madness = pd.merge(game_logs_march_madness, march_madness_seeds, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    game_logs_march_madness = pd.merge(game_logs_march_madness, march_madness_seeds, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))

    data = game_logs_march_madness[['year', 'team', 'opponent', 'seed', 'seed_dup', 'offensive_rating']].rename(columns={'seed_dup': 'opponent_seed', 'offensive_rating': 'offensive_rating_true'})

    if len(data[data['seed'].isna()]) > 0:
        print('seed')
        print(data[data['seed'].isna()])

        fail()

    if len(data[data['opponent_seed'].isna()]) > 0:
        print('opponent_seed')
        print(data[data['opponent_seed'].isna()])

        fail()

    data['seed'] = data['seed'].astype(int)
    data['opponent_seed'] = data['opponent_seed'].astype(int)

    data['game_id'] = data['year'].astype(str) + '_' + data[['team', 'opponent']].apply(lambda x: '_'.join(sorted(x)), axis=1)

    counts = pd.DataFrame(data['game_id'].value_counts())

    if len(counts[counts['count'] != 2]) > 0:
        print(counts)

        fail()

    data = pd.merge(data, offense_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    data = pd.merge(data, offense_home_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    data = pd.merge(data, offense_away_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    data = pd.merge(data, offense_neutral_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))

    data = pd.merge(data, defense_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))
    data = pd.merge(data, defense_home_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))
    data = pd.merge(data, defense_away_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))
    data = pd.merge(data, defense_neutral_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))

    data.drop(columns=[x for x in data.columns if '_dup' in x], inplace=True)

    game_ids = data['game_id'].unique()

    game_ids_train, game_ids_test = train_test_split(game_ids, test_size=test_size, random_state=random_state)

    data_train = data[data['game_id'].isin(game_ids_train)].sort_values('game_id')
    data_test = data[data['game_id'].isin(game_ids_test)].sort_values('game_id')

    return data_train, data_test


def read_data():
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Model Data'

    data_train = pd.read_parquet(f'{directory}/data_train.parquet')
    data_test = pd.read_parquet(f'{directory}/data_test.parquet')

    return data_train, data_test


def read_model(version):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Models'
    filename = f'{directory}/{version}.json'

    if 'regression' in version:
        model = xgb.XGBRegressor()

    elif 'classification' in version:
        model = xgb.XGBClassifier()

    model.load_model(filename)

    return model


def write_team_game_logs(team_name, team_id, year, verbose=True):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Team Game Logs/{year}'
    filename = f'{directory}/{team_name}.parquet'

    if not os.path.exists(directory):
        os.makedirs(directory)

    team_game_logs = get_team_game_logs(team_name=team_name, team_id=team_id, year=year)

    if not team_game_logs.empty:
        team_game_logs.to_parquet(filename)

        if verbose:
            print(f'{year} {team_name} Team Game Log Written Successfully')

    else:
        if verbose:
            print(f'{year} {team_name} Team Game Log Not Found')


def write_game_logs(year, verbose=True):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Team Game Logs/{year}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    written_teams = [x.split('.parquet')[0] for x in os.listdir(directory)]

    teams = pd.read_parquet(f'{cwd}/March Madness/College Basketball Teams.csv')

    teams = teams[(teams['From'] <= str(year)) & (teams['To'] >= str(year))]

    num_teams = len(teams)

    teams = teams[~teams['School'].isin(written_teams)]

    if len(teams) > 0:
        if verbose:
            print(f'Writing {len(teams)}/{num_teams} {year} Team Game Logs')

        teams = list(zip(teams['School'], teams['Team Id']))

        for team_name, team_id in teams:
            write_team_game_logs(team_name=team_name, team_id=team_id, year=year, verbose=verbose)

            time.sleep(4)

    if verbose:
        print(f'All {year} Team Game Logs Written')


def write_data(test_size=0.25, random_state=None, verbose=True):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Model Data'
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    data_train, data_test = get_data(test_size=test_size, random_state=random_state)

    data_train.to_parquet(f'{directory}/data_train.parquet')
    data_test.to_parquet(f'{directory}/data_test.parquet')

    if verbose:
        print(f'Model Data Written Successfully')


def write_model(model, version, verbose=True):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Models'
    filename = f'{directory}/{version}.json'

    if not os.path.exists(directory):
        os.makedirs(directory)

    model.save_model(filename)

    if verbose:
        print(f'Model Version: {version} Written Successfully')


def train_regression_model(data_train, verbose=True):
    x_train = data_train[REGRESSION_COLUMNS_X]
    y_train = data_train[REGRESSION_COLUMNS_X]

    model = xgb.XGBRegressor()

    param_grid = {
        'objective': ['reg:squarederror', 'reg:absoluteerror'],
        'n_estimators': [50, 100, 150, 200, 250, 500],
        # 'learning_rate': [0.01, 0.05, 0.1, 0.5],
        # 'max_depth': [4, 6, 8],
        # 'subsample': [0.7, 0.85, 1.0],
        # 'colsample_bytree': [0.7, 0.85, 1.0]
    }

    if verbose:
        print(f'Training Regression Model')

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error', refit=True, verbose=3)
    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_

    if verbose:
        print(f'Finished Training Regression Model')

    if verbose:
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Parameters: {grid_search.best_params_}')

    return model


def train_classification_model(data_train, verbose=True):
    data_train = pd.merge(data_train, data_train, how='left', left_on=['game_id', 'year', 'team', 'opponent', 'seed', 'opponent_seed'], right_on=['game_id', 'year', 'opponent', 'team', 'opponent_seed', 'seed'], suffixes=('_team_0', '_team_1'))
    
    data_train.drop(columns=[x for x in data_train.columns if x in ['team_team_1', 'opponent_team_1']], inplace=True)

    data_train['winner_true'] = np.where(data_train['offensive_rating_true_team_0'] > data_train['offensive_rating_true_team_1'], 0, 1)

    x_train = data_train[CLASSIFICATION_COLUMNS_X]
    y_train = data_train['winner_true']

    model = xgb.XGBClassifier()

    param_grid = {
        'objective': ['binary:logistic'],
        'n_estimators': [50, 100, 150, 200, 250, 500],
        # 'learning_rate': [0.01, 0.05, 0.1, 0.5],
        # 'max_depth': [4, 6, 8],
        # 'subsample': [0.7, 0.85, 1.0],
        # 'colsample_bytree': [0.7, 0.85, 1.0]
    }

    if verbose:
        print(f'Training Classification Model')

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_log_loss', refit=True, verbose=3)
    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_

    if verbose:
        print(f'Finished Training Classification Model')

    if verbose:
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Parameters: {grid_search.best_params_}')

    return model


def test_regression_models(data_test):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Models'

    if not os.path.exists(directory):
        os.makedirs(directory)

    versions = [x.split('.json')[0] for x in os.listdir(directory) if 'regression' in x]

    y_true = data_test[REGRESSION_COLUMNS_Y]

    maes = []
    rmses = []
    r2s = []
    was = []

    for version in versions:
        model = read_model(version)

        y_pred = model.predict(data_test[REGRESSION_COLUMNS_X])

        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)

        offensive_ratings = data_test[['game_id', 'offesnive_rating_true']]
        offensive_ratings['offensive_rating_pred'] = y_pred

        winner_correct = 0
        winner_wrong = 0

        for i in range(0, len(offensive_ratings), 2):
            if offensive_ratings['offensive_rating_true'].iloc[i] > offensive_ratings['offensive_rating_true'].iloc[i + 1]:
                if offensive_ratings['offensive_rating_pred'].iloc[i] > offensive_ratings['offensive_rating_pred'].iloc[i + 1]:
                    winner_correct += 1

                else:
                    winner_wrong += 1

            else:
                if offensive_ratings['offensive_rating_pred'].iloc[i + 1] > offensive_ratings['offensive_rating_pred'].iloc[i]:
                    winner_correct += 1

                else:
                    winner_wrong += 1

        wa = winner_correct / (winner_correct + winner_wrong + 1e-9)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        was.append(wa)

    result = pd.DataFrame([maes, rmses, r2s, was], index=['Mean Absolute Error', 'Root Mean Squared Error', 'R2 Score', 'Win Accuracy'], columns=[versions])
    
    print(f'Regression Results:\n{result}')

    return result


def test_classification_models(data_test):
    cwd = os.getcwd().split('/March Madness')[0]

    directory = f'{cwd}/March Madness/Models'

    if not os.path.exists(directory):
        os.makedirs(directory)

    versions = [x.split('.json')[0] for x in os.listdir(directory) if 'classification' in x]

    data_test = pd.merge(data_test, data_test, how='left', left_on=['game_id', 'year', 'team', 'opponent', 'seed', 'opponent_seed'], right_on=['game_id', 'year', 'opponent', 'team', 'opponent_seed', 'seed'], suffixes=('_team_0', '_team_1'))
    
    data_test.drop(columns=[x for x in data_test.columns if x in ['team_team_1', 'opponent_team_1']], inplace=True)

    data_test['winner_true'] = np.where(data_test['offensive_rating_true_team_0'] > data_test['offensive_rating_true_team_1'], 0, 1)

    y_true = data_test[CLASSIFICATION_COLUMNS_Y]

    aurocs = []
    lls = []
    was = []

    for version in versions:
        model = read_model(version)

        y_pred = model.predict(data_test[CLASSIFICATION_COLUMNS_X])
        y_score = model.predict_proba(data_test[CLASSIFICATION_COLUMNS_X])[:, 1]

        auroc = roc_auc_score(y_true=y_true, y_score=y_score)
        ll = log_loss(y_true=y_true, y_pred=y_pred)
        wa = accuracy_score(y_true=y_true, y_pred=y_pred)

        aurocs.append(auroc)
        lls.append(ll)
        was.append(wa)

    result = pd.DataFrame([aurocs, lls, was], index=['AUROC', 'Log Loss', 'Win Accuracy'], columns=[versions])
    
    print(f'Classification Results:\n{result}')

    return result


def test_models(data_test):
    regression_results = test_regression_models(data_test=data_test)
    classification_results = test_classification_models(data_test=data_test)

    results = pd.concat([regression_results, classification_results], axis=1)

    print(f'Results:\n{results}')

    return results


def predict_march_madness(year, version):
    cwd = os.getcwd().split('/March Madness')[0]

    model = read_model(version=version)

    bracket = pd.read_csv(f'{cwd}/March Madness/March Madness Seeds.csv')

    bracket['year'] = bracket['year'].astype(str)

    bracket = bracket[bracket['year'] == year]

    bracket['team'] = bracket['team'].apply(lambda x: TEAM_NAME_MAP[x] if x in TEAM_NAME_MAP else x)

    offense_regular_season, offense_home_regular_season, offense_away_regular_season, offense_neutral_regular_season, defense_regular_season, defense_home_regular_season, defense_away_regular_season, defense_neutral_regular_season, _ = get_regular_season_data(year=year)

    bracket = pd.merge(bracket, offense_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    bracket = pd.merge(bracket, offense_home_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    bracket = pd.merge(bracket, offense_away_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))
    bracket = pd.merge(bracket, offense_neutral_regular_season, how='left', left_on=['year', 'team'], right_on=['year', 'team'], suffixes=('', '_dup'))

    if len(bracket[bracket['home_offensive_rating'].isna()]) > 0:
        print(bracket[bracket['home_offensive_rating'].isna()])

        fail()

    rounds = ['Round of 64', 'Round of 32', 'Sweet Sixteen', 'Elite Eight', 'Final Four', 'Championship']

    count = 0

    print(f'March Madness {year}')

    while count < 6:
        opponents = []
        opponent_seeds = []

        for i in range(0, len(bracket), 2):
            opponents.append(bracket['team'].iloc[i + 1])
            opponents.append(bracket['team'].iloc[i])

            opponent_seeds.append(bracket['seed'].iloc[i + 1])
            opponent_seeds.append(bracket['seed'].iloc[i])

        bracket['opponent'] = opponents
        bracket['opponent_seed'] = opponent_seeds

        bracket = pd.merge(bracket, defense_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))
        bracket = pd.merge(bracket, defense_home_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))
        bracket = pd.merge(bracket, defense_away_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))
        bracket = pd.merge(bracket, defense_neutral_regular_season, how='left', left_on=['year', 'opponent'], right_on=['year', 'team'], suffixes=('', '_dup'))

        bracket.drop(columns=[x for x in bracket.columns if '_dup' in x], inplace=True)
        
        bracket['offensive_rating_pred'] = model.predict(bracket[COLUMNS_X])

        print(f'\n{rounds[count]}\n')

        winners = []

        for i in range(0, len(bracket), 2):
            if bracket['offensive_rating_pred'].iloc[i] > bracket['offensive_rating_pred'].iloc[i + 1]:
                winner = bracket['team'].iloc[i]
                winner_seed = bracket['seed'].iloc[i]

            else:
                winner = bracket['team'].iloc[i + 1]
                winner_seed = bracket['seed'].iloc[i + 1]

            winners.append(winner)

            print(f'{bracket['team'].iloc[i]} ({bracket['seed'].iloc[i]}) vs. {bracket['team'].iloc[i + 1]} ({bracket['seed'].iloc[i + 1]}): {int(bracket['offensive_rating_pred'].iloc[i])}-{int(bracket['offensive_rating_pred'].iloc[i + 1])} {winner} ({winner_seed})')

        bracket = bracket[bracket['team'].isin(winners)].reset_index(drop=True)

        bracket.drop(columns=[x for x in bracket.columns if 'defensive' in x], inplace=True)

        count += 1

#####


if __name__ == '__main__':   
    random_state = 0
    verbose = True

    test_size = 0.25

    write_game_logs(year='2010', verbose=verbose)

    write_data(test_size=test_size, random_state=random_state, verbose=verbose)

    data_train, data_test = read_data()

    print(f'Data Set Size: {len(data_train) + len(data_test)}')
    print(f'Training Size: {len(data_train)}')
    print(f'Test Size: {len(data_test)}')

    regression_model = train_regression_model(data_train, verbose=verbose)

    write_model(model=regression_model, version='regression', verbose=verbose)

    classification_model = train_classification_model(data_train, verbose=verbose)

    write_model(model=classification_model, version='classification', verbose=verbose)

    test_models(data_test=data_test)

    predict_march_madness(year='2025', version=version)

    
