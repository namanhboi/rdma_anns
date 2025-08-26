import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import json
import argparse
from pprint import pprint
from get_log_time import get_start_end_type_log_tags, get_durations, get_log_tags_dataframe, tag_value_to_name

warnings.filterwarnings("ignore")
suffix = ".dat"

MULTIPLIER = 1 << 32

def get_dfg_file(local_dir):
    os.path.join(local_dir, "dfgs.json.tmp")
    return os.path.join(local_dir, "dfgs.json.tmp")


# need to modify it for our needs
def get_dfg_information(dfg_file):
    information = {}
    with open(dfg_file) as f:
        dfg_data = json.load(f)
    information['top_num_centroids'] = dfg_data[0]["graph"][0]["user_defined_logic_config_list"][0]["top_num_centroids"]
    return information


def get_log_files(local_dir, suffix):
    log_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file[-4:] == suffix:
                file_path = os.path.join(root, file)
                log_files.append(file_path)
    return log_files


def get_log_files_dataframe(log_files):
    log_data = []
    for log_file_path in log_files:
        df = pd.read_csv(log_file_path,
                         delim_whitespace=True,
                         comment='#',
                         names=["tag", "timestamp", "client_node_id",
                                "msg_id", "cluster_id", "extra"],
                         header=None)
        log_data.append(df)
    combined_df = pd.concat(log_data, ignore_index=True)
    return combined_df

def trim_df(df, start_loc, end_loc):
     df = df.iloc[start_loc:end_loc]
     return df

def clean_log_dataframe(log_data, drop_warmup=0):
    df = pd.DataFrame(log_data, columns=["tag", "timestamp", "client_node_id", "msg_id","cluster_id", "extra"])
    df = df.drop(columns=['extra'])
    df['tag'] = df['tag'].astype(int)
    df['client_node_id'] = df['client_node_id'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = df['timestamp']/1000 # convert to microseconds
    converted = pd.to_numeric(df['msg_id'], errors='coerce')

    problematic_rows = df[converted.isna()]

    if not problematic_rows.empty:
        pass
        print("Problematic rows:")
        for idx in problematic_rows.index:
            print(f"Row {idx}: msg_id = '{df.loc[idx, 'msg_id']}'")
    else:
        df['msg_id'] = converted.astype(int)

    df['cluster_id'] = df['cluster_id'].astype(int)
         
    df = df[df['msg_id'] >= drop_warmup ]
         
    return df

def add_query_id_node_id_dataframe(df):
    log_tags_df = get_log_tags_dataframe()
    tag_type_dict = dict()
    for index, row in log_tags_df.iterrows():
        tag_type_dict[row["tag_value"]] = str(row["msg_id_type"])
        print(row["tag_value"], row["msg_id_type"])

    df["query_id"] = df.apply(lambda row: row["msg_id"] if tag_type_dict[row["tag"].item()] == "query_id" else (row["msg_id"] //MULTIPLIER), axis = 1)
    df["node_id"] = df.apply(lambda row: pd.NA if tag_type_dict[row["tag"].item()] == "query_id" else (row["msg_id"] % MULTIPLIER), axis = 1)
    return df


def print_summary_start_series(s):
    print("Mean " , s.mean())
    print("std " , s.std())
    print("90p " , s.quantile(0.9))
    print("99p " , s.quantile(0.99))
    
if __name__ == "__main__":     
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()

    log_folder: str = args.path
    log_files = get_log_files(log_folder, suffix)
    df = get_log_files_dataframe(log_files)
    # pprint(df)
    df = clean_log_dataframe(df)
    df = add_query_id_node_id_dataframe(df)
    log_tags_df = get_log_tags_dataframe()
    start_end_type_tags = get_start_end_type_log_tags(log_tags_df)

    # for start_tag, end_tag, tag_type in start_end_type_tags:
    #     data_df = None
    #     if tag_type == "query_id":
    #         data_df = get_durations(df, start_tag, end_tag, tag_type, ['client_node_id', 'query_id'])
    #     elif tag_type == "query_id_and_node_id":
    #         data_df = (get_durations(df, start_tag, end_tag, tag_type, ['client_node_id', 'query_id', 'node_id']))

    #     if tag_value_to_name(log_tags_df, start_tag).startswith("LOG_GLOBAL_INDEX_SEARCH_COMPUTE"):
    #         print("===", tag_value_to_name(log_tags_df, start_tag), tag_value_to_name(log_tags_df, end_tag), "===")
    #         print_summary_start_series(data_df["latency"])

    #     if tag_value_to_name(log_tags_df, start_tag).startswith("LOG_GLOBAL_INDEX_SEARCH_READ"):
    #         print("===", tag_value_to_name(log_tags_df, start_tag), tag_value_to_name(log_tags_df, end_tag), "===")
    #         print_summary_start_series(data_df["latency"])

    #     if tag_value_to_name(log_tags_df, start_tag).startswith("LOG_GLOBAL_INDEX_COMPUTE_READ"):
    #         print("===", tag_value_to_name(log_tags_df, start_tag), tag_value_to_name(log_tags_df, end_tag), "===")
    #         print_summary_start_series(data_df["latency"])


    print("round trip latency of compute task")
    data_df = get_durations(df, 20120, 20121, "query_id_and_node_id", ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(data_df["latency"])
    
    print(" time from sending to it being pushed to remote cluster compute task queue")
    data_df = get_durations(df, 20120, 20040, "query_id_and_node_id", ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(data_df["latency"])

    print(" time from it being pushed to it being computed")
    data_df = get_durations(df, 20040, 20020 ,"query_id_and_node_id", ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(data_df["latency"])

    print("time from the computation done to the requesting thread actually receiving the results")
    data_df = get_durations(df, 20021, 20121 ,"query_id_and_node_id", ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(data_df["latency"])

    print("compute results serialization time")
    data_df = get_durations(df, 20072, 20073 ,"query_id_and_node_id", ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(data_df["latency"])

