import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import json
import argparse
from pprint import pprint
from get_log_time import get_start_end_type_log_tags, get_durations, get_log_tags_dataframe, tag_value_to_name, tag_name_to_value

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
                                "msg_id", "extra", "0ull"],
                         header=None)
        log_data.append(df)
    combined_df = pd.concat(log_data, ignore_index=True)
    return combined_df

def trim_df(df, start_loc, end_loc):
     df = df.iloc[start_loc:end_loc]
     return df

def clean_log_dataframe(log_data):
    df = pd.DataFrame(log_data, columns=["tag", "timestamp", "client_node_id", "msg_id", "extra", "0ull"])
    df = df.drop(columns = ["0ull"])
    df['tag'] = df['tag'].astype('uint64')
    df['client_node_id'] = df['client_node_id'].astype('uint64')
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
        df['msg_id'] = converted.astype('uint64')

    df['extra'] = df['extra'].astype('uint64')
         
    return df

def add_query_id_node_id_dataframe(df):
    log_tags_df = get_log_tags_dataframe()
    tag_type_dict = dict()
    for index, row in log_tags_df.iterrows():
        tag_type_dict[row["tag_value"]] = str(row["msg_id_type"])
        print(row["tag_value"], row["msg_id_type"])

    abnormal_rows = []
    def get_query_id(row):
        tag_type = tag_type_dict[row["tag"]]
        msg_id = np.uint64(row["msg_id"])  
        if tag_type == "query_id":
            return msg_id
        elif tag_type == "query_id_and_node_id":
            # print(row, msg_id // MULTIPLIER)
            if (msg_id // MULTIPLIER) > 10000:
                print(msg_id // MULTIPLIER, int(row["msg_id"]))
            return msg_id // MULTIPLIER
        elif tag_type == "not_query_id_and_node_id":
            msg_id = msg_id
            return (msg_id) // MULTIPLIER
        else:
            return np.nan

    def get_node_id(row):
        tag_type = tag_type_dict[row["tag"]]
        # msg_id = int(row["msg_id"]
        if tag_type == "query_id":
            return np.nan
        elif tag_type == "query_id_and_node_id":
            # print(row, int(msg_id % MULTIPLIER), msg_id)
            if (row["msg_id"] % MULTIPLIER) > 10000000:
                print(row["msg_id"] % MULTIPLIER, row["msg_id"])
                # abnormal_rows.append(row)
            return row["msg_id"] % MULTIPLIER
        elif tag_type == "not_query_id_and_node_id":
            return row["msg_id"] % MULTIPLIER
        else:
            return np.nan

    def get_batch_id(row):
        tag_type = tag_type_dict[row["tag"]]
        if tag_type == "batch_id":
            return row["msg_id"]
        else:
            return np.nan
        
    df["query_id"] = df.apply(get_query_id, axis = 1)
    df["node_id"] = df.apply(get_node_id, axis = 1)
    # node id is the node id of the node requested in compute query and in compute results
    df["batch_id"] = df.apply(get_batch_id, axis = 1)
    # for batches
    return df, abnormal_rows

def get_msg_id_to_batch_id_dict(df, tag_fn):
    relevant_tags = [
        tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_PUSH_BATCHER"), 
        tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_PUSH_BATCHER")
    ]
    data_df = df[df["tag"].isin(relevant_tags)]
    
    # Ensure uint64 types in dictionary
    return dict(zip(data_df["msg_id"].astype('uint64'), data_df["extra"].astype('uint64')))

def add_batch_id_to_all_compute(df, msg_id_to_batch_id_dict):
    """
    for each tag involving a compute task/result, match its msg_id to a batch_id
    """
    # Convert NaN to 0 first, then to uint64
    df["batch_id"] = df["batch_id"].fillna(0).astype('uint64')
    
    existing_values_mask = (df["batch_id"] != 0)  # No need for notna() check now
    expected_values = df["msg_id"].map(msg_id_to_batch_id_dict).fillna(0).astype('uint64')
    
    # Now check conflicts with same types
    has_expected = expected_values != 0  # Rows where we have a mapping
    print("number of existing batch id", has_expected.sum())
    conflicts = df[existing_values_mask & has_expected & (df["batch_id"] != expected_values)]

    if not conflicts.empty:
        print(f"{len(conflicts)} Conflicts found:")
        pprint(conflicts.head())
    else:
        print("No conflicts - all existing batch_ids match the dictionary")

    # Update only rows that need it (currently 0) and have a mapping
    needs_update = df["batch_id"] == 0
    update_mask = needs_update & has_expected
    
    df.loc[update_mask, "batch_id"] = expected_values.loc[update_mask]
    return conflicts

def print_summary_start_series(s):
    print("count" , s.count())
    print("mean: ", s.mean())
    print("std: ", s.std())
    print("90p:  " , s.quantile(0.9))
    print("99p: " , s.quantile(0.99))


def misc_times(df, tag_fn):
    """
    time for a read, time to do 1 step of greedy search

    """
    print("time to read a node in compute thread")
    compute_read_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_READ_START"), tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_READ_END") , ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(compute_read_df["latency"])

    print("time to read a node in search thread")
    search_read_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_READ_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_READ_END") , ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(search_read_df["latency"])

    print("time to do one step in greedy search")
    search_step_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_STEP_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_STEP_END") , ['client_node_id', 'query_id', 'node_id'])
    # print(search_step_df)
    print_summary_start_series(search_step_df["latency"])

    
def compute_task_times(df, tag_fn):
    print("round trip latency of compute query")    
    roundtrip_durations = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_SEND"), tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_RECEIVE"),  ['client_node_id', 'query_id', 'node_id'])
    print_summary_start_series(roundtrip_durations["latency"])
    

    print("compute query duration from send request to start of serialization of that batch on batching thread")
    query_send_to_serialize_durations = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_SEND"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_PUSH_BATCHER"),  ['client_node_id', "query_id", "node_id"])
    print_summary_start_series(query_send_to_serialize_durations["latency"])

    print("batch serialization time between udls for global search")
    batch_serialization_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_SERIALIZE_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_SERIALIZE_END"),  ["batch_id"])
    print_summary_start_series(batch_serialization_time["latency"])

    print("batch sending latency between udls for global search")
    batch_send_latency = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_SEND_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_START"),  ["batch_id"])
    batch_send_latency["batch_id"] = batch_send_latency["batch_id"].apply(lambda row: row[0])

    # batch_id = 0 we don't give a fuck about because its not batch sending between udls
    batch_send_latency = batch_send_latency[batch_send_latency["batch_id"] != 0] 
    print_summary_start_series(batch_send_latency["latency"])

    print("time to complete a put_and_forget")
    put_and_forget_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_SEND_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_SEND_END"), ["batch_id"])
    print_summary_start_series(put_and_forget_time["latency"])

    print("time to transfer messages from cluster messages to to_send")
    transfer_messages_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCHING_TRANSFER_MESSAGES_START"), tag_fn("LOG_GLOBAL_INDEX_BATCHING_TRANSFER_MESSAGES_END"), ["msg_id"])
    print_summary_start_series(transfer_messages_time["latency"])

    print("time to prep a batch for serialization for compute results and queries")
    prep_batch_serialize_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_PREP_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_PREP_END"), ["batch_id"])
    print_summary_start_series(prep_batch_serialize_time["latency"])

    print("time to deserialize a batch")
    batch_deserialize_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_END"), ["batch_id"])
    batch_deserialize_time["batch_id"] = batch_deserialize_time["batch_id"].apply(lambda row: row[0])

    # batch_id = 0 we don't give a fuck about because its not batch sending between udls
    batch_deserialize_time = batch_deserialize_time[batch_deserialize_time["batch_id"] != 0] 
    print_summary_start_series(batch_deserialize_time["latency"])

    print("time from compute query pushed to queue to it being popped off to start compute")
    query_pushed_to_start_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_QUEUE_PUSHED"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_START"), ["client_node_id", "query_id", "node_id"])
    print_summary_start_series(query_pushed_to_start_time["latency"])

    print("time to complete compute query")
    compute_query_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_START"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_END"), ["client_node_id", "query_id", "node_id"])
    print_summary_start_series(compute_query_time["latency"])

    print("time to push a compute result")
    compute_result_push_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_PUSH_TO_BATCHING_START"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_PUSH_TO_BATCHING_END"), ["client_node_id", "query_id", "node_id"])
    print_summary_start_series(compute_result_push_time["latency"])

    print("time from compute query complettion to result starting to be serialized in batching thread")
    compute_result_to_serialize_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_END"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_PUSH_BATCHER"), ["client_node_id", "query_id", "node_id"])
    print_summary_start_series(compute_result_to_serialize_time["latency"])
    
    # return roundtrip_durations, query_send_to_serialize_durations, batch_serialization_time, batch_send_latency, put_and_forget_time,transfer_messages_time, prep_batch_serialize_time, batch_serialization_time,


    # print("compute result duration from done calc to start of serializaion")
    # result_send_to_serialize_duration = get_durations_test(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_END"), tag_fn("LOG_GLOBAL_INDEX_BATCH_SERIALIZE_START"),  ['batch_id'])
    # print_summary_start_series(result_send_to_serialize_duration["latency"])

    # print("time to push compute result")
    # result_push_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_PUSH_TO_BATCHING_START"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_PUSH_TO_BATCHING_END"), ['client_node_id', "query_id", "node_id"])
    # print_summary_start_series(result_push_time["latency" ])

    # print("time to prep compute query and result")
    # prep_batch_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_PREP_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_PREP_END"), ["batch_id"])
    # print_summary_start_series(prep_batch_time["latency"])

    # print("time for a trigger put batch")
    # trigger_put_batch_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_SEND_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_SEND_END"), ["batch_id"])
    # print_summary_start_series(trigger_put_batch_time["latency"])

    # print("sending latency for a batch")
    # batch_sending_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_SEND_END"), tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_START"), ["batch_id"])
    # print_summary_start_series(batch_sending_time["latency"])

    # print("time to transfer messages")
    # transfer_messages_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCHING_TRANSFER_MESSAGES_START"), tag_fn("LOG_GLOBAL_INDEX_BATCHING_TRANSFER_MESSAGES_END"), ["msg_id"])
    # print_summary_start_series(transfer_messages_time["latency"])



    # print("time to deserialize a batch")
    # batch_deserialize_time = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_START"), tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_END"), ["batch_id"])
    # print_summary_start_series(batch_deserialize_time["latency"])

    # print("time from deserialization to results being received")
    # deserialization_to_received = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_BATCH_DESERIALIZE_END"),tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_RECEIVE"), ["batch_id"])
    # print_summary_start_series(deserialization_to_received["latency"])
    # # print("compute query serialization time on batching thread")
    # # query_send_to_serialize_duration = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_SERIALIZATION_START"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_SERIALIZATION_END"),  ['client_node_id', 'query_id', 'node_id'])
    # # print_summary_start_series(query_send_to_serialize_duration["latency"])

    # # print("compute query done serialization to query recevied on handler")


    # #need to process batch data
    # # for each compute query, there is batch id for the extra column
    
if __name__ == "__main__":     
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()

    log_folder: str = args.path
    log_files = get_log_files(log_folder, suffix)
    df = get_log_files_dataframe(log_files)

    log_tags_df = get_log_tags_dataframe()
    tag_fn = lambda tag_name: tag_name_to_value(log_tags_df, tag_name)
    
    df = clean_log_dataframe(df)
    df, abnormal_rows = add_query_id_node_id_dataframe(df)

    msg_id_to_batch_id_dict = get_msg_id_to_batch_id_dict(df, tag_fn)
    # print(msg_id_to_batch_id_dict)
    conflicts = add_batch_id_to_all_compute(df, msg_id_to_batch_id_dict)
    pprint(df)
    compute_task_times(df, tag_fn)
    misc_times(df, tag_fn)
    
    # print("round trip latency of compute task")    
    # data_df = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_SEND"), tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_RECEIVE"),  ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print("time from sending to compute task to it  being pushed to remote cluster compute task queue")
    # data_df = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_SEND"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_QUEUE_PUSHED"),  ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print(" time from it being pushed to it being computed")
    # data_df = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_QUERY_QUEUE_PUSHED"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_START") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print("time from the computation done to the requesting thread actually receiving the results")
    # data_df = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_END"), tag_fn("LOG_GLOBAL_INDEX_SEARCH_COMPUTE_RECEIVE") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print("compute results serialization time")
    # data_df = get_durations(df, tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_SERIALIZATION_START"), tag_fn("LOG_GLOBAL_INDEX_COMPUTE_RESULT_SERIALIZATION_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # # need to calculate how late each result arrives compared to when the query finishes
    # print("time from query search ending to the results arriving")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_END"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_COMPUTE_RECEIVE") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])

    # print("time to finish a global search")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_END") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])

    # print("time to finish a compute task")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_START"), tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print("Below, we will examine the time for each individual components when executing a compute task")

    # print("time to setup the scratch")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_GET_SCRATCH_START"), tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_GET_SCRATCH_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print("time to prepare the query")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_PREP_QUERY_START"), tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_PREP_QUERY_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])


    # print("time to compute a distance")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_CALC_START"), tag_fn( "LOG_GLOBAL_INDEX_COMPUTE_CALC_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    
    # print("Below, we will examine the time for each individual components when doing a global search")

    # print("Time to get scratch")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_GET_SCRATCH_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_GET_SCRATCH_START") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])

    # print("Time to prep the query")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_PREP_QUERY_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_PREP_QUERY_END") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])

    # print("Time to initialize the candidate queue") # we can optimize this 
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_INIT_DISTANCES_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_INIT_DISTANCES_END") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])


    # print("time to read a node")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_READ_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_READ_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])

    # print("time to compute distance for a node")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_COMPUTE_DIST_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_COMPUTE_DIST_END") , ['client_node_id', 'query_id', 'node_id'])
    # print_summary_start_series(data_df["latency"])
    
    # print("time for the search")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_SEARCH_SEARCH_START"), tag_fn( "LOG_GLOBAL_INDEX_SEARCH_SEARCH_END") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])

    # print("global index deserialize time")
    # data_df = get_durations(df, tag_fn( "LOG_GLOBAL_INDEX_UDL_DESERIALIZE_START"), tag_fn("LOG_GLOBAL_INDEX_UDL_DESERIALIZE_END") , ['client_node_id', 'query_id'])
    # print_summary_start_series(data_df["latency"])
