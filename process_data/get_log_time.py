'''
This file has functions used to get/plot the time from the events listed in ../src/log_tags.csv, given a dataframe processed by process_data.py

plots should be distributions, print should be avg + std
'''
import pandas as pd
from pprint import pprint
LOG_TAG_PATH = "../src/log_tags.csv"
END_SUFFIX_LIST = ["_END", "_RECEIVE"] 
START_SUFFIX_LIST = ["_START", "_SEND"]

def get_log_tags_dataframe():
    log_tags_df = pd.read_csv(LOG_TAG_PATH, comment="#")
    return log_tags_df

def is_tag_start(tag: str):
    """
    given a string/tag, determine if its a start tag based on START_SUFFIX_LIST.
    If it is a start tag, then return the index of the suffix in START_SUFFIX_LIST
    Else: return -1
    """
    for i, suffix in enumerate(START_SUFFIX_LIST):
        if tag.endswith(suffix):
            return i
    return -1

def tag_value_to_name(tags_df, tag_value: int):
    return tags_df[tags_df["tag_value"] == tag_value]["tag_name"].item()

def tag_name_to_value(tags_df, tag_name: str):
    return tags_df[tags_df["tag_name"] == tag_name]["tag_value"].item()

def get_corresponding_end_tag(start_tag:str):
    """
    precondition is that start_tag must end with one of the suffixes in END_SUFFIX_LIST,
    return the corresponding end_tag as defined in END_SUFFIX_LIST
    """
    tag_suffix_id = is_tag_start(start_tag)
    assert(tag_suffix_id >= 0)
    start_suffix = START_SUFFIX_LIST[tag_suffix_id]
    end_suffix = END_SUFFIX_LIST[tag_suffix_id]
    end_tag = start_tag[:-len(start_suffix)] + end_suffix
    print(end_tag)
    return end_tag

def get_start_end_type_log_tags(log_tags_df):
    tag_list = log_tags_df["tag_name"].tolist()
    pprint(tag_list)
    start_end_tags = dict()
    for tag in tag_list:
        if tag not in start_end_tags:
            if is_tag_start(tag) >= 0:
                end_tag = get_corresponding_end_tag(tag)
                assert(end_tag in tag_list)
                start_end_tags[tag] = end_tag
    start_end_type_list = []
    for start_tag, end_tag in start_end_tags.items():
        tag_type = log_tags_df[log_tags_df["tag_name"] == start_tag]["msg_id_type"].item()
        start_tag_id = log_tags_df[log_tags_df["tag_name"] == start_tag]["tag_value"].item()
        end_tag_id = log_tags_df[log_tags_df["tag_name"] == end_tag]["tag_value"].item()
        start_end_type_list.append((start_tag_id, end_tag_id, tag_type))
                        
    return start_end_type_list



def get_durations(log_df, start_tag, end_tag, group_by_columns=['client_node_id'],duration_name='latency'):
     filtered_df = log_df[(log_df['tag'] == start_tag) | (log_df['tag'] == end_tag)]
     grouped = filtered_df.groupby(group_by_columns)['timestamp']
     duration_results = []
     num_malformed = 0
     for group_values, timestamps in grouped:
          latency = timestamps.max() - timestamps.min()
          # timestamps.sort_values()
          # num_timestamps = len(timestamps)
          # for i in range(0, num_timestamps - 1):
              # latency = timestamps.max() - timstamps
          if (len(timestamps) != 2):
              # print("len timestamp is " , len(timestamps))
              # print("tag is" , group_values, start_tag, end_tag, filtered_df.loc[timestamps.index[0]])
              # len_timestamps_list.append(len(timestamps))
              num_malformed += 1
              continue
          if len(group_by_columns) > 1:
               result = {group_by_column: value for group_by_column, value in zip(group_by_columns, group_values)}
               result[duration_name] = latency
               result["timestamp"] = timestamps.min()
               duration_results.append(result)
          else:
               duration_results.append({group_by_columns[0]: group_values, duration_name: latency, "timestamp": timestamps.min()})
     duration_df = pd.DataFrame(duration_results)
     if (num_malformed != 0):
         print("num_malformed", num_malformed)
     # print(f"{duration_name} duration size",len(duration_df))
     # print("num malformed", num_malformed)
     # print(pd.Series(len_timestamps_list).describe())
     return duration_df



if __name__ == "__main__":
    pass
        

